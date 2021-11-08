import sys
sys.path.append('../')

import DDPG
import OurDDPG
import TD3
import utils
import numpy as np
import torch
import argparse
import os
import random

from common import make_env
from evals import *

default_timestep = 0.02
default_frame_skip = 2


# Main function of the policy. Model is trained and evaluated inside.
def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          save_model=False, load_model="", jit_duration=0.02, g_ratio=1, response_rate=0.08, std_eval=False,
          catastrophe_frequency=1, reflex_response_rate=0.02):

    delayed_env = True
    hori_force = g_ratio * 9.81
    env_name = 'InvertedPendulum-v2'
    eval_policy = eval_policy_std if std_eval else eval_policy_ori
    arguments = [policy, 'reflex', env_name, seed, jit_duration, g_ratio, response_rate, catastrophe_frequency, reflex_response_rate]
    file_name = '_'.join([str(x) for x in arguments])
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if response_rate % default_timestep == 0:
        frame_skip = response_rate / default_timestep
        timestep = default_timestep
    elif jit_duration < response_rate:
        timestep = jit_duration
        frame_skip = response_rate / timestep
    else:
        timestep = response_rate
        frame_skip = 1

    jit_frames = 0  # How many frames the horizontal jitter force lasts each time
    if jit_duration:
        if jit_duration % timestep == 0:
            jit_frames = int(jit_duration / timestep)
        else:
            raise ValueError(
                "jit_duration should be a multiple of the timestep: " + str(timestep))

    reflex_frames = int(reflex_response_rate/timestep)

    print('timestep:', timestep)  # How long does it take before two frames
    # How many frames to skip before return the state, 1 by default
    print('frameskip:', frame_skip)

    # The ratio of the default time consumption between two states returned and reset version.
    # Used to reset the max episode number to guarantee the actual max time is always the same.
    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)

    print('time change factor', time_change_factor)
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_timesteps = max_timesteps * time_change_factor
    eval_freq = int(eval_freq * time_change_factor)
    start_timesteps = start_timesteps * time_change_factor

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "observation_space": env.observation_space,
        "delayed_env": delayed_env,
        "reflex": True
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3.TD3(**kwargs)
    elif policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if load_model != "":
        policy_file = file_name if load_model == "default" else load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim+action_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor,
                               jit_duration=jit_duration, env_timestep=timestep, force=hori_force, frame_skip=frame_skip,
                               jit_frames=jit_frames, delayed_env=delayed_env, reflex_frames=reflex_frames)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    if jit_duration:
        counter = 0
        disturb = random.randint(50, 100) * 0.04 * (1/catastrophe_frequency)
        print("==> Using Horizontal Jitter!")

    jittering = False
    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            if not reflex_frames:
                action = policy.select_action(state)
            else:
                reflex, action = policy.select_action(state)
        else:
            if not reflex_frames:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            else:
                reflex, a = policy.select_action(np.array(state))
                action = (a + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Perform action
        if jit_duration:
            if not jittering and round(disturb - counter, 2) >= response_rate:  # Not during the frames when jitter force keeps existing
                next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
                counter += response_rate

                # print(next_state)
            elif not jittering and round(disturb - counter, 2) < response_rate:
                jitter_force = np.random.random() * hori_force * (2 * (np.random.random() > 0.5) - 1)  # Jitter force strength w/ direction
                frames_simulated = 0
                force_frames_simulated = 0
                if reflex and (disturb - counter) / timestep >= reflex_frames:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, 0, reflex_frames, 0)
                    frames_simulated += reflex_frames
                elif reflex and (disturb - counter) / timestep < reflex_frames:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, 0,
                                                                           (disturb - counter) / timestep, 0)
                    next_state, reward, done, _ = env.jitter_step_end(reflex, jitter_force, reflex_frames - (
                                (disturb - counter) / timestep), 0)
                    frames_simulated += reflex_frames
                    force_frames_simulated += reflex_frames - ((disturb - counter) / timestep)

                next_state, reward, done, _ = env.jitter_step_start(action, jitter_force, max(
                    ((disturb - counter) / timestep) - frames_simulated, 0), (frame_skip - ((disturb - counter) / timestep)) - frames_simulated, jit_frames - force_frames_simulated)

                jittered_frames = frame_skip - ((disturb - counter)/timestep)
                if jittered_frames >= jit_frames:
                    jittered_frames = 0
                    jittering = False
                    env.model.opt.gravity[0] = 0
                    counter = 0
                    disturb = random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency)
                else:
                    jittering = True
                    env.model.opt.gravity[0] = jitter_force
                    counter += response_rate

            elif jit_frames - jittered_frames < frame_skip:  # Jitter force will dispear from now!
                frames_simulated = 0
                if reflex:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, jitter_force, reflex_frames, 0)
                    frames_simulated += reflex_frames
                next_state, reward, done, _ = env.jitter_step_end(
                    action, jitter_force, jit_frames - jittered_frames - frames_simulated,
                                          frame_skip - (jit_frames - jittered_frames))
                jittering = False  # Stop jittering now
                disturb = random.randint(50, 100) * 0.04 * (1/catastrophe_frequency) # Define the next jittering frame
                env.model.opt.gravity[0] = 0
                jittered_frames = 0
                counter = 0
            else:  # Jitter force keeps existing now!
                next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
                jittered_frames += frame_skip
                counter += response_rate
                if jittered_frames == jit_frames:
                    jittering = False
                    disturb = random.randint(50, 100) * 0.04 * (1/catastrophe_frequency)
                    env.model.opt.gravity[0] = 0
                    counter = 0
        else:
            next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
            counter += response_rate
        done_bool = float(done) if episode_timesteps < env.env.env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        counter = round(counter, 2)

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if jit_duration:
                counter = 0
                env.model.opt.gravity[0] = 0
                disturb = random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency)
                jittering = False
                jittered_frames = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(
                eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor,
                            jit_duration=jit_duration, env_timestep=timestep, force=hori_force, frame_skip=frame_skip,
                            jit_frames=jit_frames, delayed_env=delayed_env, reflex_frames=reflex_frames))
            np.save(f"./results/{file_name}", evaluations)
            # if save_model:
            #     policy.save(f"./models/{file_name}_{t}")

        if jit_duration:
            if counter == disturb:  # Execute adding jitter horizontal force here
                jitter_force = np.random.random() * hori_force * \
                    (2*(np.random.random() > 0.5)-1)  # Jitter force strength w/ direction
                env.model.opt.gravity[0] = jitter_force
                jittering = True
                jittered_frames = 0

    if save_model:
        policy.save(f"./models/{file_name}_final")
        # if t >= 25000:
        #     env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--start_timesteps", default=25e3, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=5e3, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1e5, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, help="Discount factor")
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--load_model", default="", help="Model load file name, `` doesn't load, `default` uses file_name")
    parser.add_argument("--jit_duration", default=0.04, type=float, help="Duration in seconds for the horizontal force")
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.04, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--std_eval", action="store_true", help="Use standard evaluation or original evaluation policy")
    parser.add_argument("--catastrophe_frequency", default=1.0, type=float, help="Modify how often to apply catastrophe")


    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
