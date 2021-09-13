import numpy as np
import torch
import gym
from gym.envs.mujoco import inverted_pendulum
import argparse
import os
import random

import sys

sys.path.append('../')
import utils
import TD3
import OurDDPG
import DDPG

default_timestep = 0.02
default_frame_skip = 2

def jitter_step(self, a, force, frames1, frames2):
    self.model.opt.gravity[0] = force
    reward = 1.0
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = force
    self.do_simulation(a, int(frames2))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}


inverted_pendulum.InvertedPendulumEnv.jitter_step = jitter_step


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10, time_change_factor=1, jit_duration=0, env_timestep=0.02, force=1,
                frame_skip=1, jit_frames=0):
    eval_env = make_env(env_name, 100, time_change_factor, env_timestep, frame_skip)

    avg_reward = 0.
    if jit_duration:
        counter = 1
        disturb = random.randint(50, 100) * time_change_factor
        print("==> Using Horizontal Jitter!")

    jittering = False
    t = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            # Perform action
            if not jittering:
                next_state, reward, done, _ = eval_env.step(action)
            elif jit_frames - jittered_frames < frame_skip:
                next_state, reward, done, _ = eval_env.jitter_step(action, jitter_force, jit_frames - jittered_frames,
                                                              frame_skip - (jit_frames - jittered_frames))
                jittering = False
                disturb = random.randint(50, 100) * time_change_factor
                eval_env.model.opt.gravity[0] = 0
                counter = 1
            else:
                next_state, reward, done, _ = eval_env.step(action)
                jittered_frames += frame_skip
                if jittered_frames == jit_frames:
                    jittering = False
                    disturb = random.randint(50, 100) * time_change_factor
                    eval_env.model.opt.gravity[0] = 0
                    counter = 1
            avg_reward += reward
            state = next_state
            if jit_duration:
                if counter % disturb == 0:
                    jitter_force = np.random.random() * force
                    eval_env.model.opt.gravity[0] = jitter_force
                    jittering = True
                    jittered_frames = 0
                counter += 1

            t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def make_env(env_name, seed, time_change_factor, env_timestep, frameskip):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env._max_episode_steps = 1000 * time_change_factor
    env.model.opt.timestep = env_timestep
    env.frameskip = frameskip

    return env


def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          save_model=False, load_model="", jit_duration=0.02, g_ratio=1, response_rate=0.04):
    hori_force = g_ratio * 9.81
    env_name = 'InvertedPendulum-v2'
    arguments = [policy, env_name, seed, jit_duration, g_ratio, response_rate]
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
    elif jit_duration:
        timestep = jit_duration
        frame_skip = response_rate / timestep
    else:
        timestep = response_rate
        frame_skip = 1
    jit_frames = 0
    if jit_duration:
        if jit_duration % timestep == 0:
            jit_frames = jit_duration / timestep
        else:
            raise ValueError("jit_duration should be a multiple of the timestep: " + str(timestep))

    print('timestep:', timestep)
    print('frameskip:', frame_skip)

    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    env = make_env(env_name, seed, time_change_factor, timestep, frame_skip)

    print('time change factor', time_change_factor)
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_timesteps = max_timesteps * time_change_factor
    eval_freq = eval_freq * time_change_factor
    start_timesteps = start_timesteps * time_change_factor


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
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

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor,
                               jit_duration=jit_duration, env_timestep=timestep, force=hori_force, frame_skip=frame_skip,
                               jit_frames=jit_frames)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    if jit_duration:
        counter = 1
        disturb = random.randint(50, 100) * time_change_factor
        print("==> Using Horizontal Jitter!")

    jittering = False
    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        if not jittering:
            next_state, reward, done, _ = env.step(action)
        elif jit_frames - jittered_frames < frame_skip:
            next_state, reward, done, _ = env.jitter_step(action, jitter_force, jit_frames - jittered_frames, frame_skip - (jit_frames - jittered_frames))
            jittering = False
            disturb = random.randint(50, 100) * time_change_factor
            env.model.opt.gravity[0] = 0
            counter = 1
        else:
            next_state, reward, done, _ = env.step(action)
            jittered_frames += frame_skip
            if jittered_frames == jit_frames:
                jittering = False
                disturb = random.randint(50, 100) * time_change_factor
                env.model.opt.gravity[0] = 0
                counter = 1

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

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

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(
                eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor,
                            jit_duration=jit_duration, env_timestep=timestep, force=hori_force, frame_skip=frame_skip,
                            jit_frames=jit_frames))
            np.save(f"./results/{file_name}", evaluations)
            if save_model: policy.save(f"./models/{file_name}")

        if jit_duration:
            if counter % disturb == 0:
                jitter_force = np.random.random() * hori_force
                env.model.opt.gravity[0] = jitter_force
                jittering = True
                jittered_frames = 0
            counter += 1


        # if t >= 25000:
        #     env.render()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--jit_duration", default=0.0, type=float, help="Duration in seconds for the horizontal force")
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.04, type=float, help="Response time of the agent in seconds")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
