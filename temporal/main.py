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
import neptune.new as neptune
from common import make_env, create_folders, get_frame_skip_and_timestep, perform_action, random_jitter_force, random_disturb
from evals import *




# Main function of the policy. Model is trained and evaluated inside.
def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          save_model=False, load_model="", jit_duration=0.02, g_ratio=1, response_rate=0.04, std_eval=False,
          catastrophe_frequency=1, delayed_env=False, env_name='InvertedPendulum-v2', neurons=256):

    if env_name == 'InvertedPendulum-v2':
        default_timestep = 0.02
        default_frame_skip = 2
    elif env_name == 'Hopper-v2':
        default_timestep = 0.002
        default_frame_skip = 4
    elif env_name == 'Walker2d-v2':
        default_timestep = 0.002
        default_frame_skip = 4

    max_force = g_ratio * 9.81
    eval_policy = eval_policy_std if std_eval else eval_policy_increasing_force
    arguments = [policy, env_name, seed, jit_duration, g_ratio, response_rate, catastrophe_frequency, delayed_env]
    file_name = '_'.join([str(x) for x in arguments])

    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    parameters = {
        'policy': policy,
        'env_name': env_name,
        'seed': seed,
        'jit_duration': jit_duration,
        'g_ratio': g_ratio,
        'response_rate': response_rate,
        'catastrophe_frequency': catastrophe_frequency,
        'delayed_env': delayed_env,
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, response_rate, default_timestep)

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

    state_dim = env.observation_space[0].shape[0] if delayed_env else env.observation_space.shape[0]
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
        "reflex": False,
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

    if delayed_env:
        replay_buffer = utils.ReplayBuffer(state_dim+action_dim, action_dim)
    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    avg_reward, _, _, _ = eval_policy(policy, env_name, max_action, eval_episodes=10, time_change_factor=time_change_factor,
                             env_timestep=timestep, frame_skip=frame_skip, jit_frames=jit_frames,
                             response_rate=response_rate, delayed_env=delayed_env)
    evaluations = [avg_reward]
    run['avg_reward'].log(avg_reward * response_rate)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    jittered_frames = 0
    max_episode_timestep = env.env.env._max_episode_steps if delayed_env else env.env._max_episode_steps

    counter = 0
    best_performance = 0
    disturb = round(random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency), 3)
    print("==> Using Horizontal Jitter!")

    jittering = False
    jitter_force = 0

    def stop_force():
        nonlocal jittered_frames
        nonlocal jittering
        nonlocal env
        nonlocal counter
        nonlocal disturb

        jittered_frames = 0
        jittering = False
        env.model.opt.gravity[0] = 0
        counter = 0
        disturb = round(random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency), 3)


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
        jittering, disturb, counter, jittered_frames, jitter_force, max_force, next_state, reward, done = perform_action(jittering, disturb, counter, response_rate, env, False, action, 0, frame_skip, random_jitter_force, max_force, timestep, jit_frames, jittered_frames, random_disturb, jitter_force, catastrophe_frequency, delayed_env)
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        counter = round(counter, 3)

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
            stop_force()

        # Evaluate episode
        if (t + 1) % eval_freq == 0:

            avg_reward, _, _, _ = eval_policy(policy, env_name, max_action, eval_episodes=10, time_change_factor=time_change_factor,
                                     env_timestep=timestep, frame_skip=frame_skip, jit_frames=jit_frames,
                                     response_rate=response_rate, delayed_env=delayed_env)
            evaluations.append(avg_reward)
            print(f" --------------- Evaluation reward {avg_reward * response_rate:.3f}")
            run['avg_reward'].log(avg_reward * response_rate)
            np.save(f"./results/{file_name}", evaluations)

            if best_performance < avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance * response_rate)
                if save_model:
                    policy.save(f"./models/{file_name}_best")

        if counter == disturb:  # Execute adding jitter horizontal force here
            jitter_force, _ = random_jitter_force(max_force)  # Jitter force strength w/ direction
            env.model.opt.gravity[0] = jitter_force
            jittering = True
            jittered_frames = 0

    if save_model:
        policy.save(f"./models/{file_name}_final")
        # if t >= 25000:
        #     env.render()

    run.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--start_timesteps", default=1000, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=5000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1000000, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, help="Discount factor")
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--load_model", default="", help="Model load file name, `` doesn't load, `default` uses file_name")
    parser.add_argument("--jit_duration", default=0.02, type=float, help="Duration in seconds for the horizontal force")
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.04, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--std_eval", action="store_true", help="Use standard evaluation or original evaluation policy")
    parser.add_argument("--catastrophe_frequency", default=1.0, type=float, help="Modify how often to apply catastrophe")
    parser.add_argument("--delayed_env", action="store_true", help="Delay the environment by 1 step")
    parser.add_argument("--neurons", default=256, type=int, help='number of neurons in hidden layer')


    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
