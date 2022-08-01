import sys
sys.path.append('../../')
import DDPG
import OurDDPG
import TD3
import utils
from utils import Reflex
import numpy as np
import torch
from torch import nn
import argparse
import os
import random
import neptune.new as neptune
sys.path.append('../')
from common import make_env, create_folders, get_frame_skip_and_timestep, perform_action, const_jitter_force, random_disturb, get_TD, get_Q, const_disturb_half
from evals import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


default_timesteps = {'InvertedPendulum-v2':0.02, 'Hopper-v2': 0.002, 'Walker2d-v2': 0.002, 'InvertedDoublePendulum-v2':0.01}
default_frame_skips = {'InvertedPendulum-v2':2, 'Hopper-v2': 4, 'Walker2d-v2': 4, 'InvertedDoublePendulum-v2':5}


# Main function of the policy. Model is trained and evaluated inside.
def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          response_rate=0.04, env_name='InvertedPendulum-v2', parent_response_rate=0.04):

    delayed_env = True
    default_timestep = default_timesteps[env_name]
    default_frame_skip = default_frame_skips[env_name]
    policy_name = policy

    arguments = ["reflex_train_normal_penalty", policy_name, env_name, seed, response_rate, parent_response_rate]

    file_name = '_'.join([str(x) for x in arguments])

    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    parameters = {
        'policy': policy,
        'env_name': env_name,
        'seed': seed,
        'response_rate': response_rate,
        'parent_response_rate': parent_response_rate,
        'type': 'Reflex train normal'
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()
    timestep = default_timestep if default_timestep <= response_rate else response_rate
    frame_skip = response_rate / timestep
    print('timestep:', timestep)
    print('frameskip:', frame_skip)
    parent_steps = int(parent_response_rate/response_rate)  # Number children steps in on parent step

    # The ratio of the default time consumption between two states returned and default version.
    # Used to reset the max episode number to guarantee the actual max time is always the same.
    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    print('time change factor for child', time_change_factor)
    # Create environment
    env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Scale all timestep related parameters

    state_dim = env.observation_space[0].shape[0] if delayed_env else env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    parent_max_action = float(env.action_space.high[0])
    child_max_action = 2 * parent_max_action
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": parent_max_action,
        "discount": discount,
        "tau": tau,
        "observation_space": env.observation_space,
        "delayed_env": delayed_env,
        "reflex": False,
    }
    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * parent_max_action
        kwargs["noise_clip"] = noise_clip * parent_max_action
        kwargs["policy_freq"] = policy_freq
        parent_policy = TD3.TD3(**kwargs)
        kwargs["max_action"] = child_max_action
        kwargs["policy_noise"] = policy_noise * child_max_action
        kwargs["noise_clip"] = noise_clip * child_max_action
        policy = TD3.TD3(**kwargs)
    elif policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    arguments = [policy_name, env_name, seed, parent_response_rate, delayed_env, 'best']

    parent_file_name = '_'.join([str(x) for x in arguments])
    parent_policy.load(f"./models_paper/{parent_file_name}")

    # for child network: add parent state value as the input
    replay_buffer = utils.ReplayBuffer(state_dim + action_dim, action_dim)

    # Evaluate untrained policy
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = env.env.env._max_episode_steps if delayed_env else env.env._max_episode_steps

    best_performance = 0

    print('parent steps', parent_steps)
    parent_action = env.previous_action
    for t in range(int(max_timesteps)):
        if episode_timesteps % parent_steps == 0:
            next_parent_action = parent_policy.select_action(state)
        elif (episode_timesteps+1) % parent_steps == 0:
            parent_action = next_parent_action

        if t < start_timesteps:
            child_action = env.action_space.sample() * 2
        else:
            child_action = (
                    policy.select_action(state)
                    + np.random.normal(0, child_max_action * expl_noise, size=action_dim)
            )

        action = (parent_action + child_action).clip(-parent_max_action, parent_max_action)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        reward = reward - abs(np.mean(child_action)/child_max_action)
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        replay_buffer.add(state, child_action, next_state, reward, done_bool)

        state = next_state

        episode_timesteps += 1

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            parent_action = env.previous_action

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            eval_env = make_env(env_name, 100, time_change_factor, timestep, frame_skip, delayed_env)
            rewards = 0
            for _ in range(10):
                eval_state, eval_done = eval_env.reset(), False
                eval_parent_action = eval_env.previous_action
                eval_episode_timesteps = 0
                while not eval_done:
                    if eval_episode_timesteps % parent_steps == 0:
                        eval_next_parent_action = parent_policy.select_action(eval_state)
                    elif (eval_episode_timesteps + 1) % parent_steps == 0:
                        eval_parent_action = eval_next_parent_action
                    eval_child_action = policy.select_action(eval_state)
                    eval_action = (eval_parent_action + eval_child_action).clip(-parent_max_action, parent_max_action)
                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_state = eval_next_state
                    eval_episode_timesteps += 1
                    rewards += eval_reward
            avg_reward = rewards / 10
            evaluations.append(avg_reward)
            print(f" --------------- Evaluation reward {avg_reward:.3f}")
            run['avg_reward'].log(avg_reward)
            np.save(f"./results/{file_name}", evaluations)

            if best_performance <= avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance)
                policy.save(f"./models/{file_name}_best")

        if t >= start_timesteps:
           policy.train(replay_buffer, batch_size)

    policy.save(f"./models/{file_name}_final")

    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--start_timesteps", default=1000, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=5000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1000000, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, help="Discount factor")
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--response_rate", default=0.02, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--parent_response_rate", default=0.04, type=float, help="Response time of the agent in seconds")



    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
