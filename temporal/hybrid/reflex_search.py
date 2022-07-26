from tqdm import tqdm
import torch
import sys
import threading
import time
import os
import argparse
import neptune.new as neptune
import numpy as np
import pandas as pd
sys.path.append('../')
from common import make_env, get_frame_skip_and_timestep
from evals import *
sys.path.append('../../')
import TD3
import utils

default_timesteps = {'InvertedPendulum-v2':0.02, 'Hopper-v2': 0.002, 'Walker2d-v2': 0.002, 'InvertedDoublePendulum-v2':0.01}
default_frame_skips = {'InvertedPendulum-v2':2, 'Hopper-v2': 4, 'Walker2d-v2': 4, 'InvertedDoublePendulum-v2':5}

class CEMThread(threading.Thread):
    def __init__(self, threshold_means, threshold_stds, scale_means, scale_stds, observation_space, parent_policy, env_name, max_action, time_change_factor, timestep, frame_skip, jit_frames, response_rate, parent_steps, df, action_space):
        threading.Thread.__init__(self)
        self.threshold = np.random.normal(threshold_means, threshold_stds)
        self.scale = np.random.normal(scale_means, scale_stds)
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = utils.CEMReflex(self.observation_space, self.action_space, self.threshold, self.scale).to('cuda')
        self.parent_policy = parent_policy
        self.env_name = env_name
        self.max_action = max_action
        self.time_change_factor = time_change_factor
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.jit_frames = jit_frames
        self.response_rate = response_rate
        self.parent_steps = parent_steps
        self.df = df

    def run(self):
        # reward_total = 0
        # for parent_policy in parent_policies[]:
        delayed_env = True
        avg_reward, avg_angle, jerk, actions = eval_policy_increasing_force_hybrid_reflex(self.policy, self.parent_policy,
                                                                                          self.env_name, self.max_action, 10,
                                                                                          self.time_change_factor,
                                                                                          self.timestep, self.frame_skip,
                                                                                          self.jit_frames, self.response_rate,
                                                                                          delayed_env, self.parent_steps,
                                                                                          False, fast_eval=True)
        # reward_total += avg_reward
        # reward_total = reward_total/len(parent_policies)
        self.df.loc[len(self.df.index)] = [self.threshold, self.scale, avg_reward]


def eval(env_name='InvertedPendulum-v2', response_rate=0.02, g_ratio=0, seed=0, population=20, jit_duration=0.02, parent_response_rate=0.04):
    default_timestep = default_timesteps[env_name]
    default_frame_skip = default_frame_skips[env_name]
    delayed_env = True
    elite_population = int(population/5)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    arguments = ['reflex_fixed', 'TD3', env_name, seed, jit_duration, float(g_ratio), response_rate, 1.0, delayed_env,
                 parent_response_rate, True, 'best']
    file_name = '_'.join([str(x) for x in arguments])
    frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, response_rate,
                                                                   default_timestep)
    parent_steps = int(parent_response_rate / response_rate)
    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    eval_env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)
    eval_env.env.env._max_episode_steps = 100000
    state_dim = eval_env.observation_space[0].shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "observation_space": eval_env.observation_space,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
        "delayed_env": delayed_env
    }
    kwargs["policy_noise"] = 2 * max_action
    kwargs["noise_clip"] = 0.5 * max_action
    kwargs["policy_freq"] = 2
    parent_policy = TD3.TD3(**kwargs)
    policy_file = file_name
    if os.path.exists('models_paper/' + policy_file + "_critic"):
        parent_policy.load(f"models_paper/{policy_file}")
    else:
        print(policy_file)


    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    parameters = {
        'env_name': env_name,
        'jit_duration': jit_duration,
        'g_force': g_ratio,
        'seed': seed,
        'response_rate': response_rate,
        'delayed_env': delayed_env,
        'type': 'CEM Reflex Search',
        'population': population
    }
    run["parameters"] = parameters

    arguments = ['reflex_search', env_name, seed, float(g_ratio), population]
    file_name = '_'.join([str(x) for x in arguments])
    num_actions = len(eval_env.action_space.high)
    threshold_means = np.zeros(state_dim * num_actions)
    threshold_stds = np.ones(state_dim * num_actions) * 4
    scale_means = np.zeros(state_dim * num_actions)
    scale_stds = np.ones(state_dim * num_actions) * max_action
    run['max_reward'].log(0)
    run['elite_avg_reward'].log(0)
    for step in range(100):

        df = pd.DataFrame(columns=['thresholds', 'scales', 'rewards'])
        threads = []
        for pop in range(population):
            thread = CEMThread(threshold_means, threshold_stds, scale_means, scale_stds, eval_env.observation_space, parent_policy, env_name, max_action, time_change_factor, timestep, frame_skip, jit_frames, response_rate, parent_steps, df, eval_env.action_space.high)
            thread.start()
            threads.append(thread)

        for thread in tqdm(threads):
            thread.join()



        df = df.sort_values(by=['rewards'], ascending=False, ignore_index=True)
        print("Max Reward for step ", step, ':', df['rewards'][0] * response_rate, "Elite avg reward:", np.mean(df['rewards'][0:elite_population]) * response_rate)
        run['max_reward'].log(df['rewards'][0] * response_rate)
        run['elite_avg_reward'].log(np.mean(df['rewards'][0:elite_population]) * response_rate)
        threshold_means = np.mean(df['thresholds'][0:elite_population])
        scale_means = np.mean(df['scales'][0:elite_population])

        threshold_stds = np.std(np.array(df['thresholds'][0:elite_population]))
        scale_stds = np.std(np.array(df['scales'][0:elite_population]))
        np.save(f"./models/{file_name}_thresholds_intermediate", df['thresholds'][0])
        np.save(f"./models/{file_name}_scales_intermediate", df['scales'][0])

    np.save(f"./models/{file_name}_thresholds", df['thresholds'][0])
    np.save(f"./models/{file_name}_scales", df['scales'][0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.02, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--population", default=20, type=int, help="Population size")
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--parent_response_rate", default=0.04, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--jit_duration", default=0.02, type=float, help="Duration in seconds for the horizontal force")


    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    eval(**args)

