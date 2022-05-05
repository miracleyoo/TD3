import os
import sys
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common import make_env, get_frame_skip_and_timestep
from evals import eval_TD_error_increasing_force

sys.path.append("../../")
import TD3

# Hyperparameters to consider
response_times = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
g_forces = [5, 10]
seeds = [0, 1, 2, 3, 4]

# Ensure all runs exist
for response_rate in response_times:
    for g_force in g_forces:
        for seed in seeds:
            if not os.path.isfile('models3/TD3_InvertedPendulum-v2_'+str(seed)+'_0.02_'+str(float(g_force))+'_'+str(response_rate)+'_1.0_True_256_final_actor'):
                print(f"Missing run: {response_rate, g_force, seed}")

df = pd.DataFrame(columns=["seed", "g_force", "response_rate", "eval_episode_num", "total_reward", "timestep", "time", "TD_error_1", "TD_error_2"])

# Environment hyperparameters
default_timestep = 0.02
default_frame_skip = 2
jit_duration = 0.02
env_name = "InvertedPendulum-v2"
delayed_env = True

# Experiment hyperparameters
CRITIC_RESPONSE_RATE = 0.02
if CRITIC_RESPONSE_RATE not in response_times: raise ValueError
NUM_EVAL_EPISODES = 10
STANDARDIZED_RESPONSE_RATE = 0.04

for i, g_force in enumerate(g_forces):
    print(f"G Force: {i+1}/{len(g_forces)}")
    for j, seed in enumerate(seeds):
        print(f"    - Seed: {j+1}/{len(seeds)}")

        # --- Load evaluation action-value function (critic) --- #

        # Get environment hyperparameters
        frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, CRITIC_RESPONSE_RATE)
        time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
        temp_env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)
        temp_env.env.env._max_episode_steps = 100000
        state_dim = temp_env.observation_space[0].shape[0]
        action_dim = temp_env.action_space.shape[0]
        max_action = float(temp_env.action_space.high[0])
        
        # Initialise critic class
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "observation_space": temp_env.observation_space,
            "max_action": max_action,
            "discount": 0.99,
            "tau": 0.005,
            "delayed_env": delayed_env
        }
        kwargs["policy_noise"] = 2 * max_action
        kwargs["noise_clip"] = 0.5 * max_action
        kwargs["policy_freq"] = 2
        critic = TD3.TD3(**kwargs)

        # Load critic parameters
        arguments = ["TD3", env_name, seed, jit_duration, float(g_force), CRITIC_RESPONSE_RATE, 1.0, delayed_env, 256, "best"]
        file_name = '_'.join([str(x) for x in arguments])
        if os.path.exists('models3/'+file_name+"_critic"):
            critic.load(f"models3/{file_name}")
        else:
            raise FileNotFoundError

        del temp_env

        # --- Evaluate agents of various response times --- #

        for response_rate in response_times:

            # Determine reward scale factor (for 40ms standardized reward)
            REWARD_SCALE_FACTOR = response_rate / STANDARDIZED_RESPONSE_RATE
            Q_SCALE_FACTOR = CRITIC_RESPONSE_RATE / STANDARDIZED_RESPONSE_RATE

            # Get environment hyperparameters
            frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, response_rate)
            time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
            eval_env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)
            eval_env.env.env._max_episode_steps = 100000
            state_dim = eval_env.observation_space[0].shape[0]
            action_dim = eval_env.action_space.shape[0]
            max_action = float(eval_env.action_space.high[0])
            
            # Initialise policy class
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
            policy = TD3.TD3(**kwargs)

            # Load policy parameters
            arguments = ["TD3", env_name, seed, jit_duration, float(g_force), response_rate, 1.0, delayed_env, 256, "best"]
            file_name = '_'.join([str(x) for x in arguments])
            if os.path.exists('models3/'+file_name+"_critic"):
                policy.load(f"models3/{file_name}")
            else:
                raise FileNotFoundError

            # Evaluate policy and record TD error over episode
            evaluation_data = eval_TD_error_increasing_force(critic, policy, env_name, NUM_EVAL_EPISODES, time_change_factor, timestep, frame_skip, jit_frames, response_rate, delayed_env, None, REWARD_SCALE_FACTOR, Q_SCALE_FACTOR)

            for eval_episode_num, episode_data in evaluation_data.items():
                total_reward = episode_data[0]
                episode_TD_error_1 = episode_data[1]
                episode_TD_error_2 = episode_data[2]
                assert len(episode_TD_error_1) == len(episode_TD_error_2)
                for episode_timestep in range(len(episode_TD_error_1)):
                    df.loc[len(df.index)] = [seed, g_force, response_rate, eval_episode_num, total_reward, 
                                             (episode_timestep + 1), (episode_timestep + 1) * response_rate, 
                                             episode_TD_error_1[episode_timestep], 
                                             episode_TD_error_2[episode_timestep]
                                            ]

torch.save(df, "dataframe_TD_error")