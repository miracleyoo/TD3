from tqdm import tqdm
import sys
import os
import argparse
sys.path.append('../')
from common import make_env, get_frame_skip_and_timestep
from evals import *
sys.path.append('../../')
import TD3
import utils


def eval(response_rate=0.02, g_force=5, angle=0.15, reflex_force_scale=0.5):
    default_timestep = 0.02
    default_frame_skip = 2
    jit_duration = 0.02
    env_name = 'InvertedPendulum-v2'
    delayed_env = True
    parent_response_rate = 0.04
    for seed in tqdm(range(10)):
        states = []
        arguments = ['reflex', 'TD3', env_name, seed, jit_duration, float(g_force), 0.02, 1.0, delayed_env,
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
        policy = utils.HandCraftedReflex(eval_env.observation_space, angle, reflex_force_scale).to('cuda')
        avg_reward, avg_angle, jerk, actions = eval_policy_increasing_force_hybrid_reflex(policy, parent_policy,
                                                                                          env_name, max_action, 10,
                                                                                          time_change_factor,
                                                                                          timestep, frame_skip,
                                                                                          jit_frames, response_rate,
                                                                                          delayed_env, parent_steps,
                                                                                          False)
        utils.append_data_to_excel('results/eval_reflex_after_training.csv',
                                   ['seed', 'g_force', 'response_rate', 'parent_response_rate', 'reward', 'angle', 'reflex_force_scale'],
                                   [seed, g_force, response_rate, parent_response_rate, avg_reward, angle, reflex_force_scale])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_force", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.02, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--angle", default=0.15, type=float, help="angle to trigger reflex")
    parser.add_argument("--reflex_force_scale", default=0.5, type=float, help="scale force")


    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    eval(**args)

