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
from common import make_env, create_folders, get_frame_skip_and_timestep, perform_action, random_jitter_force, random_disturb, get_TD, get_Q
from evals import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


default_timesteps = {'InvertedPendulum-v2':0.02, 'Hopper-v2': 0.002, 'Walker2d-v2': 0.002, 'InvertedDoublePendulum-v2':0.01}
default_frame_skips = {'InvertedPendulum-v2':2, 'Hopper-v2': 4, 'Walker2d-v2': 4, 'InvertedDoublePendulum-v2':5}

# Main function of the policy. Model is trained and evaluated inside.
def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          save_model=False, jit_duration=0.02, g_ratio=0, response_rate=0.04, catastrophe_frequency=1,
          delayed_env=False, env_name='InvertedPendulum-v2', parent_response_rate=0.04, zero_reflex=False):

    default_timestep = default_timesteps[env_name]
    default_frame_skip = default_frame_skips[env_name]

    max_force = g_ratio * 9.81
    eval_policy = eval_policy_increasing_force_hybrid_reflex
    arguments = ["reflex_fixed", policy, env_name, seed, jit_duration, g_ratio, response_rate, catastrophe_frequency,
                 delayed_env, parent_response_rate, zero_reflex]

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
        'type': 'reflex_hybrid'
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, response_rate, default_timestep)  # child policy
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
    max_timesteps = max_timesteps * time_change_factor
    eval_freq = int(eval_freq * time_change_factor)
    start_timesteps = start_timesteps * time_change_factor

    state_dim = env.observation_space[0].shape[0] if delayed_env else env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    parent_max_action = float(env.action_space.high[0])
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
    elif policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    arguments = ['reflex_search', env_name, seed, float(g_ratio), 20]
    file_name = '_'.join([str(x) for x in arguments])
    threshold = np.load(f"./models_paper/{file_name}_thresholds_intermediate.npy")
    scale = np.load(f"./models_paper/{file_name}_scales_intermediate.npy")
    policy = utils.CEMReflex(env.observation_space, threshold, scale).to('cuda')
    # for child network: add parent state value as the input
    if delayed_env:
            replay_buffer = utils.ReplayBuffer(state_dim + action_dim, action_dim)
    else:
            replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    avg_reward, _, _, _ = eval_policy(policy, parent_policy, env_name, parent_max_action, eval_episodes=10,
                                      time_change_factor=time_change_factor, env_timestep=timestep,
                                      frame_skip=frame_skip, jit_frames=jit_frames, response_rate=response_rate,
                                      delayed_env=delayed_env, parent_steps=parent_steps, zero_reflex=zero_reflex)

    print('initial avg reward:', avg_reward * response_rate)

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

    print('parent steps', parent_steps)
    child_state = state
    parent_action = env.previous_action
    prev_parent_state = state
    for t in range(int(max_timesteps)):
        if episode_timesteps % parent_steps == 0:
            next_parent_action = (parent_policy.select_action(state) + np.random.normal(0, parent_max_action * expl_noise, size=action_dim)).clip(-parent_max_action, parent_max_action)
            prev_parent_state = state
        elif (episode_timesteps+1) % parent_steps == 0:
            parent_action = next_parent_action

        child_action = policy(torch.Tensor(child_state).to(device)).cpu().detach().numpy() * (1 - zero_reflex)
        action = (parent_action + child_action).clip(-parent_max_action, parent_max_action)


        jittering, disturb, counter, jittered_frames, jitter_force, next_state, reward, done = perform_action(
            jittering, disturb, counter, response_rate, env, False, action, 0, frame_skip, random_jitter_force,
            max_force, timestep, jit_frames, jittered_frames, random_disturb, jitter_force, catastrophe_frequency,
            delayed_env)
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        if (episode_timesteps+1) % parent_steps == 0 or done:
            replay_buffer.add(prev_parent_state, parent_action, next_state, reward, done_bool)

        child_state = next_state
        state = next_state
        episode_reward += reward
        counter = round(counter, 3)
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
            stop_force()
            child_state = state
            parent_action = env.previous_action

        if counter == disturb:  # Execute adding jitter horizontal force here
            jitter_force = random_jitter_force(max_force)  # Jitter force strength w/ direction
            env.model.opt.gravity[0] = jitter_force
            jittering = True
            jittered_frames = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:

            avg_reward, _, _, _ = eval_policy(policy, parent_policy, env_name, parent_max_action, eval_episodes=10,
                                                 time_change_factor=time_change_factor,
                                                 env_timestep=timestep, frame_skip=frame_skip,
                                                 jit_frames=jit_frames,
                                                 response_rate=response_rate, delayed_env=delayed_env,
                                                 parent_steps=parent_steps, zero_reflex=zero_reflex)
            evaluations.append(avg_reward)
            print(f" --------------- Evaluation reward {avg_reward * response_rate:.3f}")
            run['avg_reward'].log(avg_reward * response_rate)
            np.save(f"./results/{file_name}", evaluations)

            if best_performance < avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance * response_rate)
                if save_model:
                    parent_policy.save(f"./models/{file_name}_best")

        if t >= start_timesteps and episode_timesteps % parent_steps == 0:
           parent_policy.train(replay_buffer, batch_size)



    if save_model:
        parent_policy.save(f"./models/{file_name}_final")
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
    parser.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, help="Discount factor")
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--jit_duration", default=0.02, type=float, help="Duration in seconds for the horizontal force")
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--response_rate", default=0.02, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--parent_response_rate", default=0.04, type=float, help="Response time of the agent in seconds")
    parser.add_argument("--catastrophe_frequency", default=1.0, type=float, help="Modify how often to apply catastrophe")
    parser.add_argument("--delayed_env", action="store_true", help="Delay the environment by 1 step")
    parser.add_argument("--zero_reflex", action="store_true", help="Zero reflex")



    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
