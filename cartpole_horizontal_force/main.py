import numpy as np
import torch
import gym
import argparse
import os
import random

import sys

sys.path.append('../')
import utils
import TD3
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10, time_change_factor=1, jit=False, env_timestep=0.02, g_ratio=1):
    eval_env = gym.make(env_name)
    eval_env.seed(100)
    eval_env.unwrapped.spec.max_episode_steps = 1000 * time_change_factor
    eval_env.model.opt.timestep = env_timestep
    lasttime = 2 * time_change_factor

    avg_reward = 0.
    if jit:
        counter = 0
        disturb = random.randint(50, 100)
        print("==> Using Horizontal Jitter!")

    t = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

            if jit:
                if counter > 0 and counter % disturb == 0:
                    hori_force = (g_ratio * random.random() * 9.81) * pow(-1, (random.random() > 0.5))
                    eval_env.model.opt.gravity[0] = hori_force
                if counter > lasttime and counter % disturb == lasttime:
                    eval_env.model.opt.gravity[0] = 0
                    disturb = random.randint(50, 100)
                    counter = 0
                counter += 1

            t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(policy='TD3', env_name='InvertedPendulum-v2', seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2,noise_clip=0.5,
          save_model=False, load_model="", jit=False, g_ratio=1, env_timestep=0.02):
    arguments = [policy, env_name, seed, jit, g_ratio, env_timestep]
    file_name = '_'.join([str(x) for x in arguments])
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(env_name)
    time_change_factor = 0.02 / env_timestep

    print('time change factor', time_change_factor)
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env.unwrapped.spec.max_episode_steps = 1000 * time_change_factor
    max_timesteps = max_timesteps * time_change_factor
    eval_freq = eval_freq * time_change_factor
    start_timesteps = start_timesteps * time_change_factor
    env.model.opt.timestep = env_timestep
    lasttime = time_change_factor * 2

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
    evaluations = [eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor, jit=jit,
                               env_timestep=env_timestep, g_ratio=g_ratio)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    if jit:
        counter = 0
        disturb = random.randint(50, 100)
        print("==> Using Horizontal Jitter!")

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
        next_state, reward, done, _ = env.step(action)
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
                eval_policy(policy, env_name, eval_episodes=10, time_change_factor=time_change_factor, jit=jit,
                            env_timestep=env_timestep, g_ratio=g_ratio))
            np.save(f"./results/{file_name}", evaluations)
            if save_model: policy.save(f"./models/{file_name}")

        if jit:
            if counter > 0 and counter % disturb == 0:
                hori_force = (g_ratio * random.random() * 9.81) * pow(-1, (random.random() > 0.5))
                env.model.opt.gravity[0] = hori_force
            if counter > lasttime and counter % disturb == lasttime:
                env.model.opt.gravity[0] = 0
                disturb = random.randint(50, 100)
                counter = 0
            counter += 1

        # if t >= 25000:
        #     env.render()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env_name", default="InvertedPendulum-v2")  # OpenAI gym environment name
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
    parser.add_argument("--jit", action="store_true")  # Whether use jetter or not
    parser.add_argument("--g_ratio", default=0, type=float, help='Maximum horizontal force g ratio')
    parser.add_argument("--env_timestep", default=0.02, type=float, help="environment time between each frame")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
