import sys
sys.path.append('../../')
import DDPG
import OurDDPG
import TD3
import utils
import numpy as np
import torch
import argparse
import pandas as pd
import os
import random
import neptune.new as neptune
from utils import Reflex
from torch.utils.data import DataLoader
from torch import nn

sys.path.append('../')
from common import make_env, create_folders, get_frame_skip_and_timestep, perform_action, random_jitter_force, random_disturb, get_TD, get_Q
from evals import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_timestep = 0.02  # for Inverted Pendulum-V2 todo: add for others
default_frame_skip = 2


def train_reflex(dataloader, model, loss_fn, optimizer, run):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)[:, 0]
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print(f"loss: {loss:>7f}")
    run['loss'].log(loss)


def collect_frames(env, g_ratio, df, policy, max_action, response_rate, frame_skip, timestep, jit_frames,
                   max_episode_timestep, delayed_env, collect_failure=False):
    env.model.opt.gravity[0] = 0
    counter = 0
    disturb = round(random.randint(50, 100) * 0.04, 3)
    max_force = g_ratio * 9.81 * (2 * collect_failure)
    jittered_frames = 0
    jittering = False
    jitter_force = 0
    state, done = env.reset(), False
    episode_timesteps = 0
    while len(df[df.failure == collect_failure]) < 2000:
        episode_timesteps += 1
        action = policy.select_action(state).clip(-max_action, max_action)
        jittering, disturb, counter, jittered_frames, jitter_force, next_state, reward, done = perform_action(
            jittering, disturb, counter, response_rate, env, False, action, 0, frame_skip,
            random_jitter_force, max_force, timestep, jit_frames, jittered_frames, random_disturb, jitter_force, 1,
            delayed_env)
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0.0
        if done:
            if collect_failure and done_bool:
                df.loc[len(df.index)] = [state, action, done_bool]
                if len(df[df.failure == collect_failure]) % 100 == 0:
                    print(len(df[df.failure == collect_failure]), '/4000')
            env.model.opt.gravity[0] = 0
            counter = 0
            disturb = round(random.randint(50, 100) * 0.04, 3)
            jittered_frames = 0
            jittering = False
            jitter_force = 0
            state, done = env.reset(), False
            episode_timesteps = 0
        else:
            if not collect_failure:
                df.loc[len(df.index)] = [state, action, done_bool]
                if len(df[df.failure == collect_failure]) % 100 == 0:
                    print(len(df[df.failure == collect_failure]), '/4000')
            state = next_state
            counter = round(counter, 3)
            if counter == disturb:
                jitter_force = random_jitter_force(max_force)
                env.model.opt.gravity[0] = jitter_force
                jittering = True
                jittered_frames = 0


    return


# Main function of the policy. Model is trained and evaluated inside.
def train(policy='TD3', seed=0, start_timesteps=25e3, eval_freq=5e3, expl_noise=0.1, batch_size=256, discount=0.99,
          tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5, save_model=False, jit_duration=0.02, g_ratio=1,
          response_rate=0.04, catastrophe_frequency=1, delayed_env=False, env_name='InvertedPendulum-v2'):

    max_force = g_ratio * 9.81
    arguments = ["reflex_network", policy, env_name, seed, jit_duration, g_ratio, response_rate, catastrophe_frequency,
                 delayed_env]

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
        'type': 'train reflex'
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    frame_skip, timestep, jit_frames = get_frame_skip_and_timestep(jit_duration, response_rate, default_timestep)  # child policy
    print('timestep:', timestep)
    print('frameskip:', frame_skip)

    # The ratio of the default time consumption between two states returned and default version.
    # Used to reset the max episode number to guarantee the actual max time is always the same.
    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    print('time change factor', time_change_factor)
    # Create environment
    env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    arguments = ['TD3', env_name, seed, jit_duration, float(g_ratio), response_rate, 1.0, delayed_env, 'best']
    policy_file = '_'.join([str(x) for x in arguments])
    policy.load(f"../models_paper/{policy_file}")
    max_episode_timestep = env.env.env._max_episode_steps if delayed_env else env.env._max_episode_steps

    # df = pd.DataFrame(columns=['states', 'action', 'failure'])
    # collect_frames(env, g_ratio, df, policy, max_action, response_rate, frame_skip, timestep, jit_frames,
    #                max_episode_timestep, delayed_env, False)
    # print("Non failure frames collected", len(df))
    # collect_frames(env, g_ratio * 2, df, policy, max_action, response_rate, frame_skip, timestep, jit_frames,
    #                max_episode_timestep, delayed_env, True) # double g_ratio to fail faster
    # print("All frames collected")
    # torch.save(df, 'temp_df')
    df = torch.load('temp_df')
    training_data = utils.StatesDataset(df)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Reflex().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 100
    for t in range(epochs):
        train_reflex(train_dataloader, model, loss_fn, optimizer, run)
    print("Done!")

    model.eval()
    test_dataloader = DataLoader(utils.StatesDataset(df[df.failure == True]), batch_size=64, shuffle=True, )
    total_loss = 0
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)[:, 0]
        loss = loss_fn(pred, y)
        total_loss += loss
    print(total_loss * 64 / 2000)

    test_dataloader = DataLoader(utils.StatesDataset(df[df.failure == False]), batch_size=64, shuffle=True, )
    total_loss = 0
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)[:, 0]
        loss = loss_fn(pred, y)
        total_loss += loss
    print(total_loss * 64 / 2000)



    if save_model:
        torch.save(model, f"./models/{file_name}")
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
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
    parser.add_argument("--catastrophe_frequency", default=1.0, type=float, help="Modify how often to apply catastrophe")
    parser.add_argument("--delayed_env", action="store_true", help="Delay the environment by 1 step")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
