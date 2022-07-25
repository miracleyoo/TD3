import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Reflex(nn.Module):
    def __init__(self):
        super(Reflex, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class HandCraftedReflex(nn.Module):
    def __init__(self, observation_space, threshold=0.15, reflex_force_scale=1.0):
        super(HandCraftedReflex, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space)
        self.reflex_detector = nn.Linear(input_dim, 2)
        self.reflex_detector.weight.requires_grad = False
        self.reflex_detector.bias.requires_grad = False
        self.reflex_detector.weight.data = torch.zeros(self.reflex_detector.weight.shape)
        self.reflex_detector.weight.data[0, 1] = 1
        self.reflex_detector.weight.data[1, 1] = -1
        self.reflex_detector.bias.data = torch.ones(self.reflex_detector.bias.data.shape) * threshold * -1

        self.reflex = nn.Linear(2, 1)
        self.reflex.weight.requires_grad = False
        self.reflex.bias.requires_grad = False
        self.reflex.weight.data[0, 0] = reflex_force_scale / (0.20 - threshold)
        self.reflex.weight.data[0, 1] = -reflex_force_scale / (0.20 - threshold)
        self.reflex.bias.data[0] = 0

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)
        return reflex


class CEMReflex(nn.Module):
    def __init__(self, observation_space, action_space, thresholds=None, reflex_force_scales=None):
        super(CEMReflex, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space)
        num_action = len(action_space)
        if thresholds is None:
            thresholds = np.zeros((input_dim - 1) * num_action)
        if reflex_force_scales is None:
            reflex_force_scales = np.zeros((input_dim - 1) * num_action)

        self.reflex_detector = nn.Linear(input_dim, (input_dim - 1) * 2 * num_action)
        self.reflex_detector.weight.requires_grad = False
        self.reflex_detector.bias.requires_grad = False
        self.reflex_detector.weight.data = torch.zeros(self.reflex_detector.weight.shape)
        for action in range(num_action):
            for i in range(input_dim - 1):
                self.reflex_detector.weight.data[(action * (input_dim - 1) * 2) + i * 2, i] = 1
                self.reflex_detector.weight.data[(action * (input_dim - 1) * 2) + i * 2 + 1, i] = -1
                self.reflex_detector.bias.data[(action * (input_dim - 1) * 2) + i * 2] = thresholds[(action * (input_dim - 1)) + i] * -1
                self.reflex_detector.bias.data[(action * (input_dim - 1) * 2) + i * 2 + 1] = thresholds[(action * (input_dim - 1)) + i] * -1

        self.reflex = nn.Linear((input_dim - 1) * 2 * num_action, num_action)
        self.reflex.weight.requires_grad = False
        self.reflex.bias.requires_grad = False
        for action in range(num_action):
            for i in range(input_dim - 1):
                self.reflex.weight.data[action,  (action*(input_dim - 1) * 2) + (i * 2)] = reflex_force_scales[(action * (input_dim - 1)) + i]
                self.reflex.weight.data[action, (action*(input_dim - 1) * 2) + (i * 2) + 1] = -reflex_force_scales[(action * (input_dim - 1)) + i]
        self.reflex.bias.data = torch.zeros(self.reflex.bias.shape)

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)
        return reflex


class StatesDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        state = self.df['states'].iloc[idx]
        action = self.df['action'].iloc[idx][0]
        failure = self.df['failure'].iloc[idx]
        label = 0 if failure == 0.0 else action

        return torch.Tensor(state), label


def append_data_to_excel(excel_name, columns, data):
    if not os.path.isfile(os.path.join(excel_name)):
        with open(os.path.join(excel_name), 'w') as f:
            f.write(','.join([str(x) for x in columns]) + '\n')

    with open(os.path.join(excel_name), 'a') as f:
        f.write(','.join([str(x) for x in data]) + '\n')

