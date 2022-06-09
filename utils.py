import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


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