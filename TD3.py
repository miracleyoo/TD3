import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class DelayedActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, neurons=[400, 300]):
        super(DelayedActor, self).__init__()

        self.l1 = nn.Linear(state_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class DelayedActorFastHybrid(nn.Module):
    def __init__(self, observation_space, action_dim, max_action):
        super(DelayedActorFastHybrid, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space) * 2 + action_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class DelayedQuickActor(nn.Module):
    def __init__(self, observation_space, action_dim, max_action, threshold=0.15, reflex_force_scale=1.0):
        super(DelayedQuickActor, self).__init__()

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

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state = torch.cat(state, dim=1)
        a1 = F.relu(self.reflex_detector(state))
        reflex = self.reflex(a1)

        a2 = F.relu(self.l1(state))
        # a = F.relu(self.l2(torch.cat((a2, a1), dim=1)))
        # print(torch.cat((a1, a2), dim=1).shape, a1.shape, a2.shape)
        a = F.relu(self.l2(a2))

        return reflex, self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DelayedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=[400, 300]):
        super(DelayedCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l2 = nn.Linear(neurons[0], neurons[1])
        self.l3 = nn.Linear(neurons[1], 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, neurons[0])
        self.l5 = nn.Linear(neurons[0], neurons[1])
        self.l6 = nn.Linear(neurons[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DelayedCriticFastHybrid(nn.Module):
    def __init__(self, observation_space, action_dim):
        super(DelayedCriticFastHybrid, self).__init__()

        input_dim = sum(s.shape[0] for s in observation_space) * 2 + action_dim
        # Q1 architecture
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1



class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            observation_space,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            delayed_env=False,
            reflex=False,
            threshold=0.15,
            reflex_force_scale=1.0,
            fast_hybrid=False,
            neurons=[400,300]
    ):

        self.delayed_env = delayed_env
        self.reflex = reflex

        if reflex:
            self.actor = DelayedQuickActor(observation_space, action_dim, max_action, threshold, reflex_force_scale).to(device)
            self.critic = DelayedCritic(observation_space, action_dim).to(device)
        elif self.delayed_env:
            if fast_hybrid:
                self.actor = DelayedActorFastHybrid(observation_space, action_dim, max_action).to(device)
                self.critic = DelayedCriticFastHybrid(observation_space, action_dim).to(device)
            else:
                self.actor = DelayedActor(state_dim, action_dim, max_action, neurons).to(device)
                self.critic = DelayedCritic(state_dim, action_dim, neurons).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.reflex:
            return self.actor(state)[0].cpu().data.numpy().flatten(), self.actor(state)[1].cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            if self.reflex:
                next_action = (
                        self.actor_target(next_state)[1] + noise
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = (
                        self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            if self.reflex:
                actor_loss = -self.critic.Q1(state, self.actor(state)[1]).mean()
            else:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


def collate(batch, device=None):
  elem = batch[0]
  if isinstance(elem, torch.Tensor):
    return torch.stack(batch).to(device)
    # if elem.numel() < 20000:  # TODO: link to the relavant profiling that lead to this threshold
    #   return torch.stack(batch).to(device)
    # else:
    #   return torch.stack([b.contiguous().to(device) for b in batch], 0)
  elif isinstance(elem, np.ndarray):
    return collate(tuple(torch.from_numpy(b) for b in batch), device)
  elif hasattr(elem, '__torch_tensor__'):
    return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
  elif isinstance(elem, Sequence):
    transposed = zip(*batch)
    return type(elem)(collate(samples, device) for samples in transposed)
  elif isinstance(elem, Mapping):
    return type(elem)((key, collate(tuple(d[key] for d in batch), device)) for key in elem)
  else:
    return torch.from_numpy(np.array(batch)).to(device)