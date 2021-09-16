import numpy as np
import torch
import gym

__all__=["make_env", "jitter_step", "ReplayBuffer"]

# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env._max_episode_steps = 1000 * time_change_factor
    env.model.opt.timestep = env_timestep
    env.frameskip = frameskip
    env.jitter_step = jitter_step
    return env

# The alternative step function when some frames of a step are under the
# jitter force while others are not
def jitter_step(self, a, force, frames1, frames2):
    self.model.opt.gravity[0] = force
    reward = 1.0
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = 0 # force # 0 here? frames1 are with force while frames2 are supposed not.
    self.do_simulation(a, int(frames2))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}

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