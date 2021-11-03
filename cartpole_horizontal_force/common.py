import gym
import numpy as np
# from gym.envs.mujoco import inverted_pendulum
import types
__all__=["make_env"]

# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip, delayed_env):
    env = gym.make(env_name)
    if delayed_env:
        env = Float64ToFloat32(env)
        env = RealTimeWrapper(env)
    else:
        env.env.jitter_step_end = types.MethodType(jitter_step_end, env.env)
        env.env.jitter_step_start = types.MethodType(jitter_step_start, env.env)
    env.seed(seed)
    env.delayed = delayed_env
    env.action_space.seed(seed)
    env._max_episode_steps = 1000 * time_change_factor
    env.model.opt.timestep = env_timestep
    env.frame_skip = int(frameskip)
    return env


# The alternative step function when some frames of a step are under the
# jitter force while others are not
def jitter_step_end(self, a, force, frames1, frames2):
    self.model.opt.gravity[0] = force
    reward = 1.0
    self.do_simulation(a, int(round(frames1)))
    self.model.opt.gravity[0] = 0 # force # 0 here? frames1 are with force while frames2 are supposed not.
    self.do_simulation(a, int(round(frames2)))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}


def jitter_step_start(self, a, force, frames1, frames2, jit_frames):
    reward = 1.0
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = force
    if frames2 < jit_frames:
        self.do_simulation(a, int(round(frames2)))
    else:
        self.do_simulation(a, int(round(jit_frames)))
        self.model.opt.gravity[0] = 0
        self.do_simulation(a, int(round(frames2 - jit_frames)))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone

    return ob, reward, done, {}


class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        # self.initial_action = env.action_space.sample()
        assert isinstance(env.action_space, gym.spaces.Box)
        self.initial_action = env.action_space.high * 0
        self.previous_action = self.initial_action

    def reset(self):
        self.previous_action = self.initial_action
        return np.concatenate((super().reset(), self.previous_action), axis=0)

    def step(self, action):
        observation, reward, done, info = super().step(self.previous_action)
        self.previous_action = action
        return np.concatenate((observation, action), axis=0), reward, done, info

    def jitter_step_end(self, a, force, frames1, frames2):
        action = self.previous_action
        self.model.opt.gravity[0] = force
        reward = 1.0
        self.do_simulation(action, int(round(frames1)))
        self.model.opt.gravity[0] = 0  # force # 0 here? frames1 are with force while frames2 are supposed not.
        self.do_simulation(action, int(round(frames2)))
        ob = self.env.env.env._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        return ob, reward, done, {}

    def jitter_step_start(self, a, force, frames1, frames2, jit_frames):

        action = self.previous_action
        reward = 1.0
        self.do_simulation(action, int(frames1))
        self.model.opt.gravity[0] = force
        if frames2 < jit_frames:
            self.do_simulation(action, int(round(frames2)))
        else:
            self.do_simulation(action, int(round(jit_frames)))
            self.model.opt.gravity[0] = 0
            self.do_simulation(action, int(round(frames2 - jit_frames)))
        ob = self.env.env.env._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        return ob, reward, done, {}


class Float64ToFloat32(gym.ObservationWrapper):
  """Converts np.float64 arrays in the observations to np.float32 arrays."""

  # TODO: change observation/action spaces to correct dtype
  def observation(self, observation):
    observation = deepmap({np.ndarray: float64_to_float32}, observation)
    return observation

  def step(self, action):
    s, r, d, info = super().step(action)
    return s, r, d, info


def deepmap(f, m):
  """Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
  Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
  for cls in f:
    if isinstance(m, cls):
      return f[cls](m)
  if isinstance(m, Sequence):
    return type(m)(deepmap(f, x) for x in m)
  elif isinstance(m, Mapping):
    return type(m)((k, deepmap(f, m[k])) for k in m)
  else:
    raise AttributeError()


def float64_to_float32(x):
    return np.asarray(x, np.float32) if x.dtype == np.float64 else x


# inverted_pendulum.InvertedPendulumEnv.jitter_step_start = jitter_step_start
# inverted_pendulum.InvertedPendulumEnv.jitter_step_end = jitter_step_end