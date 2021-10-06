import gym
import numpy as np
from gym.envs.mujoco import inverted_pendulum
__all__=["make_env", "jitter_step"]


# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip):
    env = gym.make(env_name)
    env.seed(seed)
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
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = 0 # force # 0 here? frames1 are with force while frames2 are supposed not.
    self.do_simulation(a, int(frames2))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}


def jitter_step_start(self, a, force, frames1, frames2, jit_frames):
    reward = 1.0
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = force
    if frames2 < jit_frames:
        self.do_simulation(a, int(frames2))
    else:
        self.do_simulation(a, int(jit_frames))
        self.model.opt.gravity[0] = 0
        self.do_simulation(a, int(frames2 - jit_frames))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}


inverted_pendulum.InvertedPendulumEnv.jitter_step_start = jitter_step_start
inverted_pendulum.InvertedPendulumEnv.jitter_step_end = jitter_step_end