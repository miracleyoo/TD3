import gym
import numpy as np

__all__=["make_env", "jitter_step"]

def make_env(env_name, seed, time_change_factor, env_timestep, frameskip):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env._max_episode_steps = 1000 * time_change_factor
    env.model.opt.timestep = env_timestep
    env.frameskip = frameskip
    env.jitter_step = jitter_step
    return env

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