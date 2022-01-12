import gym
import numpy as np
# from gym.envs.mujoco import inverted_pendulum
import types
import os
import random
__all__ = ["make_env", "create_folders", "get_frame_skip_and_timestep", "perform_action", "random_jitter_force",
           "random_disturb", "const_disturb_five", "const_jitter_force"]


# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip, delayed_env):
    env = gym.make(env_name)
    if delayed_env:
        env = Float64ToFloat32(env)
        env = RealTimeWrapper(env)
        env.env.env._max_episode_steps = 1000 * time_change_factor
        env.env.env.frame_skip = int(frameskip)
        env.env.env.env.frame_skip = int(frameskip)
    else:
        if env_name == 'InvertedPendulum-v2':
            env.env.jitter_step_end = types.MethodType(jitter_step_end_pendulum, env.env)
            env.env.jitter_step_start = types.MethodType(jitter_step_start_pendulum, env.env)
        elif env_name == 'HalfCheetah-v2':
            env.env.jitter_step_end = types.MethodType(jitter_step_end_cheetah, env.env)
            env.env.jitter_step_start = types.MethodType(jitter_step_start_cheetah, env.env)
        env._max_episode_steps = 1000 * time_change_factor
    env.seed(seed)
    env.delayed = delayed_env
    env.action_space.seed(seed)
    env.model.opt.timestep = env_timestep
    # env.env.env.frame_skip = int(frameskip)
    env.env.frame_skip = int(frameskip)
    env.frame_skip = int(frameskip)

    return env


# The alternative step function when some frames of a step are under the
# jitter force while others are not
def jitter_step_end_pendulum(self, a, force, frames1, frames2):
    self.model.opt.gravity[0] = force
    reward = 1.0
    self.do_simulation(a, int(round(frames1)))
    self.model.opt.gravity[0] = 0 # force # 0 here? frames1 are with force while frames2 are supposed not.
    self.do_simulation(a, int(round(frames2)))
    ob = self._get_obs()
    notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
    done = not notdone
    return ob, reward, done, {}


def jitter_step_start_pendulum(self, a, force, frames1, frames2, jit_frames):
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

# The alternative step function when some frames of a step are under the
# jitter force while others are not
def jitter_step_end_cheetah(self, a, force, frames1, frames2):
    xposbefore = self.sim.data.qpos[0]
    self.model.opt.gravity[0] = force
    self.do_simulation(a, int(round(frames1)))
    self.model.opt.gravity[0] = 0 # force # 0 here? frames1 are with force while frames2 are supposed not.
    self.do_simulation(a, int(round(frames2)))
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(a).sum()
    reward_run = (xposafter - xposbefore) / self.dt
    reward = reward_ctrl + reward_run
    done = False
    return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)


def jitter_step_start_cheetah(self, a, force, frames1, frames2, jit_frames):
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(a, int(frames1))
    self.model.opt.gravity[0] = force

    if frames2 < jit_frames:
        self.do_simulation(a, int(round(frames2)))
    else:
        self.do_simulation(a, int(round(jit_frames)))
        self.model.opt.gravity[0] = 0
        self.do_simulation(a, int(round(frames2 - jit_frames)))
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(a).sum()
    reward_run = (xposafter - xposbefore) / self.dt
    reward = reward_ctrl + reward_run
    done = False
    return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)


class RealTimeWrapper(gym.Wrapper):
    # todo implement for other environments
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


def create_folders():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")


def get_frame_skip_and_timestep(jit_duration, response_rate, reflex_response_rate=None):

    if reflex_response_rate:
        smallest_step = jit_duration if jit_duration < reflex_response_rate else reflex_response_rate
    else:
        smallest_step = jit_duration if jit_duration < response_rate else response_rate

    frame_skip = response_rate / smallest_step
    timestep = smallest_step
    if jit_duration % timestep == 0:
        jit_frames = int(jit_duration / timestep)
    else:
        raise ValueError("jit_duration should be a multiple of the timestep: " + str(timestep))

    if response_rate % timestep != 0:
        raise ValueError("response_rate should be a multiple of the timestep: " + str(timestep))

    if reflex_response_rate and reflex_response_rate % timestep != 0:
        raise ValueError("reflex_response_rate should be a multiple of the timestep: " + str(timestep))

    return frame_skip, timestep, jit_frames


def env_step(env, reflex, action, reflex_frames, frame_skip):
    if reflex:
        next_state, reward, done, _ = env.jitter_step_end(reflex, 0, reflex_frames, 0)  # prev action happens for a short time until reflex kicks in
        next_state, reward, done, _ = env.jitter_step_end(action, 0, frame_skip - reflex_frames, 0)
    else:
        next_state, reward, done, _ = env.step(action)
    return next_state, reward, done


def random_jitter_force(force):
    return np.random.random() * force * (2 * (np.random.random() > 0.5) - 1), force  # Jitter force strength w/ direction


def const_jitter_force(force):
    return force * (2 * (np.random.random() > 0.5) - 1), force + (0.25 * 9.81)


def random_disturb(catastrophe_frequency):
    return round(random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency), 3)


def const_disturb_five(catastrophe_frequency):
    return 5


def perform_action(jittering, disturb, counter, response_rate, env, reflex, action, reflex_frames, frame_skip, get_jitter_force, max_force, timestep, jit_frames, jittered_frames, get_next_disturb, jitter_force, catastrophe_frequency):

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
        disturb = get_next_disturb(catastrophe_frequency)

    if not jittering:
        if round(disturb - counter, 3) >= response_rate:  # Not during the frames when jitter force keeps existing
            next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
            counter += response_rate
        elif round(disturb - counter, 3) < response_rate: # jitter force starts
            jitter_force, max_force = get_jitter_force(max_force)
            frames_simulated = 0
            force_frames_simulated = 0
            if reflex:
                if round(disturb - counter, 3) / timestep >= reflex_frames:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, 0, reflex_frames, 0)
                    frames_simulated += reflex_frames
                elif round(disturb - counter, 3) / timestep < reflex_frames:
                    reflex_frames_until_disturb = round(disturb - counter, 3) / timestep
                    reflex_frames_after_disturb = reflex_frames - reflex_frames_until_disturb
                    next_state, reward, done, _ = env.jitter_step_end(reflex, jitter_force, reflex_frames_until_disturb,
                                                                      reflex_frames_after_disturb)
                    frames_simulated += reflex_frames
                    force_frames_simulated += reflex_frames_after_disturb

            frames_until_disturb = max((round(disturb - counter, 3) / timestep) - frames_simulated, 0)
            frames_after_disturb = frame_skip - max((round(disturb - counter, 3) / timestep), frames_simulated)
            next_state, reward, done, _ = env.jitter_step_start(action, jitter_force, frames_until_disturb,
                                                                frames_after_disturb, jit_frames - force_frames_simulated)

            jittered_frames = frame_skip - (round(disturb - counter, 3) / timestep)
            if jittered_frames >= jit_frames:
                stop_force()
            else:
                jittering = True
                env.model.opt.gravity[0] = jitter_force
                counter += response_rate
    else:
        if jit_frames - jittered_frames < frame_skip:  # Jitter force will dispear from now!
            frames_simulated = 0
            if reflex:
                if reflex_frames <= jit_frames - jittered_frames:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, jitter_force, reflex_frames, 0)
                else:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, jitter_force,
                                                                      jit_frames - jittered_frames,
                                                                      reflex_frames - (jit_frames - jittered_frames))
                frames_simulated += reflex_frames

            frames_until_end = max(jit_frames - jittered_frames - frames_simulated, 0)
            frames_after_end = frame_skip - max(jit_frames - jittered_frames, frames_simulated)
            next_state, reward, done, _ = env.jitter_step_end(action, jitter_force, frames_until_end, frames_after_end)
            stop_force()
        else:  # Jitter force keeps existing now!
            env.model.opt.gravity[0] = jitter_force
            next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
            jittered_frames += frame_skip
            counter += response_rate
            if jittered_frames == jit_frames:
                stop_force()

    return jittering, disturb, counter, jittered_frames, jitter_force, max_force, next_state, reward, done
