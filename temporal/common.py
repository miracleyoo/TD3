import gym
import numpy as np
import torch
# from gym.envs.mujoco import inverted_pendulum
import types
import os
import random
__all__ = ["make_env", "create_folders", "get_frame_skip_and_timestep", "perform_action", "random_jitter_force",
           "random_disturb", "const_disturb_five", "const_jitter_force"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make environment using its name
def make_env(env_name, seed, time_change_factor, env_timestep, frameskip, delayed_env):
    env = gym.make(env_name)
    if delayed_env:
        env = Float64ToFloat32(env)
        env = RealTimeWrapper(env, env_name)
        env.env.env._max_episode_steps = 1000 * time_change_factor
        env.env.env.frame_skip = int(frameskip)
        env.env.env.env.frame_skip = int(frameskip)
    else:
        if env_name == 'InvertedPendulum-v2':
            env = JitterWrapper(env)
            env.env._max_episode_steps = 1000 * time_change_factor
            env.env.env.frame_skip = int(frameskip)
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


class JitterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def jitter_step_end(self,  a, force, frames1, frames2):
        assert (
                self.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        self.model.opt.gravity[0] = force
        reward = 1.0
        self.do_simulation(a, int(round(frames1)))
        self.model.opt.gravity[0] = 0  # force # 0 here? frames1 are with force while frames2 are supposed not.
        self.do_simulation(a, int(round(frames2)))
        ob = self.env.env._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone
        if self.env._elapsed_steps >= self.env._max_episode_steps:
            done = True
        return ob, reward, done, {}

    def jitter_step_start(self, a, force, frames1, frames2, jit_frames):
        assert (
                self.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        reward = 1.0
        self.do_simulation(a, int(frames1))
        self.model.opt.gravity[0] = force
        if frames2 < jit_frames:
            self.do_simulation(a, int(round(frames2)))
        else:
            self.do_simulation(a, int(round(jit_frames)))
            self.model.opt.gravity[0] = 0
            self.do_simulation(a, int(round(frames2 - jit_frames)))
        ob = self.env.env._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone
        if self.env._elapsed_steps >= self.env._max_episode_steps:
            done = True
        return ob, reward, done, {}


class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        # self.initial_action = env.action_space.sample()
        assert isinstance(env.action_space, gym.spaces.Box)
        self.initial_action = env.action_space.high * 0
        self.previous_action = self.initial_action
        if env_name == 'InvertedPendulum-v2':
            self.jitter_step_end = self.jitter_step_end_InvertedPendulum
            self.jitter_step_start = self.jitter_step_start_InvertedPendulum
        elif env_name == 'Hopper-v2':
            self.jitter_step_end = self.jitter_step_end_Hopper
            self.jitter_step_start = self.jitter_step_start_Hopper
        elif env_name == 'Walker2d-v2':
            self.jitter_step_end = self.jitter_step_end_Walker
            self.jitter_step_start = self.jitter_step_start_Walker
        elif env_name == 'InvertedDoublePendulum-v2':
            self.jitter_step_end = self.jitter_step_end_InvertedDoublePendulum
            self.jitter_step_start = self.jitter_step_start_InvertedDoublePendulum

    def reset(self):
        self.previous_action = self.initial_action
        return np.concatenate((super().reset(), self.previous_action), axis=0)

    def step(self, action):
        observation, reward, done, info = super().step(self.previous_action)
        self.previous_action = action
        return np.concatenate((observation, action), axis=0), reward, done, info

    def jitter_step_end_InvertedPendulum(self, a, force, frames1, frames2):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
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
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            done = True
        return ob, reward, done, {}

    def jitter_step_start_InvertedPendulum(self, a, force, frames1, frames2, jit_frames):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        action = self.previous_action
        reward = 1.0
        self.model.opt.gravity[0] = 0
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
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            done = True
        return ob, reward, done, {}

    def jitter_step_end_InvertedDoublePendulum(self, a, force, frames1, frames2):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        action = self.previous_action
        self.model.opt.gravity[0] = force
        self.do_simulation(action, int(round(frames1)))
        self.model.opt.gravity[0] = 0  # force # 0 here? frames1 are with force while frames2 are supposed not.
        self.do_simulation(action, int(round(frames2)))
        ob = self.env.env.env._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        terminated = bool(y <= 1)
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, r, terminated, {}

    def jitter_step_start_InvertedDoublePendulum(self, a, force, frames1, frames2, jit_frames):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        action = self.previous_action
        self.model.opt.gravity[0] = 0
        self.do_simulation(action, int(frames1))
        self.model.opt.gravity[0] = force
        if frames2 < jit_frames:
            self.do_simulation(action, int(round(frames2)))
        else:
            self.do_simulation(action, int(round(jit_frames)))
            self.model.opt.gravity[0] = 0
            self.do_simulation(action, int(round(frames2 - jit_frames)))
        ob = self.env.env.env._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        terminated = bool(y <= 1)
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, r, terminated, {}

    def jitter_step_end_Hopper(self, a, force, frames1, frames2):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        action = self.previous_action
        posbefore = self.sim.data.qpos[0]
        self.model.opt.gravity[0] = force
        reward = 1.0
        self.do_simulation(action, int(round(frames1)))
        self.model.opt.gravity[0] = 0  # force # 0 here? frames1 are with force while frames2 are supposed not.
        self.do_simulation(action, int(round(frames2)))

        posafter, height, ang = self.sim.data.qpos[0:3]

        ob = self.env.env.env._get_obs()

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (
                np.isfinite(s).all()
                and (np.abs(s[2:]) < 100).all()
                and (height > 0.7)
                and (abs(ang) < 0.2)
        )
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, reward, terminated, {}

    def jitter_step_start_Hopper(self, a, force, frames1, frames2, jit_frames):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        action = self.previous_action
        posbefore = self.sim.data.qpos[0]
        self.model.opt.gravity[0] = 0
        self.do_simulation(action, int(frames1))
        self.model.opt.gravity[0] = force
        if frames2 < jit_frames:
            self.do_simulation(action, int(round(frames2)))
        else:
            self.do_simulation(action, int(round(jit_frames)))
            self.model.opt.gravity[0] = 0
            self.do_simulation(action, int(round(frames2 - jit_frames)))

        posafter, height, ang = self.sim.data.qpos[0:3]

        ob = self.env.env.env._get_obs()
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (
                np.isfinite(s).all()
                and (np.abs(s[2:]) < 100).all()
                and (height > 0.7)
                and (abs(ang) < 0.2)
        )
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, reward, terminated, {}

    def jitter_step_end_Walker(self, a, force, frames1, frames2):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        action = self.previous_action
        posbefore = self.sim.data.qpos[0]
        self.model.opt.gravity[0] = force
        reward = 1.0
        self.do_simulation(action, int(round(frames1)))
        self.model.opt.gravity[0] = 0  # force # 0 here? frames1 are with force while frames2 are supposed not.
        self.do_simulation(action, int(round(frames2)))

        posafter, height, ang = self.sim.data.qpos[0:3]

        ob = self.env.env.env._get_obs()

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (
                np.isfinite(s).all()
                and (height > 0.8)
                and (height < 2.0)
                and (abs(ang) < 1)
        )
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, reward, terminated, {}

    def jitter_step_start_Walker(self, a, force, frames1, frames2, jit_frames):
        assert (
                self.env.env._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        action = self.previous_action
        posbefore = self.sim.data.qpos[0]
        self.model.opt.gravity[0] = 0
        self.do_simulation(action, int(frames1))
        self.model.opt.gravity[0] = force
        if frames2 < jit_frames:
            self.do_simulation(action, int(round(frames2)))
        else:
            self.do_simulation(action, int(round(jit_frames)))
            self.model.opt.gravity[0] = 0
            self.do_simulation(action, int(round(frames2 - jit_frames)))

        posafter, height, ang = self.sim.data.qpos[0:3]

        ob = self.env.env.env._get_obs()
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (
                np.isfinite(s).all()
                and (height > 0.8)
                and (height < 2.0)
                and (abs(ang) < 1)
        )
        self.previous_action = a
        ob = np.concatenate((ob, a), axis=0)
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps:
            terminated = True
        return ob, reward, terminated, {}

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


def get_frame_skip_and_timestep(jit_duration, response_rate, default_timestep, reflex_response_rate=None):

    if reflex_response_rate:
        timestep = jit_duration if jit_duration < reflex_response_rate else reflex_response_rate
    else:
        timestep = jit_duration if jit_duration < response_rate else (response_rate if response_rate < default_timestep else default_timestep)

    frame_skip = int(response_rate / timestep)
    if round((jit_duration / timestep), 3) % 1 == 0:
        jit_frames = int(jit_duration / timestep)  # number of frames for the perturbation
    else:
        raise ValueError("jit_duration should be a multiple of the timestep: " + str(timestep))

    if round((response_rate / timestep), 3) % 1 != 0:
        raise ValueError("response_rate should be a multiple of the timestep: " + str(timestep))

    if reflex_response_rate and (reflex_response_rate / timestep) % 1 != 0:
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
    return np.random.random() * force * (2 * (np.random.random() > 0.5) - 1)  # Jitter force strength w/ direction


def const_jitter_force(force):
    return force * (2 * (np.random.random() > 0.5) - 1), force + (0.25 * 9.81)


def random_disturb(catastrophe_frequency):
    return round(random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency), 3)


def const_disturb_five(catastrophe_frequency):
    return 4.98

def const_disturb_half(catastrophe_frequency):
    return 1


def perform_action(jittering, disturb, elapsed_time, response_rate, env, reflex, action, reflex_frames, frame_skip, get_jitter_force, max_force, timestep, jit_frames, jittered_frames, get_next_disturb, jitter_force, catastrophe_frequency, delayed_env):

    current_steps = env.env.env._elapsed_steps if delayed_env else env.env._elapsed_steps

    if reflex:
        reflex += env.previous_action + reflex
        reflex = np.clip(reflex, -env.action_space.high[0], env.action_space.high[0])

    def stop_force():
        nonlocal jittered_frames
        nonlocal jittering
        nonlocal env
        nonlocal elapsed_time
        nonlocal disturb

        jittered_frames = 0
        jittering = False
        env.model.opt.gravity[0] = 0
        elapsed_time = 0
        disturb = get_next_disturb(catastrophe_frequency)

    if not jittering:
        if round(disturb - elapsed_time, 3) >= response_rate:  # Not during the frames when jitter force keeps existing
            next_state, reward, done = env_step(env, reflex, action, reflex_frames, frame_skip)
            elapsed_time += response_rate
        elif round(disturb - elapsed_time, 3) < response_rate: # jitter force starts
            jitter_force, max_force = get_jitter_force(max_force)
            frames_simulated = 0
            force_frames_simulated = 0
            if reflex:
                if round(disturb - elapsed_time, 3) / timestep >= reflex_frames:
                    next_state, reward, done, _ = env.jitter_step_end(reflex, 0, reflex_frames, 0)
                    frames_simulated += reflex_frames
                elif round(disturb - elapsed_time, 3) / timestep < reflex_frames:
                    reflex_frames_until_disturb = round(disturb - elapsed_time, 3) / timestep
                    reflex_frames_after_disturb = reflex_frames - reflex_frames_until_disturb
                    next_state, reward, done, _ = env.jitter_step_start(reflex, jitter_force, reflex_frames_until_disturb,
                                                                      reflex_frames_after_disturb, jit_frames)
                    frames_simulated += reflex_frames
                    force_frames_simulated += min(reflex_frames_after_disturb, jit_frames)

            frames_until_disturb = max((round(disturb - elapsed_time, 3) / timestep) - frames_simulated, 0)
            frames_after_disturb = frame_skip - max((round(disturb - elapsed_time, 3) / timestep), frames_simulated)
            next_state, reward, done, _ = env.jitter_step_start(action, jitter_force, frames_until_disturb,
                                                                frames_after_disturb, jit_frames - force_frames_simulated)

            jittered_frames = frame_skip - (round(disturb - elapsed_time, 3) / timestep)
            if jittered_frames >= jit_frames:
                stop_force()
            else:
                jittering = True
                env.model.opt.gravity[0] = jitter_force
                elapsed_time += response_rate
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
            elapsed_time += response_rate
            if jittered_frames == jit_frames:
                stop_force()

    updated_steps = env.env.env._elapsed_steps if delayed_env else env.env._elapsed_steps

    if current_steps == updated_steps:
        if delayed_env:
            env.env.env._elapsed_steps += 1
        else:
            env.env._elapsed_steps += 1

    return jittering, disturb, elapsed_time, jittered_frames, jitter_force, next_state, reward, done, max_force


def get_TD(parent_policy, state, next_state, reward, done):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    next_state = torch.FloatTensor(next_state.reshape(1, -1)).to(device)
    q = parent_policy.critic.Q1(state, parent_policy.actor(state).clamp(-parent_policy.max_action, parent_policy.max_action))[0][0]
    target_Q = reward + (not done) * parent_policy.discount * parent_policy.critic.Q1(next_state, parent_policy.actor(next_state).clamp(-parent_policy.max_action, parent_policy.max_action))[0][0]
    TD = (target_Q - q).detach().cpu().data.numpy()
    return abs(TD)


def get_Q(parent_policy, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    q = parent_policy.critic.Q1(state, parent_policy.actor(state).clamp(-parent_policy.max_action, parent_policy.max_action))[0][0]
    return q.detach().cpu().numpy()
