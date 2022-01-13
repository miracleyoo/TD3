from common import make_env
import numpy as np
import random
import sys
from common import perform_action, random_disturb, random_jitter_force, const_jitter_force, const_disturb_five
sys.path.append('../')


__all__ = ["eval_policy_std", "eval_policy_ori", "eval_policy_increasing_force"]


# Standard evaluation function. A fixed evaluation routine will be executed to
# check the ability of the learned policy. Average step angle will be returned.
def eval_policy_std(policy, env_name, eval_episodes=10, time_change_factor=1, jit_duration=0, env_timestep=0.02, force=1,
                    frame_skip=1, jit_frames=0):
    print("\n==> Start standard evaluation...")

    eval_env = make_env(env_name, 100, time_change_factor, env_timestep, frame_skip)

    avg_reward = 0.
    avg_angle = 0.
    # Take the 1/10, 2/10, ... 10/10 of the max force for testing. Direction is alternate.
    force_ratios = np.linspace(0, 1, 11)[1:]
    # Split the whole time period into 11 pieces to test the forces above.
    # No force is added in the first time block.
    disturbs = np.linspace(0, 1, len(force_ratios) +2)[1:-1] * 1000 * time_change_factor

    disturb = disturbs[0]
    disturbs = disturbs[1:]
    disturb_idx = 0
    counter = 1
    steps = 0

    jittering = False
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            # Perform action
            if not jittering:
                next_state, reward, done, _ = eval_env.step(action)
            elif jit_frames - jittered_frames < frame_skip:
                next_state, reward, done, _ = eval_env.jitter_step(action, jitter_force, jit_frames - jittered_frames,
                                                                   frame_skip - (jit_frames - jittered_frames))
                jittering = False
                disturb = disturbs[disturb_idx]
                eval_env.model.opt.gravity[0] = 0
                counter = 1
            else:
                next_state, reward, done, _ = eval_env.step(action)
                jittered_frames += frame_skip
                if jittered_frames == jit_frames:
                    jittering = False
                    disturb = disturbs[disturb_idx]
                    eval_env.model.opt.gravity[0] = 0
                    disturb_idx += 1
                    counter = 1

            avg_reward += reward
            avg_angle += abs(next_state[1])
            steps += 1
            state = next_state

            if jit_duration:
                if counter % disturb == 0:
                    jitter_force = force_ratios[disturb_idx] * \
                        force * (-1)**disturb_idx
                    eval_env.model.opt.gravity[0] = jitter_force
                    jittering = True
                    jittered_frames = 0
                counter += 1

    avg_reward /= eval_episodes
    avg_angle /= steps
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Avg Angle: {avg_angle:.5f}")
    print("---------------------------------------\n")
    return avg_reward, avg_angle


# Origianl evaluation method
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def eval_policy_ori(policy, env_name, eval_episodes=10, time_change_factor=1, jit_duration=0, env_timestep=0.02,
                    max_force=1 * 9.81, frame_skip=1, jit_frames=0, response_rate=0.04, delayed_env=False,
                    reflex_frames=None, vertical_gravity=None, catastrophe_frequency=1):
    print("==> Start standard evaluation...")

    eval_env = make_env(env_name, 100, time_change_factor, env_timestep, frame_skip, delayed_env)
    if vertical_gravity is not None:
        eval_env.model.opt.gravity[2] = vertical_gravity

    avg_reward = 0.

    t = 0
    for _ in range(eval_episodes):
        jittering = False
        jitter_force = 0
        reflex = False
        state, done = eval_env.reset(), False
        if jit_duration:
            jittered_frames = 0
            jittering = False
            eval_env.model.opt.gravity[0] = 0
            counter = 0
            disturb = round(random.randint(50, 100) * 0.04 * (1 / catastrophe_frequency), 3)
        while not done:
            if not reflex_frames:
                action = policy.select_action(state)
            else:
                reflex, action = policy.select_action(state)
            # Perform action

            jittering, disturb, counter, jittered_frames, jitter_force, max_force, next_state, reward, done = perform_action(
                jittering, disturb, counter, response_rate, eval_env, reflex, action, reflex_frames, frame_skip, random_jitter_force,
                max_force, env_timestep, jit_frames, jittered_frames, random_disturb, jitter_force, catastrophe_frequency, delayed_env)

            counter = round(counter, 3)
            avg_reward += reward
            state = next_state
            if counter == disturb:
                jitter_force, _ = random_jitter_force(max_force)
                eval_env.model.opt.gravity[0] = jitter_force
                jittering = True
                jittered_frames = 0

            t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def eval_policy_increasing_force(policy, env_name, eval_episodes=10, time_change_factor=1,
                                 env_timestep=0.02, frame_skip=1, jit_frames=0, response_rate=0.04, delayed_env=False,
                                 reflex_frames=None):
    eval_env = make_env(env_name, 100, time_change_factor, env_timestep, frame_skip, delayed_env)
    if delayed_env:
        eval_env.env.env._max_episode_steps = 100000
    else:
        eval_env._max_episode_steps = 100000
    avg_reward = 0.
    avg_angle = 0.
    actions = 0


    t = 0
    forces = []
    force_times = []
    for _ in range(eval_episodes):
        eval_env.model.opt.gravity[0] = 0
        counter = 0
        disturb = 5
        force = 0.25 * 9.81
        prev_action = None
        jerk = 0
        jittered_frames = 0
        jittering = False
        jitter_force = 0
        reflex = False
        state, done = eval_env.reset(), False

        while not done:
            if not reflex_frames:
                action = policy.select_action(state)
            else:
                reflex, action = policy.select_action(state)

            if reflex:
                actions += 2
            else:
                actions += 1
            # Perform action
            jittering, disturb, counter, jittered_frames, jitter_force, force, next_state, reward, done = perform_action(
                jittering, disturb, counter, response_rate, eval_env, reflex, action, reflex_frames, frame_skip,
                const_jitter_force, force, env_timestep, jit_frames, jittered_frames, const_disturb_five, jitter_force, 1, delayed_env)

            avg_reward += reward
            avg_angle += abs(next_state[1])
            counter = round(counter, 3)
            state = next_state
            if counter == disturb:
                jitter_force, force = const_jitter_force(force)
                eval_env.model.opt.gravity[0] = jitter_force
                jittering = True
                jittered_frames = 0

            t += 1
            if prev_action:
                jerk += abs(action[0] - prev_action)
            prev_action = action[0]

    avg_reward /= eval_episodes
    avg_angle /= eval_episodes
    jerk /= t
    actions /= eval_episodes
    return avg_reward, avg_angle, jerk, actions


