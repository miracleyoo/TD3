from common import make_env
import numpy as np
import random
import sys
sys.path.append('../')


__all__ = ["eval_policy_std", "eval_policy_ori"]


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
def eval_policy_ori(policy, env_name, eval_episodes=10, time_change_factor=1, jit_duration=0, env_timestep=0.02, force=1,
                    frame_skip=1, jit_frames=0, response_rate=0.04):
    print("==> Start standard evaluation...")

    eval_env = make_env(env_name, 100, time_change_factor,
                        env_timestep, frame_skip)

    avg_reward = 0.
    if jit_duration:
        counter = 0
        disturb = random.randint(50, 100) * 0.04
        print("==> Using Horizontal Jitter!")

    jittering = False
    t = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            # Perform action

            if jit_duration:
                if not jittering and disturb - counter >= response_rate:  # Not during the frames when jitter force keeps existing
                    next_state, reward, done, _ = env.step(action)
                    counter += response_rate
                    # print(next_state)
                elif not jittering and disturb - counter < response_rate:
                    jitter_force = np.random.random() * hori_force * (
                                2 * (np.random.random() > 0.5) - 1)  # Jitter force strength w/ direction
                    env.jitter_step_start(action, jitter_force, (disturb - count) / timestep,
                                          frame_skip - ((disturb - count) / timestep), jit_frames)
                    jittered_frames = frame_skip - ((disturb - count) / timestep)
                    if jittered_frames >= jit_frames:
                        jittered_frames = 0
                        jittering = False
                        self.model.opt.gravity[0] = 0
                    else:
                        jittering = True
                        self.model.opt.gravity[0] = jitter_force

                    counter += response_rate

                elif jit_frames - jittered_frames < frame_skip:  # Jitter force will dispear from now!
                    next_state, reward, done, _ = env.jitter_step_end(
                        action, jitter_force, jit_frames - jittered_frames, frame_skip - (jit_frames - jittered_frames))
                    jittering = False  # Stop jittering now
                    disturb = random.randint(50, 100) * 0.04  # Define the next jittering frame
                    env.model.opt.gravity[0] = 0
                    counter = 0
                else:  # Jitter force keeps existing now!
                    next_state, reward, done, _ = env.step(action)
                    jittered_frames += frame_skip
                    if jittered_frames == jit_frames:
                        jittering = False
                        disturb = random.randint(50, 100) * 0.04
                        env.model.opt.gravity[0] = 0
                        counter = 0
            else:
                next_state, reward, done, _ = env.step(action)
                counter += response_rate

            avg_reward += reward
            state = next_state
            if jit_duration:
                if counter == disturb:
                    jitter_force = np.random.random() * force * (2*(random.random() > 0.5)-1)
                    eval_env.model.opt.gravity[0] = jitter_force
                    jittering = True
                    jittered_frames = 0

            t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
