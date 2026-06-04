import time
import numpy as np
from env import DoublePendulum  # adjust if your file/module name differs


def smoke_rgb_array():
    print("=== RGB_ARRAY SMOKE TEST ===")
    env = DoublePendulum(
        mass0=1.0,
        mass1=1.0,
        length0=0.2,
        length1=0.2,
        timestep=0.001,
        render_mode="rgb_array",
        render_width=640,
        render_height=480,
    )

    obs, info = env.reset(seed=0)
    print("reset ok | obs:", obs, "| info keys:", list(info.keys()))

    # one render right away
    frame = env.render()
    print("render ok | frame shape:", frame.shape, "| dtype:", frame.dtype)

    # step a bit
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(obs).all():
            raise RuntimeError(f"non-finite obs at step {i}: {obs}")

        if terminated or truncated:
            print("episode ended early at step", i, "terminated:", terminated, "truncated:", truncated)
            break

    env.close()
    print("rgb_array close ok\n")


def smoke_human():
    """Run 3 episodes back-to-back in the same viewer window (~2 s each).

    Each episode gets a freshly randomised ground plane (depth and tilt).
    The viewer window should stay open across all three resets — this is the
    primary thing being tested here.
    """
    print("=== HUMAN SMOKE TEST (WINDOW OPENS) ===")
    env = DoublePendulum(
        mass0=1.0,
        mass1=1.0,
        length0=0.3,
        length1=0.4,
        timestep=0.001,
        slide=True,
        render_mode="human",
    )

    n_episodes = 3
    episode_duration = 2.0  # seconds of wall-clock time per episode

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        floor_id     = env.model.geom("floor").id
        ground_depth = env.model.geom_pos[floor_id][2]
        ground_quat  = env.model.geom_quat[floor_id]
        tilt_deg     = float(np.degrees(2 * np.arccos(np.clip(ground_quat[0], -1, 1))))
        print(f"episode {ep+1}/{n_episodes} | ground depth={ground_depth:.3f} m  tilt={tilt_deg:.1f}°")

        t_end = time.time() + episode_duration
        steps = 0
        while time.time() < t_end:
            obs, reward, terminated, truncated, info = env.step((0, 0))
            steps += 1
            if terminated or truncated:
                print(f"  simulation blew up at step {steps}, restarting episode")
                obs, info = env.reset(seed=ep * 100 + steps)
                break

        print(f"  done | steps: {steps}")

    env.close()
    print("human close ok\n")


if __name__ == "__main__":
    smoke_rgb_array()
    smoke_human()
    print("ALL SMOKE TESTS PASSED ✅")
