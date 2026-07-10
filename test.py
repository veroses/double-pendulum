import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # env loop runs off the main thread under the human viewer
import matplotlib.pyplot as plt
import mujoco
from env import DoublePendulum  # adjust if your file/module name differs


def test_spaces_and_dims():
    """Verify the 3D ball-joint env exposes the right shapes and valid state.

    Checks:
      - observation space is 14D (8 quaternion + 6 angular velocity)
      - action space is 6D (3 torques per joint)
      - MuJoCo model has nq=8, nv=6, nu=6
      - reset() produces unit-norm quaternions for both joints
      - info["state"] has the expected length and embeds valid rotation matrices
    """
    print("=== DIMS / SPACES TEST ===")
    env = DoublePendulum(mass0=1.0, mass1=1.0, length0=0.3, length1=0.4)
    obs, info = env.reset(seed=0)

    assert env.observation_space.shape == (14,), env.observation_space.shape
    assert env.action_space.shape == (6,), env.action_space.shape
    assert obs.shape == (14,), obs.shape
    assert env.model.nq == 8 and env.model.nv == 6 and env.model.nu == 6, \
        (env.model.nq, env.model.nv, env.model.nu)

    # Both joint quaternions (obs[0:4] and obs[4:8]) should be unit norm.
    for j, q in enumerate((obs[0:4], obs[4:8])):
        n = np.linalg.norm(q)
        assert abs(n - 1.0) < 1e-6, f"joint{j} quaternion not unit norm: {n}"

    # State vector length: 3+6+6+6+3+3+6 + 2*K*10 + 2*K + 3 (rotations are 6D now)
    K = env.segments_per_link
    expected = 3 + 6 + 6 + 6 + 3 + 3 + 6 + 2*K*10 + 2*K + 3
    state = info["state"]
    assert state.shape == (expected,), (state.shape, expected)

    # The two joint orientations are stored as 6D rotations at state[9:15] and
    # state[15:21]. Reconstruct each and verify it is a valid SO(3) matrix.
    from utils import rotation_from_6d
    for j, d6 in enumerate((state[9:15], state[15:21])):
        R = rotation_from_6d(d6)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), f"joint{j} R not orthonormal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"joint{j} R det != 1"

    env.close()
    print(f"dims ok | obs=14 action=6 state={expected}\n")


def test_3d_motion():
    """Verify the pendulum actually moves in 3D (impossible with the old planar hinge).

    Starting from rest, a pure torque about joint0's local x-axis rotates link0 in
    the YZ plane, so link1's body must develop a non-zero world y-position. With the
    previous Y-axis hinge joints the whole system was confined to the XZ plane and y
    would stay identically zero.
    """
    print("=== 3D MOTION TEST ===")
    env = DoublePendulum(mass0=1.0, mass1=1.0, length0=0.3, length1=0.4, timestep=0.001)
    env.reset(seed=0)

    # Force a clean rest state: identity orientations, zero velocity.
    env.data.qpos[:] = [1, 0, 0, 0, 1, 0, 0, 0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    link1_id = env.body_ids["link1"]
    assert abs(env.data.xpos[link1_id][1]) < 1e-9, "link1 should start at y=0"

    # Torque about joint0 x-axis only.
    action = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    max_abs_y = 0.0
    for _ in range(200):
        env.step(action)
        max_abs_y = max(max_abs_y, abs(env.data.xpos[link1_id][1]))

    assert max_abs_y > 1e-3, f"expected out-of-plane motion, max|y|={max_abs_y:.2e}"
    env.close()
    print(f"3d motion ok | max|y| of link1 = {max_abs_y:.4f} m\n")


def test_spawn_and_tau_sampling():
    """Verify rejection-sampled spawns and exponentially smoothed torque sampling.

    Checks:
      - with slide=True, no episode ever starts with a link penetrating the floor
      - sample_tau() stays within the per-axis limits
      - spin-axis (local z) torques are identically zero
      - consecutive torques respect the exponential-smoothing step bound
    """
    print("=== SPAWN / TAU SAMPLING TEST ===")
    env = DoublePendulum(mass0=1.0, mass1=1.0, length0=0.3, length1=0.4, slide=True)

    for i in range(100):
        env.reset(seed=i)
        assert not env._spawn_penetrating(), f"seed {i}: spawned inside the ground"

    env.reset(seed=0)
    prev = env.torques.copy()
    max_step = (1.0 - env.tau_alpha) * 2.0 * env.tau_limit + 1e-12
    for t in range(200):
        tau = env.sample_tau()
        assert tau.shape == (6,)
        assert np.all(np.abs(tau) <= env.tau_limit + 1e-12), f"step {t}: tau out of bounds"
        assert tau[2] == 0.0 and tau[5] == 0.0, f"step {t}: spin-axis torque non-zero"
        assert np.all(np.abs(tau - prev) <= max_step), f"step {t}: tau jump too large"
        _, _, terminated, _, _ = env.step(tau)
        if terminated:
            env.reset(seed=1000 + t)
        prev = env.torques.copy()

    env.close()
    print("spawn/tau ok | 100 penetration-free spawns, 200 smooth bounded torques\n")


def smoke_rgb_array():
    """Verify rgb_array mode returns one frame per camera and stepping stays finite."""
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
    print("reset ok | obs shape:", obs.shape, "| info keys:", list(info.keys()))

    frames = env.render()
    assert isinstance(frames, tuple) and len(frames) == len(env.CAMERA_NAMES), \
        f"expected {len(env.CAMERA_NAMES)} frames, got {type(frames)}"
    for cam, frame in zip(env.CAMERA_NAMES, frames):
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, (cam, frame.shape, frame.dtype)
        # Each camera must actually see the pendulum (non-uniform frame).
        assert frame.std() > 1.0, f"{cam} frame looks empty (std={frame.std():.2f})"
    # The two cameras are different viewpoints, so their frames must differ.
    diff = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
    assert diff > 1.0, f"cam0 and cam1 frames are nearly identical (diff={diff:.2f})"
    print(f"render ok | {len(frames)} frames, each {frames[0].shape} {frames[0].dtype} | cam diff={diff:.1f}")

    for i in range(50):
        action = env.sample_tau()
        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(obs).all():
            raise RuntimeError(f"non-finite obs at step {i}: {obs}")

        if terminated or truncated:
            print("episode ended early at step", i, "terminated:", terminated, "truncated:", truncated)
            break

    env.close()
    print("rgb_array close ok\n")


def test_rgb_state():
    """Verify rgb_state=True embeds one RGB frame per camera in the info dict.

    The frames must appear under "rgb_<camera>" keys from both reset() and step(),
    render at the configured size, and actually show the pendulum (non-empty). This
    works with render_mode=None, proving the RGB path is independent of the viewer.
    """
    print("=== RGB STATE TEST ===")
    env = DoublePendulum(
        mass0=1.0, mass1=1.0, length0=0.2, length1=0.2,
        timestep=0.001, rgb_state=True,
        render_width=320, render_height=240,
    )

    obs, info = env.reset(seed=0)
    for cam in env.CAMERA_NAMES:
        key = f"rgb_{cam}"
        assert key in info, f"missing {key} | info keys: {list(info.keys())}"
        frame = info[key]
        assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8, (key, frame.shape, frame.dtype)
        assert frame.std() > 1.0, f"{key} frame looks empty (std={frame.std():.2f})"

    # The frames must also be present after a step.
    _, _, _, _, info = env.step(np.zeros(6))
    for cam in env.CAMERA_NAMES:
        assert f"rgb_{cam}" in info, f"missing rgb_{cam} after step"

    env.close()
    print(f"rgb_state ok | keys: {[f'rgb_{c}' for c in env.CAMERA_NAMES]}\n")


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
        rgb_state=True,
    )

    n_episodes = 3
    episode_duration = 2.0  # seconds of wall-clock time per episode
    zero_action = np.zeros(6)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        floor_id     = env.model.geom("floor").id
        ground_depth = env.model.geom_pos[floor_id][2]
        ground_quat  = env.model.geom_quat[floor_id]
        tilt_deg     = float(np.degrees(2 * np.arccos(np.clip(ground_quat[0], -1, 1))))
        print(f"episode {ep+1}/{n_episodes} | ground depth={ground_depth:.3f} m  tilt={tilt_deg:.1f}°")

        cams = env.CAMERA_NAMES
        fig, axes = plt.subplots(1, len(cams), figsize=(4 * len(cams), 4), squeeze=False)
        fig.suptitle(f"episode {ep+1}/{n_episodes} — rgb_state frames")
        for ax, cam in zip(axes[0], cams):
            frame = info[f"rgb_{cam}"]
            ax.imshow(frame)
            ax.set_title(f"rgb_{cam} {frame.shape}")
            ax.axis("off")
        plt.tight_layout()
        out_path = f"rgb_state_ep{ep+1}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  saved rgb_state frames -> {out_path}")

        t_end = time.time() + episode_duration
        steps = 0
        while time.time() < t_end:
            obs, reward, terminated, truncated, info = env.step(zero_action)
            steps += 1
            if terminated or truncated:
                print(f"  simulation blew up at step {steps}, restarting episode")
                obs, info = env.reset(seed=ep * 100 + steps)
                break

        print(f"  done | steps: {steps}")

    env.close()
    print("human close ok\n")


if __name__ == "__main__":
    test_spaces_and_dims()
    test_3d_motion()
    test_spawn_and_tau_sampling()
    smoke_rgb_array()
    test_rgb_state()
    smoke_human()
    print("ALL SMOKE TESTS PASSED ✅")
