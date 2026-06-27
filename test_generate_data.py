"""End-to-end test for the data generation pipeline.

Collects 10 trajectories (100 steps each) into two separate HDF5 files and
asserts that the structure, shapes, dtypes, and values are all correct.

Run:
    python test_generate_data.py
"""

import os

import h5py
import hdf5plugin
import numpy as np

from generate_data import (
    ACTION_DIM,
    N_STEPS,
    OBS_DIM,
    STATE_DIM,
    TARGET_H,
    TARGET_W,
    _make_env,
)
from utils import downsample_rgb

N_TEST_TRAJ = 10

# Expected (shape, dtype) for every dataset in a trajectory group.
EXPECTED = {
    "state":     ((N_STEPS, STATE_DIM),             np.float64),
    "obs":       ((N_STEPS, OBS_DIM),               np.float64),
    "actions":   ((N_STEPS, ACTION_DIM),            np.float64),
    "R":         ((N_STEPS, 3, 3),                  np.float64),
    "world_pos": ((N_STEPS, 3),                     np.float64),
    "rgb_cam0":  ((N_STEPS, TARGET_H, TARGET_W, 3), np.uint8),
    "rgb_cam1":  ((N_STEPS, TARGET_H, TARGET_W, 3), np.uint8),
}


# ── collection ────────────────────────────────────────────────────────────────

def collect(path: str, n_traj: int, seed_offset: int = 0) -> None:
    """Mirror the generate_data pipeline exactly for n_traj trajectories.

    Identical buffers, compression settings, and group layout as collect_shard.
    Adds a 'steps_taken' attribute per group so assertions on padded rows are
    unambiguous.
    """
    env = _make_env()
    lz4 = hdf5plugin.LZ4()
    gz  = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

    with h5py.File(path, "w") as f:
        for traj_idx in range(n_traj):
            buf_state     = np.zeros((N_STEPS, STATE_DIM),             dtype=np.float64)
            buf_obs       = np.zeros((N_STEPS, OBS_DIM),               dtype=np.float64)
            buf_actions   = np.zeros((N_STEPS, ACTION_DIM),            dtype=np.float64)
            buf_R         = np.zeros((N_STEPS, 3, 3),                  dtype=np.float64)
            buf_world_pos = np.zeros((N_STEPS, 3),                     dtype=np.float64)
            buf_cam0      = np.zeros((N_STEPS, TARGET_H, TARGET_W, 3), dtype=np.uint8)
            buf_cam1      = np.zeros((N_STEPS, TARGET_H, TARGET_W, 3), dtype=np.uint8)

            env.reset(seed=seed_offset + traj_idx)

            steps_taken = 0
            for step in range(N_STEPS):
                action = env.action_space.sample()
                obs, _reward, terminated, _truncated, info = env.step(action)

                buf_state[step]     = info["state"]
                buf_obs[step]       = obs
                buf_actions[step]   = action
                buf_R[step]         = info["R"]
                buf_world_pos[step] = info["world_pos"]
                buf_cam0[step]      = downsample_rgb(info["rgb_cam0"], TARGET_H, TARGET_W)
                buf_cam1[step]      = downsample_rgb(info["rgb_cam1"], TARGET_H, TARGET_W)

                steps_taken += 1
                if terminated:
                    break

            grp = f.create_group(f"traj_{traj_idx:04d}")
            grp.attrs["steps_taken"] = steps_taken

            grp.create_dataset("state",     data=buf_state,     **gz)
            grp.create_dataset("obs",       data=buf_obs,       **gz)
            grp.create_dataset("actions",   data=buf_actions,   **gz)
            grp.create_dataset("R",         data=buf_R,         **gz)
            grp.create_dataset("world_pos", data=buf_world_pos, **gz)
            grp.create_dataset("rgb_cam0",  data=buf_cam0,      **lz4)
            grp.create_dataset("rgb_cam1",  data=buf_cam1,      **lz4)

    env.close()


# ── assertions ────────────────────────────────────────────────────────────────

def check_file(path: str, n_traj: int) -> None:
    with h5py.File(path, "r") as f:
        # ── top-level structure ───────────────────────────────────────────────
        assert len(f) == n_traj, \
            f"expected {n_traj} groups, got {len(f)}"

        for traj_idx in range(n_traj):
            key = f"traj_{traj_idx:04d}"
            assert key in f, f"missing group {key}"
            grp = f[key]

            n = int(grp.attrs["steps_taken"])
            assert 1 <= n <= N_STEPS, \
                f"{key}: steps_taken={n} out of range [1, {N_STEPS}]"

            # ── shapes and dtypes ─────────────────────────────────────────────
            for name, (shape, dtype) in EXPECTED.items():
                assert name in grp, f"{key}/{name} missing"
                ds = grp[name]
                assert ds.shape == shape, \
                    f"{key}/{name}: shape {ds.shape} != {shape}"
                assert ds.dtype == dtype, \
                    f"{key}/{name}: dtype {ds.dtype} != {dtype}"

            # ── load arrays ───────────────────────────────────────────────────
            state     = grp["state"][:]
            obs       = grp["obs"][:]
            actions   = grp["actions"][:]
            R         = grp["R"][:]
            world_pos = grp["world_pos"][:]
            cam0      = grp["rgb_cam0"][:]
            cam1      = grp["rgb_cam1"][:]

            # ── filled rows: finite values ────────────────────────────────────
            assert np.isfinite(state[:n]).all(), \
                f"{key}/state has non-finite values in filled rows"
            assert np.isfinite(obs[:n]).all(), \
                f"{key}/obs has non-finite values in filled rows"
            assert np.isfinite(actions[:n]).all(), \
                f"{key}/actions has non-finite values in filled rows"
            assert np.isfinite(world_pos[:n]).all(), \
                f"{key}/world_pos has non-finite values in filled rows"

            # ── padding rows: everything beyond n stays zero ──────────────────
            if n < N_STEPS:
                assert np.all(state[n:] == 0),     f"{key}/state padding not zero"
                assert np.all(obs[n:] == 0),       f"{key}/obs padding not zero"
                assert np.all(actions[n:] == 0),   f"{key}/actions padding not zero"
                assert np.all(R[n:] == 0),         f"{key}/R padding not zero"
                assert np.all(world_pos[n:] == 0), f"{key}/world_pos padding not zero"
                assert np.all(cam0[n:] == 0),      f"{key}/rgb_cam0 padding not zero"
                assert np.all(cam1[n:] == 0),      f"{key}/rgb_cam1 padding not zero"

            # ── obs: quaternion components are unit-norm ──────────────────────
            for step in range(n):
                q0 = obs[step, 0:4]
                q1 = obs[step, 4:8]
                assert abs(np.linalg.norm(q0) - 1.0) < 1e-5, \
                    f"{key} step {step}: joint0 quat not unit-norm"
                assert abs(np.linalg.norm(q1) - 1.0) < 1e-5, \
                    f"{key} step {step}: joint1 quat not unit-norm"

            # ── obs consistent with state (obs is raw qpos+qvel, same data) ──
            # state[9:15] and state[15:21] are 6D rotation reps of the joints,
            # obs[0:8] is the two raw quaternions — they encode the same orientations
            # but different representations, so we only check obs ⊂ state indirectly:
            # state[21:24] = angvel0, state[24:27] = angvel1 = obs[8:11], obs[11:14]
            assert np.allclose(obs[:n, 8:14], state[:n, 21:27], atol=1e-10), \
                f"{key}: angular velocities in obs and state disagree"

            # ── actions within env torque bounds (±max_tau = ±4.0 N·m) ───────
            assert np.all(np.abs(actions[:n]) <= 4.0 + 1e-6), \
                f"{key}/actions out of ±4 N·m bounds: max={np.abs(actions[:n]).max():.4f}"

            # ── R matrices are valid SO(3) ────────────────────────────────────
            for step in range(n):
                r = R[step]
                assert np.allclose(r @ r.T, np.eye(3), atol=1e-5), \
                    f"{key}/R[{step}] not orthonormal"
                assert abs(np.linalg.det(r) - 1.0) < 1e-5, \
                    f"{key}/R[{step}] det != 1"

            # ── RGB: non-trivial, correct range, two cameras differ ───────────
            assert cam0[0].std() > 1.0, \
                f"{key}/rgb_cam0[0] looks empty (std={cam0[0].std():.2f})"
            assert cam1[0].std() > 1.0, \
                f"{key}/rgb_cam1[0] looks empty (std={cam1[0].std():.2f})"

            cam_diff = np.abs(cam0[0].astype(int) - cam1[0].astype(int)).mean()
            assert cam_diff > 0.5, \
                f"{key}: cam0 and cam1 are nearly identical (mean diff={cam_diff:.2f})"


# ── main test ─────────────────────────────────────────────────────────────────

def test_two_files() -> None:
    out_dir = "data/test"
    os.makedirs(out_dir, exist_ok=True)
    path_a = os.path.join(out_dir, "test_a.h5")
    path_b = os.path.join(out_dir, "test_b.h5")

    print(f"Collecting {N_TEST_TRAJ} trajectories → {path_a} ...")
    collect(path_a, N_TEST_TRAJ, seed_offset=0)
    print(f"Collecting {N_TEST_TRAJ} trajectories → {path_b} ...")
    collect(path_b, N_TEST_TRAJ, seed_offset=1000)

    print("Checking file A ...")
    check_file(path_a, N_TEST_TRAJ)
    print(f"  file A ok  ({os.path.getsize(path_a) / 1024**2:.2f} MiB)")

    print("Checking file B ...")
    check_file(path_b, N_TEST_TRAJ)
    print(f"  file B ok  ({os.path.getsize(path_b) / 1024**2:.2f} MiB)")

    # Files must contain different trajectories.
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        state_a = fa["traj_0000/state"][:]
        state_b = fb["traj_0000/state"][:]
    assert not np.allclose(state_a, state_b), \
        "Files A and B have identical traj_0000/state — seeds not advancing"
    print("  files are distinct ✓")


if __name__ == "__main__":
    test_two_files()
    print("\nALL TESTS PASSED ✅")
