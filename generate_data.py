"""Collect double-pendulum rollout trajectories and write them to HDF5 shards.

Layout: data/shard_NNN.h5, one HDF5 group per trajectory ("traj_NNNN").

Usage:
    # single shard (for parallel execution across processes):
    python generate_data.py --shard 0

    # all 30 shards sequentially:
    python generate_data.py
"""

import argparse
import os

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from env import DoublePendulum
from utils import downsample_rgb

N_SHARDS   = 30
N_TRAJ     = 1000
N_STEPS    = 100
TARGET_H   = 64
TARGET_W   = 64
STATE_DIM  = 212  # 3+6+6+6+3+3+6 + 2*8*10 + 2*8 + 3  (K=8 segments)
OBS_DIM    = 14
ACTION_DIM = 6
OUT_DIR    = "data"


def _make_env(slide: bool=True, rgb_state: bool=True) -> DoublePendulum:
    # timestep defaults to 1/60 → mujoco model dt = (1/60)/4 = 1/240 s (240 Hz)
    # rgb_state=False skips the per-step camera renders (used by eval.py when
    # evaluating the state modality, which never looks at pixels).
    return DoublePendulum(
        mass0=1.0,
        mass1=1.0,
        length0=0.3,
        length1=0.4,
        rgb_state=rgb_state,
        render_width=256,
        render_height=256,
        slide=slide
    )


def collect_shard(shard_idx: int, out_dir: str, slide: bool=True) -> None:
    env = _make_env(slide)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{shard_idx:03d}.h5")

    lz4 = hdf5plugin.LZ4()
    gz  = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

    with h5py.File(path, "w") as f:
        for traj_idx in tqdm(range(N_TRAJ), desc=f"shard {shard_idx:03d}", unit="traj"):
            # Pre-allocate; rows beyond a terminated step stay zero.
            buf_state     = np.zeros((N_STEPS, STATE_DIM),          dtype=np.float64)
            buf_obs       = np.zeros((N_STEPS, OBS_DIM),            dtype=np.float64)
            buf_actions   = np.zeros((N_STEPS, ACTION_DIM),         dtype=np.float64)
            buf_R         = np.zeros((N_STEPS, 3, 3),               dtype=np.float64)
            buf_world_pos = np.zeros((N_STEPS, 3),                  dtype=np.float64)
            buf_cam0      = np.zeros((N_STEPS, TARGET_H, TARGET_W, 3), dtype=np.uint8)
            buf_cam1      = np.zeros((N_STEPS, TARGET_H, TARGET_W, 3), dtype=np.uint8)

            # Unique, deterministic seed per trajectory so shards are reproducible
            # and never overlap. sample_tau() draws from the same seeded RNG.
            env.reset(seed=shard_idx * N_TRAJ + traj_idx)

            # Record the episode's ground pose — it isn't part of the state vector
            # but is needed to recompute contacts (e.g. for rollout evaluation).
            floor_id = env.model.geom("floor").id
            ground_pos  = env.model.geom_pos[floor_id].copy()
            ground_quat = env.model.geom_quat[floor_id].copy()

            steps_taken = 0
            for step in range(N_STEPS):
                # Exponentially smoothed torque (see DoublePendulum.sample_tau),
                # not white-noise uniform sampling.
                action = env.sample_tau()
                obs, _reward, terminated, _truncated, info = env.step(action)

                if terminated:
                    # The sim blew up during this step, so the state is non-finite —
                    # don't write it. Rows from here on stay zero-padded; loaders
                    # recover the valid length from the obs quaternion norm.
                    break

                buf_state[step]     = info["state"]
                buf_obs[step]       = obs
                buf_actions[step]   = action
                buf_R[step]         = info["R"]
                buf_world_pos[step] = info["world_pos"]
                buf_cam0[step]      = downsample_rgb(info["rgb_cam0"], TARGET_H, TARGET_W)
                buf_cam1[step]      = downsample_rgb(info["rgb_cam1"], TARGET_H, TARGET_W)
                steps_taken += 1

            grp = f.create_group(f"traj_{traj_idx:04d}")
            grp.attrs["steps_taken"] = steps_taken
            grp.attrs["ground_pos"]  = ground_pos
            grp.attrs["ground_quat"] = ground_quat
            grp.create_dataset("state",     data=buf_state,     **gz)
            grp.create_dataset("obs",       data=buf_obs,       **gz)
            grp.create_dataset("actions",   data=buf_actions,   **gz)
            grp.create_dataset("R",         data=buf_R,         **gz)
            grp.create_dataset("world_pos", data=buf_world_pos, **gz)
            grp.create_dataset("rgb_cam0",  data=buf_cam0,      **lz4)
            grp.create_dataset("rgb_cam1",  data=buf_cam1,      **lz4)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect double-pendulum NeRD dataset shards")
    parser.add_argument(
        "--shard", type=int, default=None,
        help=f"Shard index in [0, {N_SHARDS - 1}]. Omit to run all shards sequentially.",
    )
    parser.add_argument("--no-slide", dest="slide", action="store_false", default=True)
    parser.add_argument(
        "--out_dir", type=str, default=OUT_DIR,
        help="Output directory for HDF5 shard files.",
    )
    args = parser.parse_args()
    
    if args.shard is not None:
        if not (0 <= args.shard < N_SHARDS):
            raise ValueError(f"--shard must be in [0, {N_SHARDS - 1}], got {args.shard}")
        collect_shard(args.shard, args.out_dir, slide=args.slide)
    else:
        for i in range(N_SHARDS):
            collect_shard(i, args.out_dir, slide=args.slide)


if __name__ == "__main__":
    main()
