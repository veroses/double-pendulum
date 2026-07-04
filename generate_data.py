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


def _make_env(slide: bool=True) -> DoublePendulum:
    # timestep defaults to 1/60 → mujoco model dt = (1/60)/4 = 1/240 s (240 Hz)
    return DoublePendulum(
        mass0=1.0,
        mass1=1.0,
        length0=0.3,
        length1=0.4,
        rgb_state=True,
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

            env.reset()

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

                if terminated:
                    break

            grp = f.create_group(f"traj_{traj_idx:04d}")
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
