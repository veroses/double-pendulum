"""Evaluate a trained double-pendulum dynamics checkpoint.

Runs locally on CPU / Apple MPS with nothing but the checkpoint — eval episodes
are generated live by the simulator, so no dataset shards are required.

Two evaluation modes:

1. Live episodes (default). Fresh episodes are rolled out with the same env and
   action sampling used for training data (seeded far outside the training
   range), then the model is scored two ways:

     * single-step (teacher-forced) — the exact quantity the training loss
       optimises, reported as normalised MSE plus raw rotation/angvel splits.
     * open-loop autoregressive rollout — NeRD-style: the network predicts Δs,
       the predicted state is re-orthonormalised (rotation_from_6d), and the
       *non-predicted* inputs (contact bins, gravity direction) are recomputed
       at the predicted state via the simulator's collision detection — the
       dynamics stay fully neural. For the vision modality the cameras are
       re-rendered at the predicted state, closing the pixels-in loop.
       Reported as error-vs-horizon curves (joint rotation geodesic error,
       angular-velocity error, pendulum tip position error) against a
       hold-last-state persistence baseline.

   Use --no-slide for the held-out no-ground generalisation setting.

2. Stored shards (--use-shards). Single-step errors on --data-dir/shard_*.h5.
   By default this reconstructs the *validation split* with the same seed /
   val-frac the checkpoint was trained with (for pointing at the training
   shards). Add --holdout when --data-dir is a dedicated eval set (generated
   with its own --seed-offset) to score every trajectory in the directory.

Both checkpoint schemas are supported: train.py's ckpt_best.pt / ckpt_last.pt
and the train.ipynb Trainer's best_model.pt / final_model.pt / model_epochN.pt.

Usage:
    python eval.py --ckpt runs/state_run/ckpt_best.pt
    python eval.py --ckpt runs/state_run/ckpt_best.pt --episodes 50 --no-slide
    python eval.py --ckpt runs/vision_run/best_model.pt --episodes 10
    python eval.py --ckpt runs/state_run/ckpt_best.pt --use-shards --data-dir data

Outputs (under --out-dir, default eval_out/<ckpt parent dir name>/):
    metrics.json        — all summary numbers
    rollout_curves.npz  — full per-horizon error arrays (mean/std over episodes)
    rollout_curves.png  — error-vs-horizon plots, model vs persistence baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader

from generate_data import _make_env, N_STEPS
from utils import downsample_rgb, rotation_from_6d, rotation_to_6d

# train.py puts models/ on sys.path and defines the shared data/model plumbing.
from train import (
    IMG,
    PRED,
    RunningMeanStd,
    WindowDataset,
    build_index,
    build_model,
    build_state_input,
    compute_losses,
    evaluate as shard_evaluate,
    pick_device,
    predictable,
)
from dynamics import FRAMES_KEY, STATE_KEY, TORQUE_KEY  # noqa: E402

# Pendulum geometry — must match generate_data._make_env (used for tip error).
LINK_LENGTHS = (0.3, 0.4)


# --------------------------------------------------------------------------- #
# Checkpoint loading (handles both train.py and train.ipynb schemas)
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str, device: str):
    """Rebuild the model + normalisers from a checkpoint file.

    Returns (model, modality, history, output_rms, meta) with the model on
    `device`, in eval mode, and with its normalisers registered.
    """
    ck = torch.load(path, map_location=device, weights_only=False)

    if "args" in ck:  # train.py: {model, optimizer, input_rms, output_rms, step, best_val, modality, args}
        a = ck["args"]
        modality = ck["modality"]
        history = int(a["history"])
        hp = SimpleNamespace(
            n_layer=a["n_layer"], n_head=a["n_head"], n_embd=a["n_embd"],
            dropout=a["dropout"], head_hidden=list(a["head_hidden"]),
        )
        meta = {"schema": "train.py", "step": ck.get("step"),
                "best_val": ck.get("best_val"),
                "seed": a.get("seed", 0), "val_frac": a.get("val_frac", 0.05)}
    elif "config" in ck:  # train.ipynb Trainer.save_model
        c = ck["config"]
        modality = c["modality"]
        history = int(c["sample_sequence_length"])
        hp = SimpleNamespace(
            n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],
            dropout=c["dropout"], head_hidden=list(c["head_hidden"]),
        )
        meta = {"schema": "train.ipynb", "best_val": ck.get("best_val"),
                "seed": c.get("seed", 0), "val_frac": c.get("val_frac", 0.05)}
    else:
        raise ValueError(
            f"{path}: unrecognised checkpoint — expected an 'args' key (train.py) "
            f"or a 'config' key (train.ipynb), got {sorted(ck.keys())}"
        )

    model = build_model(modality, history, hp, device)
    model.load_state_dict(ck["model"])

    input_rms = {k: RunningMeanStd.from_dict(v) for k, v in ck["input_rms"].items()}
    output_rms = RunningMeanStd.from_dict(ck["output_rms"])
    model.set_input_rms(input_rms)
    model.set_output_rms(output_rms)
    model.eval()
    return model, modality, history, output_rms, meta


# --------------------------------------------------------------------------- #
# Live episode collection
# --------------------------------------------------------------------------- #
def collect_episode(env, seed: int, n_steps: int, want_frames: bool) -> dict:
    """Roll out one full episode exactly like generate_data.collect_shard.

    Retries with a perturbed seed on the (now rare) simulation blow-up so every
    returned episode has the full n_steps transitions.
    """
    for attempt in range(20):
        env.reset(seed=seed + attempt * 1_000_003)
        floor_id = env.model.geom("floor").id
        ground_pos = env.model.geom_pos[floor_id].copy()
        ground_quat = env.model.geom_quat[floor_id].copy()

        states, actions, frames = [], [], []
        ok = True
        for _ in range(n_steps):
            action = env.sample_tau()
            _obs, _r, terminated, _tr, info = env.step(action)
            if terminated:
                ok = False
                break
            states.append(info["state"])
            actions.append(action)
            if want_frames:
                f0 = downsample_rgb(info["rgb_cam0"], IMG, IMG)
                f1 = downsample_rgb(info["rgb_cam1"], IMG, IMG)
                frames.append(np.stack([f0, f1]))  # [2, 64, 64, 3] uint8
        if ok:
            ep = {
                "states": np.asarray(states),    # [N, 212]
                "actions": np.asarray(actions),  # [N, 6]
                "ground_pos": ground_pos,
                "ground_quat": ground_quat,
            }
            if want_frames:
                ep["frames"] = np.asarray(frames)  # [N, 2, 64, 64, 3] uint8
            return ep
    raise RuntimeError(f"episode with seed {seed} blew up 20 times in a row")


def frames_to_tensor(frames_u8: np.ndarray) -> torch.Tensor:
    """[.., H, W, 3] uint8 -> [.., 3, H, W] float in [-1, 1] (training scaling)."""
    x = torch.from_numpy(frames_u8.astype(np.float32) / 127.5 - 1.0)
    return x.movedim(-1, -3)


# --------------------------------------------------------------------------- #
# Shadow simulator — recomputes the non-predicted inputs at a predicted state
# --------------------------------------------------------------------------- #
class ShadowSim:
    """Kinematic twin of the env, used during autoregressive rollout.

    The neural model only predicts [R_j0, R_j1, w0, w1]; the remaining token
    inputs (contact bins, gravity-in-body-frame — and pixels, for the vision
    model) are functions of that state and the known world geometry. This class
    holds the episode's ground pose and, given a predicted state, sets qpos/qvel,
    runs mj_forward (collision detection + kinematics only — no dynamics step),
    and reads back the full 212-dim state vector / rendered cameras.
    """

    def __init__(self, slide: bool, want_frames: bool):
        self.env = _make_env(slide=slide, rgb_state=want_frames)
        self.env.reset(seed=0)  # builds the MuJoCo model
        self.want_frames = want_frames
        self._floor_id = self.env.model.geom("floor").id

    def set_ground(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.env.model.geom_pos[self._floor_id] = pos
        self.env.model.geom_quat[self._floor_id] = quat

    def info_at(self, R0: np.ndarray, R1: np.ndarray, qvel: np.ndarray) -> dict:
        q = np.zeros(8)
        mujoco.mju_mat2Quat(q[0:4], np.ascontiguousarray(R0.reshape(9)))
        mujoco.mju_mat2Quat(q[4:8], np.ascontiguousarray(R1.reshape(9)))
        self.env.data.qpos[:] = q
        self.env.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env._get_info()

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------- #
# Error metrics on the 18-dim predictable state
# --------------------------------------------------------------------------- #
def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def tip_position(R0: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """World position of the pendulum tip. link1's world rotation is R0 @ R1."""
    l0, l1 = LINK_LENGTHS
    down = np.array([0.0, 0.0, -1.0])
    return R0 @ (l0 * down) + (R0 @ R1) @ (l1 * down)


def state_errors(est: np.ndarray, true: np.ndarray) -> dict:
    """Interpretable errors between two 18-dim predictable states."""
    R0e, R1e = rotation_from_6d(est[0:6]), rotation_from_6d(est[6:12])
    R0t, R1t = rotation_from_6d(true[0:6]), rotation_from_6d(true[6:12])
    return {
        "rot0_deg": geodesic_deg(R0e, R0t),
        "rot1_deg": geodesic_deg(R1e, R1t),
        "angvel": float(np.linalg.norm(est[12:18] - true[12:18])),
        "tip_m": float(np.linalg.norm(tip_position(R0e, R1e) - tip_position(R0t, R1t))),
    }


ERR_KEYS = ("rot0_deg", "rot1_deg", "angvel", "tip_m")


# --------------------------------------------------------------------------- #
# Single-step (teacher-forced) evaluation on live episodes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def single_step_eval(model, episodes, history, modality, device, out_std,
                     chunk: int = 16) -> dict:
    """Score the model exactly like the training loss: dense Δs prediction over
    length-`history` windows (stride = history, remainder dropped)."""
    tot = {"norm_mse": 0.0, "raw_mse": 0.0, "rot_mse": 0.0, "angvel_mse": 0.0}
    n_chunks = 0

    for ep in episodes:
        states, actions = ep["states"], ep["actions"]
        cur, nxt = states[:-1], states[1:]
        drive = actions[1:]
        target = (predictable(nxt) - predictable(cur)).astype(np.float32)

        n_w = len(cur) // history
        if n_w == 0:
            continue
        w = lambda arr: arr[: n_w * history].reshape(n_w, history, *arr.shape[1:])
        tgt_w = torch.from_numpy(w(target))

        if modality == "state":
            x = build_state_input(cur, drive).astype(np.float32)
            inputs = [({STATE_KEY: torch.from_numpy(w(x)[i:i + chunk]).to(device)},
                       tgt_w[i:i + chunk]) for i in range(0, n_w, chunk)]
        else:
            fr = frames_to_tensor(w(ep["frames"][:-1]))          # [n_w, T, 2, 3, H, W]
            tq = torch.from_numpy(w(drive.astype(np.float32)))
            inputs = [({FRAMES_KEY: fr[i:i + chunk].to(device),
                        TORQUE_KEY: tq[i:i + chunk].to(device)},
                       tgt_w[i:i + chunk]) for i in range(0, n_w, chunk)]

        for input_dict, tgt in inputs:
            pred = model(input_dict)
            loss, diag = compute_losses(pred, tgt.to(device), out_std)
            tot["norm_mse"] += loss.item()
            for k in ("raw_mse", "rot_mse", "angvel_mse"):
                tot[k] += diag[k]
            n_chunks += 1

    return {k: v / max(n_chunks, 1) for k, v in tot.items()}


# --------------------------------------------------------------------------- #
# Open-loop autoregressive rollout
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_episode(model, ep, history, modality, shadow, device):
    """Autoregressive rollout over one episode.

    The first `history` transitions are ground truth (context seeding); from
    there every state is model-predicted, with contacts/gravity (and frames,
    for vision) recomputed at the predicted state by the shadow simulator.

    Returns (errs, base): dicts of per-horizon error arrays [N - history] for
    the model and for the hold-last-seeded-state persistence baseline.
    """
    states, actions = ep["states"], ep["actions"]
    N, T = len(states), history
    shadow.set_ground(ep["ground_pos"], ep["ground_quat"])

    if modality == "state":
        x = build_state_input(states[:T], actions[1:T + 1]).astype(np.float32)
        tokens = list(x)                                   # each [203]
    else:
        frame_tokens = list(frames_to_tensor(ep["frames"][:T]))   # each [2, 3, H, W]
        torque_tokens = list(actions[1:T + 1].astype(np.float32))

    truth = predictable(states)          # [N, 18]
    s_est = truth[T - 1].copy()
    s_frozen = truth[T - 1]              # persistence baseline

    errs = {k: [] for k in ERR_KEYS}
    base = {k: [] for k in ERR_KEYS}

    for t in range(T - 1, N - 1):
        if modality == "state":
            inp = {STATE_KEY: torch.from_numpy(
                np.stack(tokens[-T:], 0)[None]).to(device)}
        else:
            inp = {
                FRAMES_KEY: torch.stack(frame_tokens[-T:], 0)[None].to(device),
                TORQUE_KEY: torch.from_numpy(
                    np.stack(torque_tokens[-T:], 0)[None]).to(device),
            }
        delta = model.evaluate(inp)[0, 0].float().cpu().numpy()

        s_est = s_est + delta
        # Project the two 6D rotations back onto SO(3) so error can't compound
        # into invalid rotations.
        R0 = rotation_from_6d(s_est[0:6])
        R1 = rotation_from_6d(s_est[6:12])
        s_est[0:6] = rotation_to_6d(R0)
        s_est[6:12] = rotation_to_6d(R1)

        for k, v in state_errors(s_est, truth[t + 1]).items():
            errs[k].append(v)
        for k, v in state_errors(s_frozen, truth[t + 1]).items():
            base[k].append(v)

        # Assemble the next input token at the predicted state.
        if t + 2 <= N - 1:
            info = shadow.info_at(R0, R1, s_est[12:18])
            if modality == "state":
                tokens.append(build_state_input(
                    info["state"][None], actions[t + 2][None])[0].astype(np.float32))
            else:
                f0 = downsample_rgb(info["rgb_cam0"], IMG, IMG)
                f1 = downsample_rgb(info["rgb_cam1"], IMG, IMG)
                frame_tokens.append(frames_to_tensor(np.stack([f0, f1])))
                torque_tokens.append(actions[t + 2].astype(np.float32))

    return ({k: np.asarray(v) for k, v in errs.items()},
            {k: np.asarray(v) for k, v in base.items()})


def summarise_rollout(all_errs: list[dict], all_base: list[dict]) -> dict:
    """Stack per-episode curves and produce mean/std per horizon + key horizons."""
    out = {"horizon_steps": len(all_errs[0][ERR_KEYS[0]])}
    curves = {}
    for k in ERR_KEYS:
        m = np.stack([e[k] for e in all_errs])   # [episodes, horizon]
        b = np.stack([e[k] for e in all_base])
        curves[k] = {"model_mean": m.mean(0), "model_std": m.std(0),
                     "base_mean": b.mean(0), "base_std": b.std(0)}
    out["curves"] = curves

    key_h = [h for h in (1, 5, 10, 25, 50, out["horizon_steps"]) if h <= out["horizon_steps"]]
    out["at_horizon"] = {
        h: {k: {"model": float(curves[k]["model_mean"][h - 1]),
                "baseline": float(curves[k]["base_mean"][h - 1])} for k in ERR_KEYS}
        for h in key_h
    }
    return out


def plot_rollout(summary: dict, path: str) -> None:
    curves = summary["curves"]
    h = np.arange(1, summary["horizon_steps"] + 1)
    panels = [
        ("rot0_deg", "joint0 rotation error (deg)"),
        ("rot1_deg", "joint1 rotation error (deg)"),
        ("angvel", "angular velocity error (rad/s)"),
        ("tip_m", "tip position error (m)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (k, title) in zip(axes, panels):
        c = curves[k]
        ax.plot(h, c["model_mean"], label="model", color="C0")
        ax.fill_between(h, c["model_mean"] - c["model_std"],
                        c["model_mean"] + c["model_std"], alpha=0.2, color="C0")
        ax.plot(h, c["base_mean"], label="hold-state baseline", ls="--", color="C1")
        ax.set_title(title)
        ax.set_xlabel("rollout horizon (steps)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Open-loop rollout error vs horizon (mean ± std over episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Stored-shard validation-split evaluation
# --------------------------------------------------------------------------- #
def eval_on_shards(model, modality, history, out_std, meta, args, device) -> dict:
    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.h5")))
    if not shard_paths:
        raise SystemExit(f"--use-shards: no shard_*.h5 in {args.data_dir}/")
    if args.holdout:
        # Dedicated eval set: every trajectory is held out, score all of it.
        print(f"scoring all trajectories in {args.data_dir} "
              f"({len(shard_paths)} shards) ...")
        _train_w, val_w = build_index(shard_paths, history, val_frac=1.0, seed=0)
    else:
        seed, val_frac = meta["seed"], meta["val_frac"]
        print(f"rebuilding val split (seed={seed}, val_frac={val_frac}) over "
              f"{len(shard_paths)} shards ...")
        _train_w, val_w = build_index(shard_paths, history, val_frac, seed)
    if len(val_w) == 0:
        raise SystemExit("validation split is empty — check --data-dir / val_frac")
    ds = WindowDataset(shard_paths, val_w, history, modality)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    metrics = shard_evaluate(model, loader, modality, device, out_std,
                             max_batches=args.max_batches)
    model.eval()  # shard_evaluate leaves the model in train mode
    return {"n_val_windows": int(len(val_w)),
            "batches_scored": min(args.max_batches, int(np.ceil(len(val_w) / args.batch_size))),
            **metrics}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to ckpt_best.pt / best_model.pt etc.")
    p.add_argument("--episodes", type=int, default=20, help="Live eval episodes.")
    p.add_argument("--steps", type=int, default=N_STEPS, help="Steps per live episode.")
    p.add_argument("--no-slide", dest="slide", action="store_false", default=True,
                   help="Park the ground out of reach (held-out generalisation setting).")
    p.add_argument("--seed", type=int, default=5_000_000,
                   help="Base episode seed — keep outside the training range.")
    p.add_argument("--use-shards", action="store_true",
                   help="Also score stored shards in --data-dir (single-step).")
    p.add_argument("--holdout", action="store_true",
                   help="With --use-shards: --data-dir is a dedicated eval set — "
                        "score every trajectory instead of the training val split.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=200, help="Cap for --use-shards.")
    p.add_argument("--out-dir", default=None,
                   help="Default: eval_out/<checkpoint's parent dir name>/")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, modality, history, output_rms, meta = load_checkpoint(args.ckpt, device)
    out_std = output_rms.std.to(device)
    n_params = sum(pm.numel() for pm in model.parameters())
    print(f"ckpt={args.ckpt} ({meta['schema']}) | modality={modality} | "
          f"history={history} | params={n_params:,} | device={device}")

    out_dir = args.out_dir or os.path.join(
        "eval_out", os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))))
    os.makedirs(out_dir, exist_ok=True)

    results = {"ckpt": os.path.abspath(args.ckpt), "modality": modality,
               "history": history, "meta": meta, "device": device,
               "episodes": args.episodes, "steps": args.steps,
               "slide": args.slide, "seed": args.seed}

    # --- live episodes -------------------------------------------------------
    if args.episodes > 0:
        if args.steps < history + 2:
            raise SystemExit(f"--steps must be at least history+2 = {history + 2}")
        want_frames = modality == "vision"
        print(f"collecting {args.episodes} live episodes "
              f"(slide={'on' if args.slide else 'off'}) ...")
        env = _make_env(slide=args.slide, rgb_state=want_frames)
        episodes = [collect_episode(env, args.seed + i, args.steps, want_frames)
                    for i in range(args.episodes)]
        env.close()

        print("single-step (teacher-forced) eval ...")
        ss = single_step_eval(model, episodes, history, modality, device, out_std)
        results["single_step"] = ss
        print("  " + " | ".join(f"{k}={v:.5f}" for k, v in ss.items()))

        print("open-loop rollout eval ...")
        shadow = ShadowSim(slide=args.slide, want_frames=want_frames)
        all_errs, all_base = [], []
        for i, ep in enumerate(episodes):
            e, b = rollout_episode(model, ep, history, modality, shadow, device)
            all_errs.append(e)
            all_base.append(b)
            print(f"  episode {i + 1}/{len(episodes)} | final-step: "
                  + " ".join(f"{k}={e[k][-1]:.3f}" for k in ERR_KEYS))
        shadow.close()

        summary = summarise_rollout(all_errs, all_base)
        results["rollout"] = {"horizon_steps": summary["horizon_steps"],
                              "at_horizon": summary["at_horizon"]}

        np.savez(os.path.join(out_dir, "rollout_curves.npz"),
                 **{f"{k}_{stat}": v
                    for k, c in summary["curves"].items() for stat, v in c.items()})
        plot_rollout(summary, os.path.join(out_dir, "rollout_curves.png"))

        print("\nrollout error at key horizons (model / hold-state baseline):")
        for h, row in summary["at_horizon"].items():
            cells = "  ".join(f"{k}: {row[k]['model']:.3f}/{row[k]['baseline']:.3f}"
                              for k in ERR_KEYS)
            print(f"  h={h:>3} | {cells}")

    # --- stored shards --------------------------------------------------------
    if args.use_shards:
        results["shard_val"] = eval_on_shards(
            model, modality, history, out_std, meta, args, device)
        print("shard val split: " + " | ".join(
            f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in results["shard_val"].items()))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out_dir}/metrics.json"
          + (f", rollout_curves.png, rollout_curves.npz" if args.episodes > 0 else ""))


if __name__ == "__main__":
    main()
