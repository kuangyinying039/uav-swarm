"""Record a dual-capture chase trajectory with kinematic Radar5 (no Isaac PhysX).

Saves pursuer 13-D states + target xyz each step so Isaac can pose-replay later
without running live Lee/PhysX control (avoids gap=25 / flash-quit blow-ups).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("apf", "frpn", "mpc"), default="mpc")
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_medium.json"),
    )
    parser.add_argument("--seed", type=int, default=7_000_008)
    parser.add_argument("--scan", type=int, default=40, help="Try seed..seed+scan-1 for a clean chase")
    parser.add_argument("--evader-policy", choices=("occlusion", "repulsive"), default="occlusion")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "pursuit_traj.npz",
    )
    return parser


def _score_episode(env: QuadrotorPursuitEnv, action_fn, seed: int) -> dict:
    env.cfg = replace(env.cfg, seed=int(seed))
    env.rng = np.random.default_rng(int(seed))
    observation = env.reset()
    frames_pursuer = []
    frames_target = []
    prev = None
    path = 0.0
    moved = 0
    multi = 0
    close_stop = 0
    max_close_stop = 0
    edge_streak = 0
    max_edge = 0
    grid = float(env.cfg.grid_size)
    captured = False
    steps = 0
    for step in range(env.cfg.search_steps):
        action = np.asarray(action_fn(observation), dtype=np.float32)
        result = env.step_joint(action)
        observation = result["obs"]
        steps = step + 1
        pursuer = np.asarray(env.quadrotor_states, dtype=np.float64)
        target = np.array([*env.dynamic_targets[0], env.target_altitudes[0]], dtype=np.float64)
        frames_pursuer.append(pursuer.copy())
        frames_target.append(target.copy())
        speed = 0.0
        if prev is not None:
            d = float(np.linalg.norm(target - prev))
            path += d
            speed = d / max(float(env.cfg.decision_dt), 1e-6)
            if speed > 0.05:
                moved += 1
        prev = target.copy()
        on_edge = (
            target[0] <= 2.5
            or target[1] <= 2.5
            or target[0] >= grid - 2.5
            or target[1] >= grid - 2.5
        )
        edge_streak = edge_streak + 1 if on_edge else 0
        max_edge = max(max_edge, edge_streak)
        active = ~np.asarray(env.disabled_uavs, dtype=bool)
        dists = np.linalg.norm(pursuer[active, :3] - target[None, :], axis=1)
        nearest = float(np.min(dists)) if len(dists) else 99.0
        if np.count_nonzero(dists < 6.0) >= 2:
            multi += 1
        if nearest < 5.0:
            close_stop = close_stop + 1 if speed < 0.08 else 0
            max_close_stop = max(max_close_stop, close_stop)
        else:
            close_stop = 0
        if result.get("terminated") or result.get("truncated"):
            captured = bool(result.get("capture_success", False))
            break
    moved_ratio = moved / max(steps, 1)
    bug = (
        max_close_stop >= 35
        or max_edge >= 60
        or (captured and moved_ratio < 0.3)
        or (captured and steps < 35)
    )
    score = (
        (160.0 if captured and not bug else 0.0)
        + min(path, 100.0)
        + 0.5 * moved
        + 0.6 * multi
        - max(0, max_close_stop - 10) * 2.5
        - max(0, max_edge - 12) * 2.0
        - (120.0 if bug else 0.0)
    )
    return {
        "seed": int(seed),
        "captured": captured,
        "bug": bug,
        "steps": steps,
        "path": round(path, 2),
        "moved_ratio": round(moved_ratio, 3),
        "multi": multi,
        "max_close_stop": max_close_stop,
        "max_edge": max_edge,
        "score": round(score, 2),
        "pursuer": np.stack(frames_pursuer, axis=0),
        "target": np.stack(frames_target, axis=0),
        "buildings": np.asarray(env.buildings, dtype=np.float64),
        "building_heights": np.asarray(env.building_heights, dtype=np.float64),
    }


def main():
    args = build_parser().parse_args()
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    cfg = replace(cfg, evader_policy=str(args.evader_policy), seed=int(args.seed))
    env = QuadrotorPursuitEnv(cfg)
    policy = BASELINES_3D[args.method]()
    action_fn = lambda _obs: policy.actions(env)

    best = None
    for offset in range(max(1, int(args.scan))):
        seed = int(args.seed) + offset
        row = _score_episode(env, action_fn, seed)
        tag = "BUG" if row["bug"] else ("CAP" if row["captured"] else "miss")
        print(
            f"[record] seed={seed} {tag} steps={row['steps']} path={row['path']} "
            f"mv={row['moved_ratio']} multi={row['multi']} cstop={row['max_close_stop']} "
            f"edge={row['max_edge']} score={row['score']}",
            flush=True,
        )
        if best is None or row["score"] > best["score"]:
            best = row
        if row["captured"] and not row["bug"] and row["score"] >= 200:
            break

    assert best is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": args.method,
        "evader_policy": args.evader_policy,
        "env_config": str(args.env_config),
        "env": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()
                if k in ("n_uavs", "grid_size", "decision_dt", "capture_radius",
                         "search_steps", "building_count", "target_speed", "evader_policy")},
        "seed": best["seed"],
        "captured": best["captured"],
        "bug": best["bug"],
        "steps": best["steps"],
        "path": best["path"],
        "moved_ratio": best["moved_ratio"],
        "multi": best["multi"],
        "score": best["score"],
    }
    np.savez_compressed(
        args.out,
        pursuer=best["pursuer"],
        target=best["target"],
        buildings=best["buildings"],
        building_heights=best["building_heights"],
        meta=json.dumps(meta),
    )
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[record] wrote {args.out} seed={best['seed']} steps={best['steps']} "
          f"cap={best['captured']} bug={best['bug']} score={best['score']}", flush=True)


if __name__ == "__main__":
    main()
