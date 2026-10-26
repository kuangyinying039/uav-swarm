"""Record a dual-capture chase with Isaac PhysX (headless), then save poses.

Unlike ``record_pursuit_trajectory.py`` (pure Radar5 kinematics), this steps
``IsaacPursuitEnv`` so pursuer motion comes from Lee + PhysX. Frames with
non-finite PhysX state are skipped / recovered; the saved trajectory is the
physical rollout for later pose-replay / video export.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_isaac_pursuit import load_policy
from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("apf", "frpn", "mpc", "mappo", "matd3"), default="mpc")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_medium.json"),
    )
    parser.add_argument("--seed", type=int, default=7_000_100)
    parser.add_argument("--scan", type=int, default=25, help="Try seed..seed+scan-1")
    parser.add_argument("--evader-policy", choices=("occlusion", "repulsive"), default="occlusion")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "pursuit_traj_physx.npz",
    )
    return parser


def _score_and_record(env, action_fn, seed: int, grid: float) -> dict | None:
    observation = env.reset(seed=int(seed))
    frames_pursuer = []
    frames_target = []
    prev = None
    path = 0.0
    moved = 0
    multi = 0
    max_close_stop = 0
    close_stop = 0
    max_edge = 0
    edge_streak = 0
    physx_ok = 0
    physx_bad = 0
    captured = False
    steps = 0
    for step in range(env.cfg.search_steps):
        action = np.asarray(action_fn(observation), dtype=np.float32)
        result = env.step_joint(action)
        observation = result["obs"]
        steps = step + 1
        bridge = env.bridge.env
        # Prefer finite PhysX state; fall back to bridge only to keep going,
        # but count it as bad so we can reject mostly-kinematic episodes.
        physical = np.asarray(env._physical_state(0), dtype=np.float64)
        if np.isfinite(physical).all():
            pursuer = physical.reshape(env.cfg.n_uavs, 13).copy()
            physx_ok += 1
        else:
            pursuer = np.asarray(bridge.quadrotor_states, dtype=np.float64).copy()
            physx_bad += 1
        target = np.array([*bridge.dynamic_targets[0], bridge.target_altitudes[0]], dtype=np.float64)
        frames_pursuer.append(pursuer)
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

        active = ~np.asarray(bridge.disabled_uavs, dtype=bool)
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
            # Headless truncates on fatal PhysX — treat as failed showcase.
            if result.get("physics_failed") or (
                result.get("truncated") and not captured and steps < env.cfg.search_steps
            ):
                return {
                    "seed": int(seed),
                    "captured": False,
                    "bug": True,
                    "steps": steps,
                    "path": round(path, 2),
                    "moved_ratio": round(moved / max(steps, 1), 3),
                    "multi": multi,
                    "max_close_stop": max_close_stop,
                    "max_edge": max_edge,
                    "physx_ok": physx_ok,
                    "physx_bad": physx_bad,
                    "score": -500.0,
                    "fail": "physics_truncated",
                }
            break

    moved_ratio = moved / max(steps, 1)
    physx_ratio = physx_ok / max(physx_ok + physx_bad, 1)
    bug = (
        max_close_stop >= 15
        or max_edge >= 70
        or (captured and moved_ratio < 0.35)
        or (captured and steps < 45)
        or physx_ratio < 0.85
        or physx_bad > 20
    )
    score = (
        (200.0 if captured and not bug else 0.0)
        + (0.0 if not captured else 0.0)
        + min(path, 100.0)
        + 0.55 * moved
        + 0.65 * multi
        + 80.0 * physx_ratio
        + 40.0 * (1.0 - min(max_close_stop, 40) / 40.0)
        - max(0, max_close_stop - 8) * 4.0
        - max(0, max_edge - 12) * 2.0
        - (150.0 if bug else 0.0)
        - (80.0 if not captured else 0.0)
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
        "physx_ok": physx_ok,
        "physx_bad": physx_bad,
        "physx_ratio": round(physx_ratio, 3),
        "score": round(score, 2),
        "pursuer": np.stack(frames_pursuer, axis=0),
        "target": np.stack(frames_target, axis=0),
        "buildings": np.asarray(bridge.buildings, dtype=np.float64),
        "building_heights": np.asarray(bridge.building_heights, dtype=np.float64),
        "backend": "isaac_physx",
    }


def main():
    args = build_parser().parse_args()
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        raise SystemExit("--checkpoint required for mappo/matd3")
    if args.method in ("apf", "frpn", "mpc") and args.checkpoint is not None:
        raise SystemExit("baseline methods do not take --checkpoint")

    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    cfg = replace(cfg, evader_policy=str(args.evader_policy), seed=int(args.seed))
    print(
        f"[record-physx] method={args.method} evader={cfg.evader_policy} "
        f"seed0={args.seed} scan={args.scan}",
        flush=True,
    )
    env = IsaacPursuitEnv(cfg, seed=int(args.seed), headless=True)
    action_fn, policy = load_policy(args.method, args.checkpoint, env, args.device)

    best = None
    for offset in range(max(1, int(args.scan))):
        seed = int(args.seed) + offset
        try:
            row = _score_and_record(env, action_fn, seed, float(cfg.grid_size))
        except Exception as exc:
            print(f"[record-physx] seed={seed} FAIL {exc}", flush=True)
            continue
        if row is None:
            continue
        tag = "BUG" if row.get("bug") else ("CAP" if row.get("captured") else "miss")
        print(
            f"[record-physx] seed={seed} {tag} steps={row['steps']} path={row['path']} "
            f"mv={row['moved_ratio']} multi={row['multi']} "
            f"physx={row.get('physx_ok', 0)}/{row.get('physx_ok', 0) + row.get('physx_bad', 0)} "
            f"score={row['score']}",
            flush=True,
        )
        if "pursuer" not in row:
            continue
        if row.get("captured") and not row.get("bug") and "pursuer" in row:
            if best is None or row["score"] > best["score"]:
                best = row
        if best is not None and best["score"] >= 250:
            break

    if best is None or "pursuer" not in best:
        raise SystemExit("no usable PhysX capture without freeze-under-pressure")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": args.method,
        "evader_policy": args.evader_policy,
        "env_config": str(args.env_config),
        "backend": "isaac_physx",
        "seed": best["seed"],
        "captured": best["captured"],
        "bug": best["bug"],
        "steps": best["steps"],
        "path": best["path"],
        "moved_ratio": best["moved_ratio"],
        "multi": best["multi"],
        "physx_ok": best.get("physx_ok"),
        "physx_bad": best.get("physx_bad"),
        "physx_ratio": best.get("physx_ratio"),
        "score": best["score"],
        "note": "Poses recorded from Isaac Lee+PhysX; replay is pose display of that rollout.",
    }
    np.savez_compressed(
        args.out,
        pursuer=best["pursuer"],
        target=best["target"],
        buildings=best["buildings"],
        building_heights=best["building_heights"],
        meta=json.dumps(meta),
    )
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[record-physx] wrote {args.out} seed={best['seed']} steps={best['steps']} "
        f"cap={best['captured']} physx_ratio={best.get('physx_ratio')} score={best['score']}",
        flush=True,
    )
    if hasattr(policy, "close"):
        policy.close()
    env.close()


if __name__ == "__main__":
    main()
