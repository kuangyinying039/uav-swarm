"""Held-out pursuit evaluation shared by MAPPO checkpoints and MATD3."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitEnv


def _uavs_in_capture(env):
    _, count, _ = env._capture_geometry()
    return int(count)


def evaluate_policy(action_fn, env_cfg, seeds, method_name="matd3"):
    rows = []
    for seed in seeds:
        env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
        obs, total, components = env.observe_search(), 0.0, {}
        collisions, safety, feasible = 0, 0, 0.0
        correction, emergency = 0.0, 0.0
        closest_capture_gap = float(env.minimum_capture_gap())
        max_uavs_in_capture = _uavs_in_capture(env)
        max_encirclement = 0.0
        for step in range(env.cfg.search_steps):
            action = action_fn(obs, env)
            result = env.step_joint(action)
            obs = result["obs"]
            total += float(result["reward"])
            collisions += int(result.get("collisions", 0))
            safety += int(result.get("continuous_safety_interventions", 0))
            feasible += float(result.get("controller_feasible_rate", 0.0))
            correction += float(result.get("safety_correction_rate", 0.0))
            emergency += float(result.get("emergency_stop_rate", 0.0))
            closest_capture_gap = min(
                closest_capture_gap, float(result.get("minimum_capture_gap", env.minimum_capture_gap()))
            )
            max_uavs_in_capture = max(max_uavs_in_capture, _uavs_in_capture(env))
            max_encirclement = max(max_encirclement, float(result.get("encirclement_score", 0.0)))
            if (step + 1) % 100 == 0:
                print(f"[evaluation] {method_name} seed={seed} step={step+1}", flush=True)
            for key, value in result["reward_components"].items():
                if key in ("pursuit", "estimation"):
                    continue
                components[key] = components.get(key, 0.0) + float(value)
            if result["terminated"] or result["truncated"]:
                break
        rows.append({
            "method": method_name,
            "seed": seed,
            "captured": bool(result["capture_success"]),
            "steps": step + 1,
            "return": total,
            "reward_components": components,
            "initial_layout": env.initial_layout,
            "initial_distances": env.initial_distances,
            "evader_safety_interventions": getattr(env, "evader_safety_interventions", 0),
            "collisions": collisions,
            "safety_interventions": safety,
            "controller_feasible_rate": feasible / (step + 1),
            "safety_correction_rate": correction / (step + 1),
            "emergency_stop_rate": emergency / (step + 1),
            "closest_capture_gap": closest_capture_gap,
            "final_capture_gap": env.minimum_capture_gap(),
            "max_uavs_in_capture": max_uavs_in_capture,
            "final_uavs_in_capture": _uavs_in_capture(env),
            "max_encirclement_score": max_encirclement,
            "final_encirclement_score": float(result.get("encirclement_score", 0.0)),
        })
        print(
            f"[evaluation] {method_name} seed={seed} captured={rows[-1]['captured']} steps={step+1}",
            flush=True,
        )
    return {"env_config": asdict(env_cfg), "rows": rows, "summary": summarize(rows, [method_name])}


def baseline_action_fn(method):
    policy = BASELINES_3D[method]()
    bound = {"env": None}

    def action_fn(obs, env):
        if bound["env"] is not env:
            policy.reset(env)
            bound["env"] = env
        return policy.actions(env)

    return action_fn


def evaluate_methods(action_fn, env_cfg, seeds, methods):
    rows = []
    summaries = {}
    for method in methods:
        if method in ("matd3", "mappo"):
            result = evaluate_policy(action_fn, env_cfg, seeds, method_name=method)
        else:
            result = evaluate_policy(baseline_action_fn(method), env_cfg, seeds, method_name=method)
        rows.extend(result["rows"])
        summaries.update(result["summary"])
    return {"env_config": asdict(env_cfg), "rows": rows, "summary": summaries}


def summarize(rows, methods):
    summary = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        successes = [row["steps"] for row in group if row["captured"]]
        summary[method] = {
            "episodes": len(group),
            "capture_rate": float(np.mean([row["captured"] for row in group])),
            "mean_censored_steps": float(np.mean([row["steps"] for row in group])),
            "mean_success_steps": float(np.mean(successes)) if successes else None,
            "mean_return": float(np.mean([row["return"] for row in group])),
            "mean_closest_capture_gap": float(np.mean([row["closest_capture_gap"] for row in group])),
            "mean_final_capture_gap": float(np.mean([row["final_capture_gap"] for row in group])),
            "mean_collisions": float(np.mean([row["collisions"] for row in group])),
            "mean_safety_interventions": float(np.mean([row["safety_interventions"] for row in group])),
            "mean_controller_feasible_rate": float(np.mean([row["controller_feasible_rate"] for row in group])),
            "mean_safety_correction_rate": float(np.mean([row["safety_correction_rate"] for row in group])),
            "mean_emergency_stop_rate": float(np.mean([row["emergency_stop_rate"] for row in group])),
            "mean_max_uavs_in_capture": float(np.mean([row["max_uavs_in_capture"] for row in group])),
            "near_miss_0_5m_failures": float(
                np.mean([
                    (not row["captured"]) and row["closest_capture_gap"] <= 0.5
                    for row in group
                ])
            ),
        }
    return summary
