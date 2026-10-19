"""Full (s, a, r, s') teacher rollouts for MATD3; successes and failures are kept."""
from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path

import numpy as np
import torch

from pursuit_baselines_3d import BASELINES_3D
from pursuit_graph_encoder import PURSUIT_GRAPH_VERSION, pursuit_graph_from_flat_observation
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv

TRANSITION_DATASET_V2 = "pursuit_transition_dataset_v2"


def copy_graph(graph):
    return {key: np.array(value, copy=True) for key, value in graph.items()}


def _empty_teacher(n_uavs):
    return {
        "teacher_role": np.full(n_uavs, -1, dtype=np.int64),
        "teacher_goal": np.zeros((n_uavs, 3), dtype=np.float32),
        "predicted_target": np.zeros((n_uavs, 3), dtype=np.float32),
        "candidate_velocity": np.zeros((n_uavs, 11, 3), dtype=np.float32),
        "selected_velocity": np.zeros((n_uavs, 3), dtype=np.float32),
    }


def teacher_step(teacher, env):
    if hasattr(teacher, "plan"):
        plan = teacher.plan(env)
        labels = {key: np.array(plan[key], copy=True) for key in _empty_teacher(env.cfg.n_uavs)}
        return np.array(plan["actions"], copy=True), labels
    return np.array(teacher.actions(env), copy=True), _empty_teacher(env.cfg.n_uavs)


def make_transition(obs, action, result, labels, active, next_active):
    next_obs = result["obs"]
    return {
        "agent_observations": np.array(obs["agent_observations"], copy=True),
        "comm_adjacency": np.array(obs["comm_adjacency"], copy=True),
        "hetero_graph": copy_graph(obs["hetero_graph"]),
        "state": np.array(obs["state_vector"], copy=True, dtype=np.float32),
        "actions": np.array(action, copy=True, dtype=np.float32),
        "reward": float(result["reward"]),
        "next_agent_observations": np.array(next_obs["agent_observations"], copy=True),
        "next_comm_adjacency": np.array(next_obs["comm_adjacency"], copy=True),
        "next_hetero_graph": copy_graph(next_obs["hetero_graph"]),
        "next_state": np.array(next_obs["state_vector"], copy=True, dtype=np.float32),
        "terminated": bool(result["terminated"]),
        "truncated": bool(result["truncated"]),
        "active": np.array(active, copy=True),
        "next_active": np.array(next_active, copy=True),
        "capture_success": bool(result.get("capture_success", False)),
        "safety_correction_rate": float(result.get("safety_correction_rate", 0.0)),
        "safety_correction_magnitude": float(result.get("safety_correction_magnitude", 0.0)),
        "safety_reasons": [list(row) for row in result.get("safety_reasons", [])],
        "executed_velocity": np.array(
            result.get("executed_velocity_references", np.zeros((len(action), 3), dtype=np.float32)),
            dtype=np.float32,
        ),
        "encirclement_score": float(result.get("encirclement_score", 0.0)),
        "minimum_capture_gap": float(result.get("minimum_capture_gap", 0.0)),
        **labels,
    }


def _same_collection_config(saved, env_cfg) -> bool:
    left = dict(saved or {})
    right = asdict(env_cfg)
    left.pop("seed", None)
    right.pop("seed", None)
    return left == right


def _resume_teacher_progress(progress_path, env_cfg, teacher_name, execution_backend, first_seed, rollouts):
    path = Path(progress_path)
    if not path.is_file():
        return {}
    try:
        previous = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[transitions] ignore unreadable progress {path}: {error}", flush=True)
        return {}
    if previous.get("teacher") != teacher_name or previous.get("execution_backend") != execution_backend:
        print(f"[transitions] ignore progress with a different teacher or backend: {path}", flush=True)
        return {}
    if not _same_collection_config(previous.get("env_config"), env_cfg):
        print(f"[transitions] ignore progress from a different environment profile: {path}", flush=True)
        return {}
    episodes = list(previous.get("episodes") or [])
    report = list(previous.get("report") or [])
    if len(episodes) != len(report):
        print(f"[transitions] ignore progress with mismatched episodes and report: {path}", flush=True)
        return {}
    wanted = set(range(int(first_seed), int(first_seed) + int(rollouts)))
    kept = {}
    for episode, row in zip(episodes, report):
        seed = int(row.get("seed", -1))
        if seed in wanted and seed not in kept:
            kept[seed] = (episode, row)
    if kept:
        print(f"[transitions] resume kept={len(kept)}/{rollouts} from {path}", flush=True)
    return kept


def collect_teacher_transitions(
    env_cfg, first_seed, rollouts, teacher_name, progress_path=None,
    *, env_factory=None, execution_backend="kinematic",
):
    """Roll out the teacher and keep every episode, including timeouts."""
    if rollouts < 1:
        raise ValueError("Need at least one teacher rollout")
    resumed = (
        _resume_teacher_progress(
            progress_path, env_cfg, teacher_name, execution_backend, first_seed, rollouts
        )
        if progress_path is not None
        else {}
    )
    episodes, report = [], []
    for seed in range(first_seed, first_seed + rollouts):
        if seed in resumed:
            episode, row = resumed[seed]
            episodes.append(episode)
            report.append(row)
            continue
        if env_factory is None:
            env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
            obs = env.observe_search()
        else:
            env = env_factory(seed)
            obs = env.reset(seed=seed)
        teacher = BASELINES_3D[teacher_name]()
        teacher.reset(env)
        trajectory = []
        for step in range(env.cfg.search_steps):
            action, labels = teacher_step(teacher, env)
            active = ~env.disabled_uavs
            result = env.step_joint(action)
            trajectory.append(make_transition(obs, action, result, labels, active, ~env.disabled_uavs))
            obs = result["obs"]
            if (step + 1) % 50 == 0:
                print(
                    f"[transitions] seed={seed} step={step+1} capture_gap={env.minimum_capture_gap():.3f}",
                    flush=True,
                )
            if result["terminated"] or result["truncated"]:
                break
        success = bool(result.get("capture_success", False))
        episodes.append(trajectory)
        report.append({"seed": seed, "success": success, "steps": len(trajectory), "kept": True})
        if progress_path is not None:
            progress_path = Path(progress_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            partial = _dataset_payload(
                env_cfg, teacher_name, episodes, report, execution_backend
            )
            temporary = progress_path.with_suffix(".pt.tmp")
            torch.save(partial, temporary)
            os.replace(temporary, progress_path)
            sidecar = progress_path.with_suffix(".json")
            sidecar_tmp = sidecar.with_suffix(".json.tmp")
            sidecar_tmp.write_text(json_sidecar(partial), encoding="utf-8")
            os.replace(sidecar_tmp, sidecar)
        print(
            f"[transitions] seed={seed} success={success} kept={len(episodes)}/{rollouts}",
            flush=True,
        )
    return _dataset_payload(env_cfg, teacher_name, episodes, report, execution_backend)


def _dataset_payload(env_cfg, teacher_name, episodes, report, execution_backend="kinematic"):
    return {
        "format": TRANSITION_DATASET_V2,
        "env_config": asdict(env_cfg),
        "execution_backend": str(execution_backend),
        "teacher": teacher_name,
        "episodes": episodes,
        "report": report,
        "pursuit_graph_version": PURSUIT_GRAPH_VERSION,
        "keeps_failures": True,
        "action_kind": "proposed",
    }


def json_sidecar(dataset):
    import json

    summary = {key: value for key, value in dataset.items() if key != "episodes"}
    summary["n_episodes"] = len(dataset["episodes"])
    summary["n_transitions"] = sum(len(episode) for episode in dataset["episodes"])
    summary["n_successes"] = sum(1 for row in dataset["report"] if row["success"])
    return json.dumps(summary, indent=2)


# Reward / shaping weights may differ for clean ablations that reuse the same
# Medium geometry prior (e.g. A4c: visibility weight off, shared MPC prior).
REWARD_ABLATION_IGNORE_KEYS = (
    "visibility_progress_weight",
    "encirclement_progress_weight",
    "capture_team_progress_weight",
    "individual_approach_weight",
    "nearest_approach_weight",
)


def validate_transition_dataset(
    dataset, env_cfg, expected_backend=None, *, ignore_config_keys=(),
):
    if dataset.get("format") != TRANSITION_DATASET_V2:
        raise ValueError("Expected pursuit_transition_dataset_v2; this file is not TD-complete")
    if expected_backend is not None and dataset.get("execution_backend") != expected_backend:
        raise ValueError(
            f"Transition backend must be {expected_backend!r}; recollect data in the target simulator"
        )
    recorded = asdict(QuadrotorPursuitConfig(**dataset["env_config"]))
    current = asdict(env_cfg)
    recorded.pop("seed", None)
    current.pop("seed", None)
    for key in ignore_config_keys:
        recorded.pop(key, None)
        current.pop(key, None)
    if recorded != current:
        raise ValueError("Transition environment differs from training; regenerate with the same config")
    if not dataset["episodes"] or any(not episode for episode in dataset["episodes"]):
        raise ValueError("Empty transition dataset")
    required = (
        "agent_observations", "comm_adjacency", "hetero_graph", "state", "actions", "reward",
        "next_agent_observations", "next_comm_adjacency", "next_hetero_graph", "next_state",
        "terminated", "truncated", "active", "next_active",
    )
    for episode in dataset["episodes"]:
        for row in episode:
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"Transition missing {missing}")
            graph = row.get("hetero_graph", {})
            if graph.get("self_nodes") is None or graph.get("uav_xyz") is None:
                row["hetero_graph"] = pursuit_graph_from_flat_observation(row["agent_observations"], env_cfg)
            next_graph = row.get("next_hetero_graph", {})
            if next_graph.get("self_nodes") is None or next_graph.get("uav_xyz") is None:
                row["next_hetero_graph"] = pursuit_graph_from_flat_observation(
                    row["next_agent_observations"], env_cfg
                )
    dataset["pursuit_graph_version"] = PURSUIT_GRAPH_VERSION


def load_transition_dataset(
    path, env_cfg, expected_backend=None, *, ignore_config_keys=(),
):
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    validate_transition_dataset(
        dataset,
        env_cfg,
        expected_backend=expected_backend,
        ignore_config_keys=ignore_config_keys,
    )
    return dataset


def flatten_transitions(dataset):
    rows = [row for episode in dataset["episodes"] for row in episode]
    if not rows:
        raise ValueError("Dataset contains no transitions")
    return rows


def flatten_successful_transitions(dataset):
    """Keep transitions from episodes the dataset report marks as captures."""
    episodes = list(dataset["episodes"])
    report = list(dataset.get("report") or [])
    if len(report) != len(episodes):
        raise ValueError(
            f"Cannot filter successes: report has {len(report)} rows for {len(episodes)} episodes"
        )
    rows = [
        row
        for episode, meta in zip(episodes, report)
        if meta.get("success")
        for row in episode
    ]
    if not rows:
        raise ValueError("Dataset contains no successful episodes")
    return rows
