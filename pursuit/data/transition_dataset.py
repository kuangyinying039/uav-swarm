"""Full (s, a, r, s') teacher rollouts for MATD3 prior replay.

By default Isaac/closed collection keeps **successful captures only** so timeout
trajectories cannot pollute BC / critic pretrain. Pass ``keep_failures=True``
only when an ablation explicitly needs failure coverage.
"""
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


def progress_path_for(dataset_path) -> Path:
    """Partial collects write here; the final ``.pt`` is only written when complete."""
    path = Path(dataset_path)
    return path.with_name(f"{path.stem}.progress.pt")


def _resume_teacher_progress(
    progress_path,
    env_cfg,
    teacher_name,
    execution_backend,
    *,
    keep_failures: bool,
):
    path = Path(progress_path)
    if not path.is_file():
        return [], [], []
    try:
        previous = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[transitions] ignore unreadable progress {path}: {error}", flush=True)
        return [], [], []
    if previous.get("teacher") != teacher_name or previous.get("execution_backend") != execution_backend:
        print(f"[transitions] ignore progress with a different teacher or backend: {path}", flush=True)
        return [], [], []
    if not _same_collection_config(previous.get("env_config"), env_cfg):
        print(f"[transitions] ignore progress from a different environment profile: {path}", flush=True)
        return [], [], []
    if bool(previous.get("keeps_failures", True)) != bool(keep_failures):
        print(f"[transitions] ignore progress with different keep_failures flag: {path}", flush=True)
        return [], [], []
    episodes = list(previous.get("episodes") or [])
    report = list(previous.get("report") or [])
    attempt_report = list(previous.get("attempt_report") or report)
    if len(episodes) != len(report):
        print(f"[transitions] ignore progress with mismatched episodes and report: {path}", flush=True)
        return [], [], []
    if episodes:
        print(
            f"[transitions] resume kept={len(episodes)} attempts={len(attempt_report)} from {path}",
            flush=True,
        )
    return episodes, report, attempt_report


def collect_teacher_transitions(
    env_cfg,
    first_seed,
    rollouts,
    teacher_name,
    progress_path=None,
    *,
    env_factory=None,
    execution_backend="kinematic",
    keep_failures: bool = False,
    keep_near_misses: bool = False,
    near_miss_gap: float = 1.5,
    max_attempts: int | None = None,
):
    """Collect teacher transitions until ``rollouts`` episodes are **kept**.

    Parameters
    ----------
    rollouts:
        Target number of episodes stored in the final dataset.
    keep_failures:
        If False (default), only successful captures are kept — timeouts never
        enter the prior. If True, every rollout is kept (legacy behaviour).
    keep_near_misses:
        If True (and keep_failures is False), also keep unsuccessful episodes
        whose closest capture gap fell below ``near_miss_gap``. These teach
        recovery near the capture boundary without dumping irreversible fails.
    near_miss_gap:
        Minimum capture-gap threshold (metres) for near-miss retention.
    max_attempts:
        Hard cap on teacher rollouts. Defaults to ``rollouts`` when keeping
        failures, otherwise ``4 * rollouts``.
    progress_path:
        Resume/checkpoint file. Must **not** be the final dataset path; callers
        should use :func:`progress_path_for`.
    """
    if rollouts < 1:
        raise ValueError("Need at least one teacher rollout")
    if max_attempts is None:
        max_attempts = int(rollouts if keep_failures else max(rollouts * 4, rollouts))
    if max_attempts < rollouts:
        raise ValueError("max_attempts must be >= rollouts")
    if float(near_miss_gap) <= 0.0:
        raise ValueError("near_miss_gap must be positive")

    episodes, report, attempt_report = (
        _resume_teacher_progress(
            progress_path,
            env_cfg,
            teacher_name,
            execution_backend,
            keep_failures=keep_failures,
        )
        if progress_path is not None
        else ([], [], [])
    )
    attempted_seeds = {int(row["seed"]) for row in attempt_report if "seed" in row}
    next_seed = int(first_seed)
    if attempted_seeds:
        next_seed = max(max(attempted_seeds) + 1, int(first_seed))

    def save_progress():
        if progress_path is None:
            return
        path = Path(progress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = _dataset_payload(
            env_cfg,
            teacher_name,
            episodes,
            report,
            execution_backend,
            keep_failures=keep_failures,
            keep_near_misses=keep_near_misses,
            near_miss_gap=near_miss_gap,
            attempt_report=attempt_report,
            complete=False,
            target_episodes=rollouts,
        )
        temporary = path.with_name(path.name + ".tmp")
        torch.save(partial, temporary)
        os.replace(temporary, path)
        sidecar = path.with_name(path.stem + ".json") if path.suffix == ".pt" else path.with_suffix(".json")
        sidecar_tmp = Path(str(sidecar) + ".tmp")
        sidecar_tmp.write_text(json_sidecar(partial), encoding="utf-8")
        os.replace(sidecar_tmp, sidecar)

    attempts_used = len(attempt_report)
    while len(episodes) < rollouts:
        if attempts_used >= max_attempts:
            raise RuntimeError(
                f"Only kept {len(episodes)}/{rollouts} teacher episodes after "
                f"{attempts_used} attempts (successes="
                f"{sum(1 for row in report if row.get('success'))}); "
                "raise --max-attempts or inspect the teacher under this config"
            )
        seed = next_seed
        next_seed += 1
        attempts_used += 1
        if env_factory is None:
            env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
            obs = env.observe_search()
        else:
            env = env_factory(seed)
            obs = env.reset(seed=seed)
        teacher = BASELINES_3D[teacher_name]()
        teacher.reset(env)
        trajectory = []
        closest_gap = float("inf")
        for step in range(env.cfg.search_steps):
            action, labels = teacher_step(teacher, env)
            active = ~env.disabled_uavs
            result = env.step_joint(action)
            trajectory.append(make_transition(obs, action, result, labels, active, ~env.disabled_uavs))
            obs = result["obs"]
            try:
                closest_gap = min(closest_gap, float(env.minimum_capture_gap()))
            except Exception:
                pass
            if (step + 1) % 50 == 0:
                print(
                    f"[transitions] seed={seed} step={step+1} capture_gap={env.minimum_capture_gap():.3f}",
                    flush=True,
                )
            if result["terminated"] or result["truncated"]:
                break
        success = bool(result.get("capture_success", False))
        near_miss = (
            (not success)
            and bool(keep_near_misses)
            and np.isfinite(closest_gap)
            and float(closest_gap) <= float(near_miss_gap)
        )
        kept = bool(success or keep_failures or near_miss)
        attempt_row = {
            "seed": seed,
            "success": success,
            "near_miss": bool(near_miss),
            "closest_capture_gap": None if not np.isfinite(closest_gap) else float(closest_gap),
            "steps": len(trajectory),
            "kept": kept,
        }
        attempt_report.append(attempt_row)
        if kept:
            episodes.append(trajectory)
            report.append(attempt_row)
        save_progress()
        print(
            f"[transitions] seed={seed} success={success} near_miss={near_miss} "
            f"closest_gap={closest_gap if np.isfinite(closest_gap) else float('nan'):.3f} "
            f"kept_flag={kept} stored={len(episodes)}/{rollouts} "
            f"attempts={attempts_used}/{max_attempts}",
            flush=True,
        )

    return _dataset_payload(
        env_cfg,
        teacher_name,
        episodes,
        report,
        execution_backend,
        keep_failures=keep_failures,
        keep_near_misses=keep_near_misses,
        near_miss_gap=near_miss_gap,
        attempt_report=attempt_report,
        complete=True,
        target_episodes=rollouts,
    )


def _dataset_payload(
    env_cfg,
    teacher_name,
    episodes,
    report,
    execution_backend="kinematic",
    *,
    keep_failures: bool = False,
    keep_near_misses: bool = False,
    near_miss_gap: float = 1.5,
    attempt_report=None,
    complete: bool = True,
    target_episodes: int | None = None,
):
    payload = {
        "format": TRANSITION_DATASET_V2,
        "env_config": asdict(env_cfg),
        "execution_backend": str(execution_backend),
        "teacher": teacher_name,
        "episodes": episodes,
        "report": report,
        "attempt_report": list(attempt_report if attempt_report is not None else report),
        "pursuit_graph_version": PURSUIT_GRAPH_VERSION,
        "keeps_failures": bool(keep_failures),
        "keeps_near_misses": bool(keep_near_misses),
        "near_miss_gap": float(near_miss_gap),
        "action_kind": "proposed",
        "complete": bool(complete),
        "target_episodes": int(target_episodes if target_episodes is not None else len(episodes)),
    }
    return payload


def prior_is_complete(dataset_or_path, *, min_episodes: int, min_successes: int, allow_failures: bool = False) -> bool:
    """Return True only for a finalized, clean prior usable by training."""
    if isinstance(dataset_or_path, (str, Path)):
        path = Path(dataset_or_path)
        if not path.is_file():
            return False
        try:
            import json
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                sidecar = path.with_suffix(".json")
                if sidecar.is_file():
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                else:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
    else:
        payload = dataset_or_path
    n_episodes = int(payload.get("n_episodes", len(payload.get("episodes") or [])))
    n_successes = int(
        payload.get(
            "n_successes",
            sum(1 for row in payload.get("report") or [] if row.get("success")),
        )
    )
    if not bool(payload.get("complete", True)):
        return False
    if n_episodes < int(min_episodes) or n_successes < int(min_successes):
        return False
    allows_near = bool(payload.get("keeps_near_misses", False))
    if not allow_failures and bool(payload.get("keeps_failures", True)) and not allows_near:
        # Legacy polluted priors (all failures stored) are rejected for closed training.
        if n_successes < n_episodes:
            return False
    return True


def json_sidecar(dataset):
    import json

    summary = {
        key: value
        for key, value in dataset.items()
        if key not in ("episodes",)
    }
    summary["n_episodes"] = len(dataset["episodes"])
    summary["n_transitions"] = sum(len(episode) for episode in dataset["episodes"])
    summary["n_successes"] = sum(1 for row in dataset["report"] if row["success"])
    summary["n_attempts"] = len(dataset.get("attempt_report") or dataset["report"])
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
    if expected_backend is not None:
        got = dataset.get("execution_backend")
        allowed = {expected_backend} if isinstance(expected_backend, str) else set(expected_backend)
        # Closed-loop PhysX is a strict subset of the Isaac PhysX contract.
        if "isaac_physx" in allowed:
            allowed.add("isaac_physx_closed")
        if got not in allowed:
            raise ValueError(
                f"Transition backend must be one of {sorted(allowed)!r}; "
                "recollect data in the target simulator"
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
        import warnings

        warnings.warn(
            "Dataset contains no successful episodes; using all prior transitions",
            RuntimeWarning,
            stacklevel=2,
        )
        rows = [row for episode in episodes for row in episode]
    if not rows:
        raise ValueError("Dataset contains no transitions")
    return rows
