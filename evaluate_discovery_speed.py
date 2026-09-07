"""Paired discovery-speed evaluation for heuristic, MAPPO and QMIX policies."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig, nominal_search_policy
from marl_trainers import (
    IPPOTrainer,
    MAPPOTrainer,
    QMIXTrainer,
    TrainConfig,
    seed_everything,
    tensorize_heterogeneous_graph,
)


COLORS = {
    "heuristic": "#555555",
    "mappo": "#1f77b4",
    "ippo": "#2ca02c",
    "qmix": "#d62728",
}


def filtered_config(cls, raw: dict):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def load_policy(checkpoint: Path, reference_env: dict):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = payload.get("algorithm")
    if algorithm not in {"mappo", "ippo", "qmix"}:
        raise ValueError(f"Unsupported checkpoint algorithm in {checkpoint}: {algorithm!r}")
    train_cfg = filtered_config(TrainConfig, payload.get("train_config", {}))

    def factory():
        return WeakCommBeliefGraphEnv(filtered_config(WeakCommConfig, reference_env))

    trainer_cls = {
        "mappo": MAPPOTrainer,
        "ippo": IPPOTrainer,
        "qmix": QMIXTrainer,
    }[algorithm]
    trainer = trainer_cls(factory, train_cfg)
    trainer.load_checkpoint(checkpoint)
    if algorithm in {"mappo", "ippo"}:
        trainer.actors.eval()
        trainer.critic.eval()
    else:
        trainer.qnets.eval()
        trainer.target_qnets.eval()
        trainer.mixer.eval()

    def action_fn(env, obs):
        if algorithm in {"mappo", "ippo"}:
            obs_vec = torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=trainer.device)
            adjacency = torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=trainer.device)
            hetero = tensorize_heterogeneous_graph(obs["hetero_graph"], trainer.device) if trainer.cfg.use_hetero_entities else None
            masks = torch.as_tensor(obs["action_masks"], dtype=torch.bool, device=trainer.device)
            with torch.no_grad():
                outputs = trainer.actor(obs_vec, adjacency, hetero) if trainer.actor is not None else None
                if trainer.cfg.use_search_weights:
                    weights = []
                    for i in range(trainer.n_agents):
                        concentration = outputs[i] if outputs is not None else F.softplus(trainer.actors[i](obs_vec[i])) + 0.2
                        concentration = torch.clamp(concentration, 0.05, 100.0)
                        weights.append(concentration / concentration.sum())
                    return env.actions_from_search_weights(torch.stack(weights).cpu().numpy())
                actions = []
                for i in range(trainer.n_agents):
                    logits = outputs[i] if outputs is not None else trainer.actors[i](obs_vec[i])
                    actions.append(int(torch.argmax(logits.masked_fill(~masks[i], -1e9)).item()))
                return np.asarray(actions, dtype=int)
        obs_vec = torch.as_tensor(obs["agent_observations"], dtype=torch.float32)
        adjacency = torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool)
        hetero = tensorize_heterogeneous_graph(obs["hetero_graph"]) if trainer.cfg.use_hetero_entities else None
        masks = torch.as_tensor(obs["action_masks"], dtype=torch.bool)
        with torch.no_grad():
            all_q = trainer.qnet(obs_vec, adjacency, hetero) if trainer.qnet is not None else None
            actions = []
            for i in range(trainer.n_agents):
                q = all_q[i] if all_q is not None else trainer.qnets[i](obs_vec[i])
                actions.append(int(torch.argmax(q.masked_fill(~masks[i], -1e9)).item()))
        return np.asarray(actions, dtype=int)

    return algorithm, payload, action_fn


def threshold_step(curve: list[float], threshold: float, horizon: int) -> int:
    for step, rate in enumerate(curve):
        if rate >= threshold:
            return step
    return horizon + 1


def run_episode(method: str, seed: int, cfg_dict: dict, action_fn):
    cfg_dict = dict(cfg_dict)
    cfg_dict["seed"] = seed
    cfg = filtered_config(WeakCommConfig, cfg_dict)
    seed_everything(seed)
    env = WeakCommBeliefGraphEnv(cfg)
    obs = env.observe_search()
    first_steps = np.full(cfg.n_targets, -1, dtype=int)
    curve = [float(np.mean(env.found_targets))]
    total_reward = 0.0
    collisions = 0
    for _ in range(cfg.search_steps):
        previous = env.found_targets.copy()
        actions = nominal_search_policy(env) if action_fn is None else action_fn(env, obs)
        result = env.step_joint(np.asarray(actions, dtype=int))
        newly_found = env.found_targets & ~previous
        first_steps[newly_found] = env.t
        total_reward += float(result["reward"])
        collisions += int(result.get("collisions", 0))
        curve.append(float(np.mean(env.found_targets)))
        obs = result["obs"]
        if result["done"]:
            break
    final_rate = curve[-1]
    curve.extend([final_rate] * (cfg.search_steps + 1 - len(curve)))
    censored = np.where(first_steps >= 0, first_steps, cfg.search_steps + 1)
    episode = {
        "method": method,
        "seed": seed,
        "horizon": cfg.search_steps,
        "discovered_targets": int(np.count_nonzero(first_steps >= 0)),
        "discovery_rate": float(final_rate),
        "discovery_auc": float(np.mean(curve)),
        "censored_mean_discovery_step": float(np.mean(censored)),
        "step_to_50pct": threshold_step(curve, 0.50, cfg.search_steps),
        "step_to_80pct": threshold_step(curve, 0.80, cfg.search_steps),
        "step_to_90pct": threshold_step(curve, 0.90, cfg.search_steps),
        "step_to_100pct": threshold_step(curve, 1.00, cfg.search_steps),
        "total_reward": total_reward,
        "collisions": collisions,
    }
    targets = [
        {
            "method": method,
            "seed": seed,
            "target_id": target,
            "discovered": int(first_steps[target] >= 0),
            "first_discovery_step": int(first_steps[target]) if first_steps[target] >= 0 else "",
            "censored_step": int(censored[target]),
            "horizon": cfg.search_steps,
        }
        for target in range(cfg.n_targets)
    ]
    curve_rows = [
        {"method": method, "seed": seed, "step": step, "discovery_rate": rate}
        for step, rate in enumerate(curve)
    ]
    return episode, targets, curve_rows


def mean_std(rows: list[dict], methods: list[str]) -> dict:
    metrics = ["discovery_rate", "discovery_auc", "censored_mean_discovery_step", "step_to_50pct", "step_to_80pct", "step_to_90pct", "step_to_100pct"]
    result = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        result[method] = {"n_scenarios": len(subset)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            result[method][f"{metric}_mean"] = float(values.mean())
            result[method][f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def paired_rows(rows: list[dict], methods: list[str]) -> list[dict]:
    """Per-scenario learned-minus-heuristic differences for paired analysis."""
    metrics = ["discovery_rate", "discovery_auc", "censored_mean_discovery_step", "step_to_50pct", "step_to_80pct", "step_to_90pct", "step_to_100pct"]
    lookup = {(row["method"], row["seed"]): row for row in rows}
    output = []
    for method in methods:
        if method == "heuristic":
            continue
        for seed in sorted({row["seed"] for row in rows}):
            learned = lookup[(method, seed)]
            baseline = lookup[("heuristic", seed)]
            row = {"method": method, "seed": seed}
            for metric in metrics:
                row[f"{metric}_delta_vs_heuristic"] = float(learned[metric]) - float(baseline[metric])
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_curve_svg(path: Path, curves: list[dict], methods: list[str], horizon: int) -> None:
    width, height, left, top, right, bottom = 1000, 600, 90, 70, 40, 70
    pw, ph = width - left - right, height - top - bottom
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#fcfcfa"/>']
    body.append('<text x="500" y="34" text-anchor="middle" font-family="Times New Roman" font-size="30" font-weight="bold">Paired discovery-speed evaluation</text>')
    for tick in range(6):
        y = top + tick * ph / 5
        value = 1.0 - tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e4e4df"/>')
        body.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Times New Roman" font-size="18">{value:.1f}</text>')
        x = left + tick * pw / 5
        body.append(f'<text x="{x:.1f}" y="{top+ph+27}" text-anchor="middle" font-family="Times New Roman" font-size="18">{tick*horizon/5:.0f}</text>')
    body.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#222"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#222"/>')
    for index, method in enumerate(methods):
        subset = [row for row in curves if row["method"] == method]
        by_step = []
        for step in range(horizon + 1):
            values = [float(row["discovery_rate"]) for row in subset if int(row["step"]) == step]
            by_step.append(float(np.mean(values)))
        pts = " ".join(f"{left+s/horizon*pw:.1f},{top+(1-r)*ph:.1f}" for s, r in enumerate(by_step))
        color = COLORS.get(method, ["#9467bd", "#ff7f0e"][index % 2])
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        lx = left + 30 + index * 180
        body.append(f'<line x1="{lx}" y1="55" x2="{lx+30}" y2="55" stroke="{color}" stroke-width="4"/><text x="{lx+38}" y="60" font-family="Times New Roman" font-size="19">{html.escape(method)}</text>')
    body.append(f'<text x="{left+pw/2}" y="{height-18}" text-anchor="middle" font-family="Times New Roman" font-size="19">Decision step</text>')
    body.append(f'<text x="22" y="{top+ph/2}" text-anchor="middle" font-family="Times New Roman" font-size="19" transform="rotate(-90 22 {top+ph/2})">Cumulative target discovery rate</text></svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappo", default=None, help="MAPPO .pt checkpoint")
    parser.add_argument("--qmix", default=None, help="QMIX .pt checkpoint")
    parser.add_argument("--seeds", default="101,113,127,139,151,163,179,191,211,227")
    parser.add_argument("--out-dir", default="../outputs/discovery_speed")
    args = parser.parse_args()
    checkpoints = [Path(value) for value in (args.mappo, args.qmix) if value]
    if not checkpoints:
        raise ValueError("Provide at least --mappo or --qmix checkpoint.")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    reference_env = dict(payloads[0].get("env_config", {}))
    reference_env["target_completion_steps"] = 0
    methods = [("heuristic", None)]
    for path in checkpoints:
        algorithm, payload, action_fn = load_policy(path, reference_env)
        for key in ("n_uavs", "n_targets", "grid_size", "search_steps"):
            if key in payload.get("env_config", {}) and key in reference_env and payload["env_config"][key] != reference_env[key]:
                raise ValueError(f"Checkpoint environment mismatch for {key}: {path}")
        methods.append((algorithm, action_fn))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    episode_rows, target_rows, curve_rows = [], [], []
    for seed in seeds:
        for method, action_fn in methods:
            episode, targets, curves = run_episode(method, seed, reference_env, action_fn)
            episode_rows.append(episode)
            target_rows.extend(targets)
            curve_rows.extend(curves)
            print(f"{method:9s} seed={seed} rate={episode['discovery_rate']:.3f} AUC={episode['discovery_auc']:.3f} t80={episode['step_to_80pct']}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "discovery_speed_episodes.csv", episode_rows)
    write_csv(out / "discovery_speed_targets.csv", target_rows)
    write_csv(out / "discovery_speed_curves.csv", curve_rows)
    method_names = [name for name, _ in methods]
    paired = paired_rows(episode_rows, method_names)
    if paired:
        write_csv(out / "discovery_speed_paired_differences.csv", paired)
    summary = mean_std(episode_rows, method_names)
    (out / "discovery_speed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    horizon = int(reference_env.get("search_steps", WeakCommConfig.search_steps))
    write_curve_svg(out / "discovery_speed_curves.svg", curve_rows, method_names, horizon)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
