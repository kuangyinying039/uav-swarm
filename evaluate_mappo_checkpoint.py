"""Deterministically evaluate a trained MAPPO policy against the heuristic.

Training seeds and evaluation seeds should be disjoint. Both methods are run on
the same environment configuration and the same evaluation seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig, nominal_search_policy
from marl_trainers import MAPPOTrainer, TrainConfig, seed_everything, tensorize_heterogeneous_graph


METRICS = (
    "reward",
    "coverage",
    "target_discovery_rate",
    "target_tracking_rate",
    "target_completion_rate",
    "collisions",
    "collision_rate",
    "outage_rate",
    "disabled_rate",
    "safety_intervention_rate",
    "new_cells",
    "revisit_gain",
    "steps",
)


def config_from_payload(payload: dict, seed: int) -> WeakCommConfig:
    allowed = {item.name for item in fields(WeakCommConfig)}
    values = {key: value for key, value in payload.get("env_config", {}).items() if key in allowed}
    values["seed"] = seed
    return WeakCommConfig(**values)


def episode_metrics(env, totals: dict) -> dict:
    return {
        "reward": float(totals["reward"]),
        "coverage": float(env.coverage()),
        "target_discovery_rate": float(np.mean(env.found_targets)),
        "target_tracking_rate": float(np.mean(env.tracked_targets | env.completed_targets)),
        "target_completion_rate": float(np.mean(env.completed_targets)),
        "collisions": int(totals["collisions"]),
        "collision_rate": float(totals["collisions"] / max(env.t, 1)),
        "outage_rate": float(np.mean(totals["outage_rates"])) if totals["outage_rates"] else 0.0,
        "disabled_rate": float(np.mean(totals["disabled_rates"])) if totals["disabled_rates"] else 0.0,
        "safety_intervention_rate": float(np.mean(totals["safety_rates"])) if totals["safety_rates"] else 0.0,
        "new_cells": int(totals["new_cells"]),
        "revisit_gain": float(totals["revisit_gain"]),
        "steps": int(env.t),
    }


def run_episode(env: WeakCommBeliefGraphEnv, action_fn) -> dict:
    obs = env.observe_search()
    totals = {"reward": 0.0, "collisions": 0, "new_cells": 0, "revisit_gain": 0.0, "outage_rates": [], "disabled_rates": [], "safety_rates": []}
    for _ in range(env.cfg.search_steps):
        actions = action_fn(env, obs)
        result = env.step_joint(np.asarray(actions, dtype=int))
        totals["reward"] += float(result["reward"])
        totals["collisions"] += int(result.get("collisions", 0))
        totals["new_cells"] += int(result.get("new_cells", 0))
        totals["revisit_gain"] += float(result.get("revisit_gain", 0.0))
        totals["outage_rates"].append(float(result.get("outage_rate", 0.0)))
        totals["disabled_rates"].append(float(result.get("disabled_rate", 0.0)))
        totals["safety_rates"].append(float(result.get("safety_intervention_rate", 0.0)))
        obs = result["obs"]
        if result["done"]:
            break
    return episode_metrics(env, totals)


def summarize(rows: list[dict]) -> dict:
    summary = {"n_episodes": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="101,113,127,139,151")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_allowed = {item.name for item in fields(TrainConfig)}
    train_cfg = TrainConfig(**{key: value for key, value in payload["train_config"].items() if key in train_allowed})
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    base_seed = seeds[0]
    trainer = MAPPOTrainer(lambda episode=0: WeakCommBeliefGraphEnv(config_from_payload(payload, base_seed)), train_cfg)
    trainer.load_checkpoint(checkpoint)
    trainer.actors.eval()
    trainer.critic.eval()

    def mappo_action(_env, obs):
        obs_vec = torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=trainer.device)
        adjacency = torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=trainer.device)
        hetero_graph = tensorize_heterogeneous_graph(obs["hetero_graph"], trainer.device) if trainer.cfg.use_hetero_entities else None
        masks = torch.as_tensor(obs["action_masks"], dtype=torch.bool, device=trainer.device)
        with torch.no_grad():
            all_outputs = trainer.actor(obs_vec, adjacency, hetero_graph) if trainer.actor is not None else None
            if trainer.cfg.use_search_weights:
                weights = []
                for i in range(trainer.n_agents):
                    concentration = all_outputs[i] if all_outputs is not None else F.softplus(trainer.actors[i](obs_vec[i])) + 0.2
                    concentration = torch.clamp(concentration, min=0.05, max=100.0)
                    weights.append(concentration / concentration.sum())
                actions = _env.actions_from_search_weights(torch.stack(weights).cpu().numpy()).tolist()
            else:
                actions = []
                for i in range(trainer.n_agents):
                    logits = all_outputs[i] if all_outputs is not None else trainer.actors[i](obs_vec[i])
                    logits = logits.masked_fill(~masks[i], -1e9)
                    actions.append(int(torch.argmax(logits).item()))
        return actions

    def heuristic_action(env, _obs):
        return nominal_search_policy(env)

    rows = []
    for seed in seeds:
        for method, action_fn in (("mappo", mappo_action), ("heuristic", heuristic_action)):
            seed_everything(seed)
            env = WeakCommBeliefGraphEnv(config_from_payload(payload, seed))
            row = run_episode(env, action_fn)
            row.update({"method": method, "seed": seed})
            rows.append(row)

    result = {
        "checkpoint": str(checkpoint),
        "evaluation_seeds": seeds,
        "mappo": summarize([row for row in rows if row["method"] == "mappo"]),
        "heuristic": summarize([row for row in rows if row["method"] == "heuristic"]),
        "episodes": rows,
    }
    out = Path(args.out) if args.out else checkpoint.with_name(f"{checkpoint.stem}_evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *METRICS])
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in writer.fieldnames} for row in rows])
    print(json.dumps({"mappo": result["mappo"], "heuristic": result["heuristic"]}, indent=2))


if __name__ == "__main__":
    main()
