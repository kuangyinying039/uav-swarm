"""Run MAPPO/QMIX parameter sensitivity experiments.

Each parameter value is trained from scratch for each seed, then summarized by
the mean of the last N training episodes. This measures how sensitive the
learning-based policy is to environment settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from marl_trainers import IPPOTrainer, MAPPOTrainer, QMIXTrainer, TrainConfig
from train_cooperative_marl import tail_average
from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "marl_sensitivity"

SWEEPS = {
    "target_speed": [0.0, 0.2, 0.42, 0.65],
    "obstacle_speed": [0.0, 0.15, 0.28, 0.45],
    "comm_radius": [4.0, 7.0, 10.0, 14.0],
    "n_obstacles": [8, 16, 24, 35],
}

METRICS = [
    "reward",
    "coverage",
    "target_discovery_rate",
    "target_tracking_rate",
    "target_completion_rate",
    "collisions",
    "collision_rate",
    "steps",
]


def make_env(seed: int, parameter: str, value, env_overrides: dict):
    def factory(episode: int = 0):
        cfg = WeakCommConfig(seed=seed * 100_000 + int(episode))
        setattr(cfg, parameter, value)
        for key, override in env_overrides.items():
            if override is not None:
                setattr(cfg, key, override)
        return WeakCommBeliefGraphEnv(cfg)

    return factory


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def run_one(
    algo: str,
    parameter: str,
    value,
    seeds: list[int],
    train_cfg: TrainConfig,
    env_overrides: dict,
    tail_window: int,
    out_dir: Path,
) -> dict:
    trainer_cls = {
        "mappo": MAPPOTrainer,
        "ippo": IPPOTrainer,
        "qmix": QMIXTrainer,
    }[algo]
    seed_summaries = []
    for seed in seeds:
        trainer = trainer_cls(make_env(seed, parameter, value, env_overrides), train_cfg)
        history = trainer.train()
        summary = tail_average(history, tail_window)
        summary.update({"algo": algo, "parameter": parameter, "value": value, "seed": seed})
        seed_summaries.append(summary)
        seed_path = out_dir / f"{algo}_{parameter}_{value}_seed_{seed}.json"
        seed_path.write_text(
            json.dumps(
                {
                    "algo": algo,
                    "parameter": parameter,
                    "value": value,
                    "seed": seed,
                    "tail_average": summary,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    row = {
        "algo": algo,
        "parameter": parameter,
        "value": value,
        "seeds": ",".join(map(str, seeds)),
        "tail_window": min(tail_window, train_cfg.episodes),
    }
    for metric in METRICS:
        values = np.asarray([summary[metric] for summary in seed_summaries], dtype=float)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=0))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["mappo", "ippo", "qmix", "both", "all"],
        default="both",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Use one seed, two values per parameter, and 30 episodes.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--decision-dt", type=float, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    episodes = 30 if args.quick else args.episodes
    if args.quick:
        seeds = seeds[:1]

    train_cfg = TrainConfig(
        episodes=episodes,
        max_steps=0,
        use_gat=not args.no_gat,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    )
    env_overrides = {
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "search_steps": args.search_steps,
        "target_completion_steps": args.completion_steps,
        "uav_speed": args.uav_speed,
        "decision_dt": args.decision_dt,
        "tpm_prior_base": args.tpm_prior_base,
        "tpm_target_prior_enabled": True if args.enable_target_prior else None,
        "tpm_target_prior_strength": args.tpm_target_prior_strength,
        "tpm_target_prior_sigma": args.tpm_target_prior_sigma,
    }
    if args.algo == "both":
        algos = ["mappo", "qmix"]
    elif args.algo == "all":
        algos = ["mappo", "ippo", "qmix"]
    else:
        algos = [args.algo]
    rows = []
    for algo in algos:
        for parameter, values in SWEEPS.items():
            selected = values[:2] if args.quick else values
            for value in selected:
                rows.append(run_one(algo, parameter, value, seeds, train_cfg, env_overrides, args.tail_window, out_dir))

    fields = ["algo", "parameter", "value", "seeds", "tail_window"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(out_dir / "marl_sensitivity_summary.csv", rows, fields)
    (out_dir / "marl_sensitivity_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
