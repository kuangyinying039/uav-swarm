#!/usr/bin/env python3
"""Emit or run the complete Radar5 pursuit paper experiment suite.

Default training domain is Medium: Nominal is easier transfer, Hard is harder
zero-shot transfer. MPC remains an expert upper bound, not a must-beat target.

Stages:
  assets      collect MPC transitions, success demos, and DAgger imitation
  baselines   APF / FRPN / MPC on Nominal, Medium, Hard
  train_main  HGAT-MATD3-Opt and MAPPO warmstart (seeds 11/12/13)
  train_ablation  A0–A4 (A5 is train_main)
  evaluate    held-out 200-seed tests across three difficulties
  aggregate   multi-seed bootstrap + five-method difficulty figure
  qualitative fixed-seed trajectory renderings
  all         assets → … → qualitative
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

LEVELS = ("nominal", "medium", "hard")
STRESS_LEVELS = ("extreme",)
ALL_EVAL_LEVELS = LEVELS + STRESS_LEVELS
LEVEL_TITLE = {
    "nominal": "Nominal",
    "medium": "Medium",
    "hard": "Hard",
    "extreme": "Extreme",
}
DEFAULT_SEEDS = (11, 12, 13)


@dataclass(frozen=True)
class SuitePaths:
    train_level: str
    prefix: str

    @property
    def train_config(self) -> Path:
        return ROOT / "configs" / "pursuit_v2" / f"radar5_benchmark_{self.train_level}.json"

    def level_config(self, level: str) -> Path:
        return ROOT / "configs" / "pursuit_v2" / f"radar5_benchmark_{level}.json"

    @property
    def no_visibility_config(self) -> Path:
        if self.train_level == "medium":
            return ROOT / "configs" / "pursuit_v2" / "radar5_ablation_no_visibility_medium.json"
        return ROOT / "configs" / "pursuit_v2" / "radar5_ablation_no_visibility.json"

    @property
    def transitions(self) -> str:
        return f"outputs/{self.prefix}_mpc_transitions200.pt"

    @property
    def no_visibility_transitions(self) -> str:
        return f"outputs/{self.prefix}_mpc_transitions200_no_visibility.pt"

    @property
    def success_demos(self) -> str:
        return f"outputs/{self.prefix}_mpc_success50.pt"

    @property
    def dagger_dir(self) -> str:
        return f"outputs/{self.prefix}_dagger"

    @property
    def dagger_actor(self) -> str:
        return f"{self.dagger_dir}/best_imitation.pt"

    @property
    def dagger_corrections(self) -> str:
        return f"{self.dagger_dir}/corrections_round_02.pt"

    def matd3_run(self, variant: str, seed: int) -> str:
        return f"outputs/{self.prefix}_{variant}_seed{seed}"

    def mappo_run(self, seed: int) -> str:
        return f"outputs/{self.prefix}_mappo_dagger_replay_seed{seed}"

    def baseline(self, level: str) -> str:
        return f"outputs/{self.prefix}_benchmark_{level}_baselines_200.json"

    def eval_matd3(self, variant: str, seed: int, level: str) -> str:
        return f"outputs/{self.prefix}_eval_{variant}_seed{seed}_{level}"

    def eval_mappo(self, seed: int, level: str) -> str:
        return f"outputs/{self.prefix}_eval_mappo_seed{seed}_{level}"

    def aggregate(self, label: str, level: str) -> str:
        return f"outputs/{self.prefix}_aggregate_{label.lower()}_{level}.json"

    @property
    def difficulty_dir(self) -> str:
        return f"outputs/{self.prefix}_difficulty_five_methods"

    @property
    def difficulty_dir_with_extreme(self) -> str:
        return f"outputs/{self.prefix}_difficulty_with_extreme"

    @property
    def qualitative_dir(self) -> str:
        return f"outputs/{self.prefix}_qualitative"

    @property
    def ablation_plot_dir(self) -> str:
        return f"outputs/{self.prefix}_ablation_comparison"

    @property
    def same_scene_dir(self) -> str:
        return f"outputs/{self.prefix}_same_scene"


def matd3_opt_flags() -> list[str]:
    return [
        "--prior-fraction", "0.5",
        "--warmup-policy", "actor",
        "--critic-pretrain-updates", "5000",
        "--actor-lr", "5e-5",
        "--critic-lr", "3e-4",
        "--exploration-std", "0.10",
        "--exploration-final-std", "0.03",
        "--exploration-decay-steps", "150000",
        "--demo-bc-weight", "1.0",
        "--demo-bc-final-weight", "0.05",
        "--demo-bc-decay-steps", "150000",
        "--validation-seed", "4000000",
        "--validation-episodes", "50",
        "--validation-interval", "100",
        "--step-checkpoint-interval", "10000",
    ]


def common_train_tail(episodes: int, seed: int, out: str) -> list[str]:
    return ["--episodes", str(episodes), "--seed", str(seed), "--out", out]


def assets_commands(paths: SuitePaths, episodes: int) -> list[dict]:
    # episodes is unused here; kept for signature symmetry.
    del episodes
    train = str(paths.train_config)
    return [
        {
            "id": "collect_transitions",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "collect-transitions",
                "--env-config", train,
                "--demo-seed", "2000000",
                "--rollouts", "200",
                "--teacher", "mpc",
                "--dataset", paths.transitions,
            ],
        },
        {
            "id": "collect_success_demos",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_with_demos.py"), "collect",
                "--env-config", train,
                "--demo-seed", "2000000",
                "--demo-successes", "50",
                "--demo-attempts", "400",
                "--teacher", "mpc",
                "--demos", paths.success_demos,
                "--out", f"outputs/{paths.prefix}_collect_success",
            ],
        },
        {
            "id": "dagger_pretrain",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_with_demos.py"), "pretrain",
                "--env-config", train,
                "--demos", paths.success_demos,
                "--bc-updates", "1500",
                "--correction-rounds", "2",
                "--correction-episodes", "40",
                "--correction-updates", "800",
                "--correction-seed", "5000000",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--seed", "11",
                "--out", paths.dagger_dir,
            ],
        },
        {
            "id": "collect_no_visibility_transitions",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "collect-transitions",
                "--env-config", str(paths.no_visibility_config),
                "--demo-seed", "2000000",
                "--rollouts", "200",
                "--teacher", "mpc",
                "--dataset", paths.no_visibility_transitions,
            ],
        },
    ]


def baseline_commands(paths: SuitePaths, levels: Iterable[str] = LEVELS) -> list[dict]:
    rows = []
    for level in levels:
        rows.append({
            "id": f"baseline_{level}",
            "command": [
                PYTHON, str(ROOT / "evaluate_3d_baselines.py"),
                "--env-config", str(paths.level_config(level)),
                "--methods", "apf", "frpn", "mpc",
                "--seed-start", "7000000",
                "--seed-count", "200",
                "--workers", "8",
                "--out", paths.baseline(level),
            ],
        })
    return rows


def train_main_commands(paths: SuitePaths, seeds: Iterable[int], episodes: int) -> list[dict]:
    rows = []
    train = str(paths.train_config)
    for seed in seeds:
        out = paths.matd3_run("hgat_matd3_opt", seed)
        rows.append({
            "id": f"train_hgat_matd3_opt_seed{seed}",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", train,
                "--actor-init", paths.dagger_actor,
                "--prior", paths.transitions,
                *matd3_opt_flags(),
                *common_train_tail(episodes, seed, out),
            ],
        })
        rows.append({
            "id": f"train_mappo_seed{seed}",
            # MAPPO forbids --env-config together with --checkpoint; the DAgger
            # checkpoint already encodes the training environment.
            "command": [
                PYTHON, str(ROOT / "train_pursuit_with_demos.py"), "train",
                "--checkpoint", paths.dagger_actor,
                "--demos", paths.success_demos,
                "--aux-demos", paths.dagger_corrections,
                "--training-profile", "warmstart",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--validation-interval", "100",
                *common_train_tail(episodes, seed, paths.mappo_run(seed)),
            ],
        })
    return rows


def train_ablation_commands(paths: SuitePaths, seeds: Iterable[int], episodes: int) -> list[dict]:
    """Paper ablations A0–A4. A5 is the main HGAT-MATD3-Opt run."""
    rows = []
    train = str(paths.train_config)
    for seed in seeds:
        rows.append({
            "id": f"ablation_A0_mlp_scratch_seed{seed}",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", train,
                "--no-gat",
                "--prior-fraction", "0",
                "--warmup-policy", "random",
                "--critic-pretrain-updates", "0",
                "--demo-bc-weight", "0",
                "--demo-bc-final-weight", "0",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--validation-interval", "100",
                "--step-checkpoint-interval", "10000",
                *common_train_tail(episodes, seed, paths.matd3_run("ablation_a0_mlp_scratch", seed)),
            ],
        })
        rows.append({
            "id": f"ablation_A1_hgat_scratch_seed{seed}",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", train,
                "--prior-fraction", "0",
                "--warmup-policy", "random",
                "--critic-pretrain-updates", "0",
                "--demo-bc-weight", "0",
                "--demo-bc-final-weight", "0",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--validation-interval", "100",
                "--step-checkpoint-interval", "10000",
                *common_train_tail(episodes, seed, paths.matd3_run("ablation_a1_hgat_scratch", seed)),
            ],
        })
        rows.append({
            "id": f"ablation_A2_dagger_prior_no_retention_seed{seed}",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", train,
                "--actor-init", paths.dagger_actor,
                "--prior", paths.transitions,
                "--prior-fraction", "0.5",
                "--warmup-policy", "actor",
                "--critic-pretrain-updates", "0",
                "--demo-bc-weight", "0",
                "--demo-bc-final-weight", "0",
                "--actor-lr", "5e-5",
                "--critic-lr", "3e-4",
                "--exploration-std", "0.10",
                "--exploration-final-std", "0.03",
                "--exploration-decay-steps", "150000",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--validation-interval", "100",
                "--step-checkpoint-interval", "10000",
                *common_train_tail(episodes, seed, paths.matd3_run("ablation_a2_no_retention", seed)),
            ],
        })
        rows.append({
            "id": f"ablation_A3_no_critic_pretrain_seed{seed}",
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", train,
                "--actor-init", paths.dagger_actor,
                "--prior", paths.transitions,
                "--prior-fraction", "0.5",
                "--warmup-policy", "actor",
                "--critic-pretrain-updates", "0",
                "--actor-lr", "5e-5",
                "--critic-lr", "3e-4",
                "--exploration-std", "0.10",
                "--exploration-final-std", "0.03",
                "--exploration-decay-steps", "150000",
                "--demo-bc-weight", "1.0",
                "--demo-bc-final-weight", "0.05",
                "--demo-bc-decay-steps", "150000",
                "--validation-seed", "4000000",
                "--validation-episodes", "50",
                "--validation-interval", "100",
                "--step-checkpoint-interval", "10000",
                *common_train_tail(episodes, seed, paths.matd3_run("ablation_a3_no_critic_pretrain", seed)),
            ],
        })
        rows.append({
            "id": f"ablation_A4_no_visibility_seed{seed}",
            # Prior rewards must match the no-visibility MDP; reuse of the
            # standard Medium transitions would fail env-config validation.
            "command": [
                PYTHON, str(ROOT / "train_pursuit_matd3.py"), "train",
                "--env-config", str(paths.no_visibility_config),
                "--actor-init", paths.dagger_actor,
                "--prior", paths.no_visibility_transitions,
                *matd3_opt_flags(),
                *common_train_tail(episodes, seed, paths.matd3_run("ablation_a4_no_visibility", seed)),
            ],
        })
    return rows


def evaluate_commands(
    paths: SuitePaths,
    seeds: Iterable[int],
    *,
    include_main: bool = True,
    ablation_ids: Iterable[str] = (),
    levels: Iterable[str] = LEVELS,
) -> list[dict]:
    rows = []
    level_list = tuple(levels)
    variants = []
    if include_main:
        variants.append("hgat_matd3_opt")
    variants.extend(list(ablation_ids))
    for seed in seeds:
        for level in level_list:
            for variant in variants:
                rows.append({
                    "id": f"eval_{variant}_seed{seed}_{level}",
                    "command": [
                        PYTHON, str(ROOT / "train_pursuit_matd3.py"), "evaluate",
                        "--checkpoint", f"{paths.matd3_run(variant, seed)}/best_capture.pt",
                        "--eval-env-config", str(paths.level_config(level)),
                        "--eval-seed", "7000000",
                        "--eval-episodes", "200",
                        "--methods", "matd3",
                        "--out", paths.eval_matd3(variant, seed, level),
                    ],
                })
            if include_main:
                rows.append({
                    "id": f"eval_mappo_seed{seed}_{level}",
                    "command": [
                        PYTHON, str(ROOT / "train_pursuit_with_demos.py"), "evaluate",
                        "--checkpoint", f"{paths.mappo_run(seed)}/best_capture.pt",
                        "--eval-env-config", str(paths.level_config(level)),
                        "--eval-seed", "7000000",
                        "--eval-episodes", "200",
                        "--methods", "mappo",
                        "--out", paths.eval_mappo(seed, level),
                    ],
                })
    return rows


def aggregate_commands(
    paths: SuitePaths,
    seeds: Iterable[int],
    *,
    include_main: bool = True,
    ablation_ids: Iterable[str] = (),
    levels: Iterable[str] = LEVELS,
) -> list[dict]:
    rows = []
    seed_list = list(seeds)
    level_list = tuple(levels)
    labels = []
    if include_main:
        labels.extend([("HGAT_MATD3_OPT", "hgat_matd3_opt"), ("MAPPO", "mappo")])
    labels.extend((ablation.upper(), ablation) for ablation in ablation_ids)
    for level in level_list:
        for label, variant in labels:
            if variant == "mappo":
                inputs = [f"{paths.eval_mappo(seed, level)}/evaluation.json" for seed in seed_list]
            else:
                inputs = [f"{paths.eval_matd3(variant, seed, level)}/evaluation.json" for seed in seed_list]
            cmd = [
                PYTHON, str(ROOT / "aggregate_pursuit_training_seeds.py"),
                "--label", label,
                "--out", paths.aggregate(label, level),
            ]
            for path in inputs:
                cmd.extend(["--input", path])
            rows.append({"id": f"aggregate_{label.lower()}_{level}", "command": cmd})

    if include_main and tuple(level_list) == LEVELS:
        compare = [
            PYTHON, str(ROOT / "compare_pursuit_difficulty.py"),
            "--require-methods", "APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3_OPT",
            "--out-dir", paths.difficulty_dir,
        ]
        for level in LEVELS:
            title = LEVEL_TITLE[level]
            for method in ("APF", "FRPN", "MPC"):
                compare.extend(["--input", f"{title}:{method}={paths.baseline(level)}"])
            compare.extend(["--input", f"{title}:MAPPO={paths.aggregate('MAPPO', level)}"])
            compare.extend([
                "--input",
                f"{title}:HGAT_MATD3_OPT={paths.aggregate('HGAT_MATD3_OPT', level)}",
            ])
        rows.append({"id": "difficulty_five_methods", "command": compare})
    if include_main and tuple(level_list) == ALL_EVAL_LEVELS:
        compare = [
            PYTHON, str(ROOT / "compare_pursuit_difficulty.py"),
            "--require-methods", "APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3_OPT",
            "--require-difficulties", "Nominal", "Medium", "Hard", "Extreme",
            "--out-dir", paths.difficulty_dir_with_extreme,
        ]
        for level in ALL_EVAL_LEVELS:
            title = LEVEL_TITLE[level]
            for method in ("APF", "FRPN", "MPC"):
                compare.extend(["--input", f"{title}:{method}={paths.baseline(level)}"])
            compare.extend(["--input", f"{title}:MAPPO={paths.aggregate('MAPPO', level)}"])
            compare.extend([
                "--input",
                f"{title}:HGAT_MATD3_OPT={paths.aggregate('HGAT_MATD3_OPT', level)}",
            ])
        rows.append({"id": "difficulty_with_extreme", "command": compare})
    return rows


def qualitative_commands(paths: SuitePaths, seed: int = 11) -> list[dict]:
    """Render one success-oriented and one hard-transfer qualitative case."""
    ckpt = f"{paths.matd3_run('hgat_matd3_opt', seed)}/best_capture.pt"
    cases = (
        ("train_domain", paths.train_level, 7000017),
        ("hard_transfer", "hard", 7000083),
    )
    rows = []
    for name, level, scene in cases:
        rows.append({
            "id": f"qualitative_{name}",
            "command": [
                PYTHON, str(ROOT / "visualize_3d_pursuit_episode.py"),
                "--method", "matd3",
                "--checkpoint", ckpt,
                "--env-config", str(paths.level_config(level)),
                "--seed", str(scene),
                "--steps", "300",
                "--out-dir", f"{paths.qualitative_dir}/{name}",
            ],
        })
    return rows


def plot_ablation_commands(paths: SuitePaths) -> list[dict]:
    """Bar chart of A0–A4 plus main HGAT-MATD3-Opt / MAPPO on the train domain."""
    level = paths.train_level
    cmd = [
        PYTHON, str(ROOT / "compare_pursuit_evaluations.py"),
        "--out-dir", paths.ablation_plot_dir,
        "--input", f"A0_MLP_scratch={paths.aggregate('ABLATION_A0_MLP_SCRATCH', level)}",
        "--input", f"A1_HGAT_scratch={paths.aggregate('ABLATION_A1_HGAT_SCRATCH', level)}",
        "--input", f"A2_no_retention={paths.aggregate('ABLATION_A2_NO_RETENTION', level)}",
        "--input", f"A3_no_critic_pretrain={paths.aggregate('ABLATION_A3_NO_CRITIC_PRETRAIN', level)}",
        "--input", f"A4_no_visibility={paths.aggregate('ABLATION_A4_NO_VISIBILITY', level)}",
        "--input", f"A5_HGAT_MATD3_OPT={paths.aggregate('HGAT_MATD3_OPT', level)}",
        "--input", f"MAPPO={paths.aggregate('MAPPO', level)}",
    ]
    return [{"id": "plot_ablation_medium", "command": cmd}]


def same_scene_commands(
    paths: SuitePaths,
    *,
    train_seed: int = 11,
    scene_seed: int = 7000017,
    level: str | None = None,
) -> list[dict]:
    """Render APF/FRPN/MPC/MAPPO/MATD3 on one fixed held-out scene."""
    level = level or paths.train_level
    env = str(paths.level_config(level))
    out_root = f"{paths.same_scene_dir}/seed{scene_seed}_{level}"
    rows = []
    for method in ("apf", "frpn", "mpc"):
        rows.append({
            "id": f"same_scene_{method}",
            "command": [
                PYTHON, str(ROOT / "visualize_3d_pursuit_episode.py"),
                "--method", method,
                "--env-config", env,
                "--seed", str(scene_seed),
                "--steps", "300",
                "--out-dir", f"{out_root}/{method}",
            ],
        })
    rows.append({
        "id": "same_scene_mappo",
        "command": [
            PYTHON, str(ROOT / "visualize_3d_pursuit_episode.py"),
            "--method", "mappo",
            "--checkpoint", f"{paths.mappo_run(train_seed)}/best_capture.pt",
            "--env-config", env,
            "--seed", str(scene_seed),
            "--steps", "300",
            "--out-dir", f"{out_root}/mappo",
        ],
    })
    rows.append({
        "id": "same_scene_matd3",
        "command": [
            PYTHON, str(ROOT / "visualize_3d_pursuit_episode.py"),
            "--method", "matd3",
            "--checkpoint", f"{paths.matd3_run('hgat_matd3_opt', train_seed)}/best_capture.pt",
            "--env-config", env,
            "--seed", str(scene_seed),
            "--steps", "300",
            "--out-dir", f"{out_root}/matd3",
        ],
    })
    return rows


ABLATION_VARIANTS = (
    "ablation_a0_mlp_scratch",
    "ablation_a1_hgat_scratch",
    "ablation_a2_no_retention",
    "ablation_a3_no_critic_pretrain",
    "ablation_a4_no_visibility",
)


def build_manifest(args) -> dict:
    prefix = args.prefix or f"radar5_{args.train_level}"
    paths = SuitePaths(train_level=args.train_level, prefix=prefix)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("Need at least one training seed")
    if args.train_level not in LEVELS:
        raise ValueError(f"train_level must be one of {LEVELS}")
    if not paths.no_visibility_config.is_file():
        raise FileNotFoundError(paths.no_visibility_config)

    stage = args.stage
    jobs: list[dict] = []
    if stage in ("assets", "all"):
        jobs.extend(assets_commands(paths, args.episodes))
    if stage in ("baselines", "all"):
        jobs.extend(baseline_commands(paths, LEVELS))
    if stage == "baselines_extreme":
        jobs.extend(baseline_commands(paths, STRESS_LEVELS))
    if stage in ("train_main", "all"):
        jobs.extend(train_main_commands(paths, seeds, args.episodes))
    if stage in ("train_ablation", "all"):
        jobs.extend(train_ablation_commands(paths, seeds, args.episodes))
    if stage in ("evaluate", "all"):
        jobs.extend(
            evaluate_commands(
                paths,
                seeds,
                include_main=True,
                ablation_ids=ABLATION_VARIANTS if args.include_ablation_eval else (),
                levels=LEVELS,
            )
        )
    if stage == "evaluate_extreme":
        # Zero-shot stress test of already-trained Medium checkpoints.
        jobs.extend(
            evaluate_commands(
                paths,
                seeds,
                include_main=True,
                ablation_ids=(),
                levels=STRESS_LEVELS,
            )
        )
    if stage in ("evaluate_ablation", "all"):
        # Ablation table on the training domain (always part of a full paper run).
        jobs.extend(
            evaluate_commands(
                paths,
                seeds,
                include_main=False,
                ablation_ids=ABLATION_VARIANTS,
                levels=(paths.train_level,),
            )
        )
    if stage in ("aggregate", "all"):
        jobs.extend(
            aggregate_commands(
                paths,
                seeds,
                include_main=True,
                ablation_ids=ABLATION_VARIANTS if args.include_ablation_eval else (),
                levels=LEVELS,
            )
        )
    if stage == "aggregate_extreme":
        jobs.extend(
            aggregate_commands(
                paths,
                seeds,
                include_main=True,
                ablation_ids=(),
                levels=ALL_EVAL_LEVELS,
            )
        )
    if stage in ("aggregate_ablation", "all"):
        jobs.extend(
            aggregate_commands(
                paths,
                seeds,
                include_main=False,
                ablation_ids=ABLATION_VARIANTS,
                levels=(paths.train_level,),
            )
        )
    if stage in ("qualitative", "all"):
        jobs.extend(qualitative_commands(paths, seed=seeds[0]))
    if stage == "plot_ablation":
        jobs.extend(plot_ablation_commands(paths))
    if stage == "same_scene":
        jobs.extend(
            same_scene_commands(
                paths,
                train_seed=seeds[0],
                scene_seed=int(getattr(args, "scene_seed", 7000017)),
                level=getattr(args, "scene_level", None) or paths.train_level,
            )
        )

    return {
        "train_level": args.train_level,
        "prefix": prefix,
        "seeds": list(seeds),
        "episodes": args.episodes,
        "paper_methods": ["APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3_OPT"],
        "ablations": {
            "A0": "MATD3-MLP scratch",
            "A1": "HGAT-MATD3 scratch",
            "A2": "DAgger+prior without BC/critic pretrain",
            "A3": "full method without critic pretrain",
            "A4": "full method without visibility reward",
            "A5": "HGAT-MATD3-Opt (main)",
        },
        "notes": [
            "MPC is an expert upper bound and demonstration teacher; do not require beating MPC.",
            "Learning methods train only on the chosen train_level; Nominal/Hard tests are transfer.",
            "Transition/demo/DAgger assets must be collected on the same train_level config.",
            "Formal tables need hierarchical bootstrap over training seeds and held-out scenes.",
        ],
        "paths": asdict(paths) | {
            "train_config": str(paths.train_config),
            "no_visibility_config": str(paths.no_visibility_config),
            "transitions": paths.transitions,
            "no_visibility_transitions": paths.no_visibility_transitions,
            "success_demos": paths.success_demos,
            "dagger_actor": paths.dagger_actor,
            "dagger_corrections": paths.dagger_corrections,
        },
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "assets", "baselines", "baselines_extreme", "train_main", "train_ablation",
            "evaluate", "evaluate_extreme", "evaluate_ablation",
            "aggregate", "aggregate_extreme", "aggregate_ablation",
            "plot_ablation", "qualitative", "same_scene", "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--train-level", choices=LEVELS, default="medium",
        help="Primary training domain. Default medium for non-saturated expert headroom.",
    )
    parser.add_argument("--prefix", default=None, help="Output name prefix; default radar5_<train-level>")
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument(
        "--scene-seed", type=int, default=7000017,
        help="Held-out scene seed for --stage same_scene",
    )
    parser.add_argument(
        "--scene-level", choices=ALL_EVAL_LEVELS, default=None,
        help="Environment profile for --stage same_scene (default: train-level)",
    )    parser.add_argument(
        "--include-ablation-eval", action="store_true",
        help="Also evaluate/aggregate A0–A4 across difficulties (expensive).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--manifest-out", default=None, help="Optional manifest JSON path")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    try:
        manifest = build_manifest(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    for job in manifest["jobs"]:
        print(subprocess.list2cmdline(job["command"]), flush=True)
        if not args.dry_run:
            subprocess.run(job["command"], cwd=ROOT, check=True)

    digest = hashlib.sha1(json.dumps(manifest["jobs"], sort_keys=True).encode("utf-8")).hexdigest()[:10]
    if args.manifest_out:
        from artifact_paths import artifact_path
        manifest_path = artifact_path(args.manifest_out)
    elif args.dry_run:
        manifest_path = ROOT / f"suite_manifest_{manifest['prefix']}_{args.stage}_{digest}.json"
    else:
        from artifact_paths import artifact_path
        manifest_path = artifact_path(
            f"outputs/{manifest['prefix']}_suite_manifest_{args.stage}_{digest}.json"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        print(f"Dry-run jobs: {len(manifest['jobs'])}")


if __name__ == "__main__":
    main()
