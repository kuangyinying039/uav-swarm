"""Run all post-training evaluation and paper-figure commands automatically."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_paper_experiments import VARIANTS, supports_variant


ROOT = Path(__file__).resolve().parents[1]
REPRO = Path(__file__).resolve().parent
EVALUATE = REPRO / "evaluate_policy_suite.py"
SENSITIVITY = REPRO / "evaluate_checkpoint_sensitivity.py"
PLOT = REPRO / "plot_paper_results.py"
SENSITIVITY_SWEEPS = (
    "target_speed",
    "obstacle_speed",
    "comm_radius",
    "sensor_radius",
    "n_obstacles",
    "outage_base_prob",
    "outage_jammed_prob",
    "outage_recovery_prob",
)


def checkpoints(experiment_root: Path, algorithm: str, variant: str, seeds: list[int]) -> list[Path]:
    paths = [
        experiment_root / algorithm / variant / f"{algorithm}_seed_{seed}.pt"
        for seed in seeds
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing final checkpoints for {algorithm}/{variant}:\n{formatted}\n"
            "Wait for training to finish or use --dry-run to inspect the commands."
        )
    return paths


def run(command: list[str], *, dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def policy_arguments(label: str, paths: list[Path]) -> list[str]:
    result = []
    for path in paths:
        result.extend(["--policy", f"{label}={path}"])
    return result


def evaluate_main(args, seeds: list[int], scenario_seeds: str) -> None:
    out = Path(args.analysis_root) / "main"
    command = [
        sys.executable,
        str(EVALUATE),
        *policy_arguments("MAPPO", checkpoints(Path(args.experiment_root), "mappo", "full", seeds)),
        *policy_arguments("IPPO", checkpoints(Path(args.experiment_root), "ippo", "full", seeds)),
        *policy_arguments("QMIX", checkpoints(Path(args.experiment_root), "qmix", "full", seeds)),
        "--seeds",
        scenario_seeds,
        "--out-dir",
        str(out),
    ]
    run(command, dry_run=args.dry_run)
    run(
        [
            sys.executable,
            str(PLOT),
            "--mode",
            "comparison",
            "--input",
            str(out / "policy_suite_summary.csv"),
            "--output",
            str(Path(args.figure_root) / "main_comparison.svg"),
            "--title",
            "Performance Comparison under Weak Communication",
        ],
        dry_run=args.dry_run,
    )


def evaluate_ablations(args, seeds: list[int], scenario_seeds: str) -> None:
    for algorithm in ("mappo", "qmix"):
        full = checkpoints(Path(args.experiment_root), algorithm, "full", seeds)
        display_algorithm = algorithm.upper()
        for variant in (name for name in VARIANTS if name != "full"):
            if not supports_variant(algorithm, variant):
                continue
            ablated = checkpoints(Path(args.experiment_root), algorithm, variant, seeds)
            out = Path(args.analysis_root) / "ablations" / algorithm / variant
            label = variant.replace("_", " ").title()
            run(
                [
                    sys.executable,
                    str(EVALUATE),
                    *policy_arguments(f"{display_algorithm}-Full", full),
                    *policy_arguments(f"{display_algorithm}-{label}", ablated),
                    "--seeds",
                    scenario_seeds,
                    "--out-dir",
                    str(out),
                ],
                dry_run=args.dry_run,
            )
            run(
                [
                    sys.executable,
                    str(PLOT),
                    "--mode",
                    "ablation",
                    "--input",
                    str(out / "policy_suite_summary.csv"),
                    "--output",
                    str(Path(args.figure_root) / f"{algorithm}_{variant}.svg"),
                    "--title",
                    f"{display_algorithm} Ablation: {label}",
                ],
                dry_run=args.dry_run,
            )


def evaluate_sensitivity(args, seeds: list[int], scenario_seeds: str) -> None:
    representative = args.representative_seed
    if representative not in seeds:
        raise ValueError("--representative-seed must be included in --training-seeds")
    mappo = checkpoints(Path(args.experiment_root), "mappo", "full", [representative])[0]
    qmix = checkpoints(Path(args.experiment_root), "qmix", "full", [representative])[0]
    out = Path(args.analysis_root) / "sensitivity"
    sweep_names = [name.strip() for name in args.sweeps.split(",") if name.strip()]
    unknown = sorted(set(sweep_names) - set(SENSITIVITY_SWEEPS))
    if unknown:
        raise ValueError(f"Unknown sensitivity sweeps: {', '.join(unknown)}")
    command = [
        sys.executable,
        str(SENSITIVITY),
        "--mappo",
        str(mappo),
        "--qmix",
        str(qmix),
        "--seeds",
        scenario_seeds,
        "--sweeps",
        ",".join(sweep_names),
        "--out-dir",
        str(out),
    ]
    if args.quick:
        command.append("--quick")
    run(command, dry_run=args.dry_run)
    for parameter in sweep_names:
        run(
            [
                sys.executable,
                str(PLOT),
                "--mode",
                "sensitivity",
                "--input",
                str(out / "checkpoint_sensitivity_summary.csv"),
                "--parameter",
                parameter,
                "--output",
                str(Path(args.figure_root) / f"sensitivity_{parameter}.svg"),
                "--title",
                f"Sensitivity to {parameter.replace('_', ' ').title()}",
            ],
            dry_run=args.dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["main", "ablations", "sensitivity", "all"], default="all")
    parser.add_argument("--training-seeds", default="11,23,41,59,83")
    parser.add_argument("--scenario-seeds", default="101,113,127,139,151,163,179,191,211,227")
    parser.add_argument("--representative-seed", type=int, default=59)
    parser.add_argument("--sweeps", default=",".join(SENSITIVITY_SWEEPS))
    parser.add_argument("--experiment-root", default=str(ROOT / "outputs" / "paper_experiments"))
    parser.add_argument("--analysis-root", default=str(ROOT / "outputs" / "paper_evaluation"))
    parser.add_argument("--figure-root", default=str(ROOT / "outputs" / "paper_figures"))
    parser.add_argument("--quick", action="store_true", help="Use the sensitivity evaluator's smoke-test mode.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.training_seeds.split(",") if value.strip()]
    if args.dry_run:
        # Command inspection should remain possible before checkpoints exist.
        original = globals()["checkpoints"]

        def unchecked(root, algorithm, variant, selected_seeds):
            return [
                root / algorithm / variant / f"{algorithm}_seed_{seed}.pt"
                for seed in selected_seeds
            ]

        globals()["checkpoints"] = unchecked
    try:
        if args.stage in {"main", "all"}:
            evaluate_main(args, seeds, args.scenario_seeds)
        if args.stage in {"ablations", "all"}:
            evaluate_ablations(args, seeds, args.scenario_seeds)
        if args.stage in {"sensitivity", "all"}:
            evaluate_sensitivity(args, seeds, args.scenario_seeds)
    finally:
        if args.dry_run:
            globals()["checkpoints"] = original


if __name__ == "__main__":
    main()
