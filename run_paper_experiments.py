"""Build reproducible multi-seed training commands for the paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).with_name("train_multi_seed_cooperative_marl.py")

VARIANTS = {
    "full": [],
    # Complete graph-encoder ablation: use a plain per-UAV MLP.
    "no_gat": ["--no-gat"],
    "no_heterogeneous_entities": ["--no-hetero-entities"],
    "no_revisit_map": ["--disable-revisit-map"],
    "no_freshness_fusion": ["--disable-freshness-fusion"],
    "hard_fusion": ["--belief-fusion-mode", "hard"],
    "no_adaptive_entropy": ["--disable-adaptive-entropy"],
    "linear_discovery_potential": ["--discovery-tail-exponent", "1.0"],
    "history_none": ["--history-mode", "none"],
    "history_edge_age": ["--history-mode", "edge_age"],
    "history_full": ["--history-mode", "full"],
    "no_communication": ["--comm-mode", "none"],
    "full_communication": ["--comm-mode", "full"],
    "no_safety_filter": ["--avoidance-mode", "none"],
}
MAPPO_ONLY_VARIANTS = {
    "hard_fusion",
    "no_adaptive_entropy",
    "linear_discovery_potential",
}


def supports_variant(algorithm: str, variant: str) -> bool:
    if variant == "full":
        return True
    if algorithm == "ippo":
        return False
    if variant in MAPPO_ONLY_VARIANTS:
        return algorithm == "mappo"
    return algorithm in {"mappo", "qmix"}


def command(args, algo: str, variant: str) -> list[str]:
    out = Path(args.out_dir) / algo / variant
    return [
        sys.executable,
        str(TRAIN),
        "--algo",
        algo,
        "--episodes",
        str(args.episodes),
        "--seeds",
        args.seeds,
        "--device",
        args.device,
        "--out-dir",
        str(out),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--log-interval",
        str(args.log_interval),
        "--completion-steps",
        str(args.completion_steps),
        "--reward-mode",
        args.reward_mode,
        "--discovery-potential-weight",
        str(args.discovery_potential_weight),
        "--exploration-potential-weight",
        str(args.exploration_potential_weight),
        "--safety-cost-weight",
        str(args.safety_cost_weight),
        *VARIANTS[variant],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["main", "ablations", "all"], default="all")
    parser.add_argument("--algos", default="mappo,ippo,qmix")
    parser.add_argument("--variants", default=None, help="Optional comma-separated ablation names.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", default="11,23,41,59,83")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "paper_experiments"))
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--completion-steps", type=int, default=0)
    parser.add_argument(
        "--reward-mode",
        choices=["team_potential", "legacy_shaped"],
        default="team_potential",
    )
    parser.add_argument("--discovery-potential-weight", type=float, default=1.0)
    parser.add_argument("--exploration-potential-weight", type=float, default=0.30)
    parser.add_argument("--safety-cost-weight", type=float, default=0.10)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    algorithms = [value.strip() for value in args.algos.split(",") if value.strip()]
    unknown_algorithms = sorted(set(algorithms) - {"mappo", "ippo", "qmix"})
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms: {', '.join(unknown_algorithms)}")
    if args.variants:
        variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    elif args.stage == "main":
        variants = ["full"]
    elif args.stage == "ablations":
        variants = [value for value in VARIANTS if value != "full"]
    else:
        variants = list(VARIANTS)
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(
            f"Unknown variants: {', '.join(unknown_variants)}. "
            f"Choices: {', '.join(VARIANTS)}"
        )

    manifest = {
        "algorithms": algorithms,
        "variants": variants,
        "episodes": args.episodes,
        "completion_steps": args.completion_steps,
        "reward": {
            "mode": args.reward_mode,
            "discovery_potential_weight": args.discovery_potential_weight,
            "exploration_potential_weight": args.exploration_potential_weight,
            "safety_cost_weight": args.safety_cost_weight,
        },
        "seeds": [int(value) for value in args.seeds.split(",") if value.strip()],
        "commands": [],
    }
    for algo in algorithms:
        for variant in variants:
            if not supports_variant(algo, variant):
                continue
            cmd = command(args, algo, variant)
            if args.resume_existing:
                cmd.append("--resume-existing")
            manifest["commands"].append({"algorithm": algo, "variant": variant, "command": cmd})
            print(subprocess.list2cmdline(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    descriptor = "__".join([*algorithms, *variants])
    digest = hashlib.sha1(descriptor.encode("utf-8")).hexdigest()[:10]
    run_name = f"{args.stage}_{digest}"
    manifest_path = out / f"experiment_manifest_{run_name}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
