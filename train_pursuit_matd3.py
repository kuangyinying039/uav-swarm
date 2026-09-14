"""HGAT-MATD3 for quadrotor pursuit. MAPPO/DAgger remain in train_pursuit_with_demos.py."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from artifact_paths import SOURCE_ROOT, artifact_path, default_output
from marl_trainers import seed_everything
from pursuit.algorithms.matd3 import Matd3Config
from pursuit.data.transition_dataset import collect_teacher_transitions, json_sidecar, load_transition_dataset
from pursuit.eval import evaluate_methods
from pursuit.trainers.matd3_trainer import Matd3Trainer, mean_actor_state_from_checkpoint
from pursuit_baselines_3d import BASELINES_3D
from pursuit_graph_encoder import PURSUIT_GRAPH_VERSION
from pursuit_training_output import write_evaluation_chart, write_pursuit_outputs
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from train_pursuit_with_demos import (
    ensure_disjoint_seed_banks,
    load_environment_config,
    load_evaluation_environment_config,
    load_seed_bank,
    seed_for_episode,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect-transitions", "train", "evaluate"])
    parser.add_argument("--env-config", type=Path, default=SOURCE_ROOT / "configs/pursuit_v2/learning_start.json")
    parser.add_argument("--eval-env-config", type=Path,
                        help="Evaluate a checkpoint in a different, dimension-compatible environment")
    parser.add_argument("--out", type=artifact_path, default=default_output("pursuit_matd3_run"))
    parser.add_argument("--dataset", type=artifact_path, default=default_output("pursuit_transition_dataset_v2.pt"))
    parser.add_argument("--prior", type=artifact_path, help="v2 transition dataset used as prior replay")
    parser.add_argument("--actor-init", type=artifact_path, help="BC/MAPPO/MATD3 checkpoint; loads HGAT mean actor only")
    parser.add_argument("--checkpoint", type=artifact_path, help="Resume or evaluate a MATD3 run")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo-seed", type=int, default=2000000)
    parser.add_argument("--rollouts", type=int, default=200, help="Teacher episodes to store, including failures")
    parser.add_argument("--teacher", choices=sorted(BASELINES_3D), default="mpc")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.2)
    parser.add_argument("--exploration-std", type=float, default=0.12)
    parser.add_argument("--replay-size", type=int, default=500_000)
    parser.add_argument("--warmup-steps", type=int, default=8_000)
    parser.add_argument("--utd", type=int, default=1)
    parser.add_argument("--prior-fraction", type=float, default=None,
                        help="Offline share of each batch; default 0.5 if --prior is set, else 0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--validation-seed", type=int, default=4000000)
    parser.add_argument("--train-seeds-file", type=artifact_path)
    parser.add_argument("--validation-seeds-file", type=artifact_path)
    parser.add_argument("--eval-seed", type=int, default=7000000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--eval-seeds-file", type=artifact_path)
    parser.add_argument("--methods", nargs="+", default=["matd3"],
                        choices=["matd3", "apf", "frpn", "mpc"])
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser


def config_from_args(args, env_cfg, recorded=None):
    values = asdict(Matd3Config())
    if recorded:
        values.update({key: recorded[key] for key in values if key in recorded})
    values.update(
        gamma=env_cfg.reward_gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
        exploration_std=args.exploration_std,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        hidden_dim=args.hidden_dim,
        critic_hidden=args.critic_hidden,
        utd=args.utd,
        prior_fraction=args.prior_fraction,
        device=args.device,
        episodes=args.episodes,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
    )
    if recorded:
        # Network widths must match the checkpoint; keep those even if CLI differs.
        for key in ("hidden_dim", "critic_hidden", "gat_heads", "gat_layers", "critic_layers"):
            if key in recorded:
                values[key] = recorded[key]
    return Matd3Config(**values)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_env_config and args.mode != "evaluate":
        parser.error("--eval-env-config is only valid for evaluate")
    try:
        train_bank = load_seed_bank(args.train_seeds_file)
        validation_bank = load_seed_bank(args.validation_seeds_file)
        eval_bank = load_seed_bank(args.eval_seeds_file)
    except ValueError as error:
        parser.error(str(error))
    if min(args.episodes, args.batch_size, args.rollouts, args.eval_episodes, args.warmup_steps, args.replay_size) < 1:
        parser.error("counts must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    env_cfg = QuadrotorPursuitConfig(**json.loads(Path(args.env_config).read_text(encoding="utf-8")))
    env_cfg.building_state_capacity = max(env_cfg.building_state_capacity, env_cfg.building_count)

    if args.mode == "collect-transitions":
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset = collect_teacher_transitions(env_cfg, args.demo_seed, args.rollouts, args.teacher, args.dataset)
        torch.save(dataset, args.dataset)
        args.dataset.with_suffix(".json").write_text(json_sidecar(dataset), encoding="utf-8")
        print(json_sidecar(dataset))
        return

    recorded = {}
    actor_init_payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        env_cfg = load_environment_config(payload["env_config"])
        recorded = payload.get("train_config", {})
    elif args.actor_init:
        actor_init_payload = torch.load(args.actor_init, map_location="cpu", weights_only=False)
        if "env_config" in actor_init_payload:
            env_cfg = load_environment_config(actor_init_payload["env_config"])
        recorded = actor_init_payload.get("train_config", {})
    if args.prior_fraction is None:
        args.prior_fraction = float(recorded.get("prior_fraction", 0.5 if args.prior else 0.0))
    if not 0.0 <= args.prior_fraction <= 1.0:
        parser.error("--prior-fraction must be in [0, 1]")
    if args.prior_fraction > 0 and args.prior is None and args.mode == "train":
        parser.error("--prior-fraction > 0 requires --prior, including when resuming a checkpoint")
    cfg = config_from_args(args, env_cfg, recorded)
    factory = lambda episode=0: QuadrotorPursuitEnv(
        replace(env_cfg, seed=seed_for_episode(train_bank, episode, args.seed))
    )
    trainer = Matd3Trainer(factory, cfg, seed=args.seed)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    elif args.actor_init:
        init = actor_init_payload
        if init.get("pursuit_graph_version") not in (None, PURSUIT_GRAPH_VERSION):
            parser.error("Actor-init checkpoint uses a different pursuit graph version")
        trainer.load_actor_weights(mean_actor_state_from_checkpoint(init))
        trainer.demo_metadata["actor_init"] = str(args.actor_init)
        trainer.demo_metadata["actor_init_algorithm"] = init.get("algorithm")
    if args.prior:
        trainer.load_prior(load_transition_dataset(args.prior, env_cfg))

    if args.mode == "evaluate":
        if not args.checkpoint and not args.actor_init:
            parser.error("evaluate requires --checkpoint or --actor-init")
        seeds = eval_bank or list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        used = set(trainer.demo_metadata.get("training_seeds", []))
        if used.intersection(seeds):
            parser.error("Evaluation seeds overlap training")
        trainer.actor.eval()
        evaluation_cfg = (
            load_evaluation_environment_config(args.eval_env_config, env_cfg)
            if args.eval_env_config else env_cfg
        )
        result = evaluate_methods(
            lambda obs, env: trainer.deterministic_action(obs), evaluation_cfg, seeds, args.methods
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_evaluation_chart(args.out / "evaluation.svg", result)
        print(json.dumps(result["summary"], indent=2))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    trainer.output_directory = args.out
    trainer.train_seed_bank = list(train_bank)
    if args.validation_interval > 0 and args.validation_episodes > 0:
        trainer.validation_seeds = validation_bank or list(
            range(args.validation_seed, args.validation_seed + args.validation_episodes)
        )
        trainer.validation_seed_bank = list(trainer.validation_seeds) if validation_bank else []
        new_seeds = [
            seed_for_episode(train_bank, episode, args.seed)
            for episode in range(trainer.start_episode, cfg.episodes)
        ]
        trainer.demo_metadata["training_seeds"] = sorted(set(trainer.demo_metadata.get("training_seeds", []) + new_seeds))
        try:
            ensure_disjoint_seed_banks({
                "training": trainer.demo_metadata["training_seeds"],
                "validation": trainer.validation_seeds,
            })
        except ValueError as error:
            parser.error(str(error))
    rng = np.random.default_rng(args.seed)
    history = trainer.train(rng)
    write_pursuit_outputs(args.out, history, [])
    print(json.dumps({"episodes": len(history), "env_steps": trainer.env_steps, "updates": trainer.total_updates}, indent=2))


if __name__ == "__main__":
    main()
