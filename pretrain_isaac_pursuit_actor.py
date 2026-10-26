"""Build an Isaac-trained MPC-BC/DAgger actor for MAPPO and MATD3 warm starts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from marl_trainers import TrainConfig, seed_everything
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig
from train_pursuit_with_demos import (
    PursuitDemoTrainer,
    deterministic_action,
    imitation_score,
    pretrain,
    pretrain_mixed,
    snapshot,
    write_dagger_validation,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_medium.json"))
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--teacher", choices=("mpc", "frpn"), default="mpc")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo-seed", type=int, default=2_000_000)
    parser.add_argument("--bc-episodes", type=int, default=50,
                        help="Target number of BC demonstration episodes to keep")
    parser.add_argument("--bc-attempts", type=int, default=0,
                        help="Max teacher rollouts while collecting BC demos "
                             "(0 → 4× --bc-episodes when --success-only-bc)")
    parser.add_argument("--bc-updates", type=int, default=1500)
    parser.add_argument(
        "--success-only-bc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only successful teacher demos for BC (matches kinematic collect_demos)",
    )
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=40)
    parser.add_argument("--dagger-updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dagger-correction-fraction", type=float, default=0.5)
    parser.add_argument(
        "--dagger-validation-episodes",
        type=int,
        default=20,
        help="Held-out capture eval after BC and each DAgger round; 0 disables",
    )
    parser.add_argument(
        "--dagger-validation-seed",
        type=int,
        default=5_000_000,
        help="First seed of the DAgger held-out bank (disjoint from demo/correction)",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--native-lidar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Isaac PhysX range sensors (default: geometric LiDAR fallback)",
    )
    parser.add_argument(
        "--physx-closed-loop",
        action="store_true",
        help="DAgger under pure PhysX 3+1 closed loop (sensing defaults to geometric)",
    )
    return parser


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_dagger_progress(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[isaac-imitation] ignore unreadable progress {path}: {error}", flush=True)
        return {}
    if int(payload.get("version", 0)) not in (1, 2):
        print(f"[isaac-imitation] ignore progress with an unexpected version: {path}", flush=True)
        return {}
    return payload


def _evaluate_learner(env, trainer, cfg, seeds):
    """Capture-rate eval on the live Isaac env (same protocol as online validation)."""
    captures = 0
    steps_list = []
    returns = []
    trainer.actor.eval()
    for seed in seeds:
        observation = env.reset(seed=int(seed))
        total = 0.0
        steps = 0
        captured = False
        for _ in range(cfg.search_steps):
            action = deterministic_action(trainer, observation)
            transition = env.step_joint(action)
            observation = transition["obs"]
            reward = transition.get("reward")
            if reward is not None:
                total += float(np.asarray(reward, dtype=np.float64).mean())
            steps += 1
            if transition.get("terminated") or transition.get("truncated"):
                captured = bool(transition.get("capture_success", False))
                break
        captures += int(captured)
        steps_list.append(steps)
        returns.append(total)
    n = max(len(seeds), 1)
    return {
        "episodes": len(seeds),
        "capture_rate": captures / n,
        "mean_censored_steps": float(np.mean(steps_list)) if steps_list else float(cfg.search_steps),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
    }


def _maybe_select_best(trainer, env, cfg, seeds, stage, out, records, best_score, best_stage):
    if not seeds:
        trainer.save_checkpoint(out / "best_imitation.pt", episode=0, history=[])
        return best_score, best_stage or stage
    metrics = _evaluate_learner(env, trainer, cfg, seeds)
    records.append({"stage": stage, **metrics})
    score = imitation_score(metrics)
    print(
        f"[isaac-imitation-val] stage={stage} capture_rate={metrics['capture_rate']:.3f} "
        f"steps={metrics['mean_censored_steps']:.1f} return={metrics['mean_return']:.2f}",
        flush=True,
    )
    if best_score is None or score > best_score:
        best_score, best_stage = score, stage
        trainer.save_checkpoint(out / "best_imitation.pt", episode=0, history=[])
        print(f"[isaac-imitation-val] selected best_imitation stage={stage}", flush=True)
    write_dagger_validation(out, records, best_stage, seeds)
    return best_score, best_stage


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.bc_episodes, args.bc_updates, args.batch_size, args.hidden_dim) < 1:
        parser.error("BC episodes/updates, batch size and hidden dimension must be positive")
    if args.dagger_rounds < 0 or args.dagger_episodes < 1 or args.dagger_updates < 1:
        parser.error("DAgger rounds must be non-negative and episode/update counts positive")
    if not 0.0 <= args.dagger_correction_fraction <= 1.0:
        parser.error("--dagger-correction-fraction must be in [0, 1]")
    if args.dagger_validation_episodes < 0:
        parser.error("--dagger-validation-episodes must be >= 0")
    bc_attempts = int(args.bc_attempts)
    if bc_attempts <= 0:
        bc_attempts = args.bc_episodes * (4 if args.success_only_bc else 1)
    if bc_attempts < args.bc_episodes:
        parser.error("--bc-attempts must be >= --bc-episodes")

    manifest_path = args.out / "imitation_manifest.json"
    if (args.out / "best_imitation.pt").is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(manifest.get("bc_episodes", -1)) == args.bc_episodes
            and int(manifest.get("dagger_rounds", -1)) == args.dagger_rounds
            and int(manifest.get("dagger_episodes_per_round", -1)) == args.dagger_episodes
            and bool(manifest.get("success_only_bc", False)) == bool(args.success_only_bc)
            and int(manifest.get("dagger_validation_episodes", -1)) == args.dagger_validation_episodes
            and (args.out / "dagger_validation.json").is_file()
        ):
            print(f"[isaac-imitation] skip existing checkpoint={args.out / 'best_imitation.pt'}", flush=True)
            return

    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    if cfg.capture_required_uavs != 2 or cfg.execution_reward_version != 5:
        parser.error("MPC-BC/DAgger initialization must use dual-capture reward-v5")
    seed_everything(args.seed)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    physx_closed = bool(args.physx_closed_loop)
    env = IsaacPursuitEnv(
        cfg,
        seed=args.demo_seed,
        headless=args.headless,
        native_lidar=bool(args.native_lidar),
        physx_closed_loop=physx_closed,
    )
    try:
        train_cfg = TrainConfig(
            episodes=1,
            gamma=cfg.reward_gamma,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gat=True,
            use_hetero_entities=True,
            continuous_initial_std=0.20,
            continuous_log_std_max=float(np.log(0.40)),
        )

        def env_factory(episode=0):
            env.reset(seed=args.demo_seed + int(episode))
            return env

        trainer = PursuitDemoTrainer(env_factory, train_cfg)
        trainer.demo_metadata.update({
            "execution_backend": "isaac_physx_closed" if physx_closed else "isaac_physx",
            "physics_dt": float(env.physics_dt),
            "physx_closed_loop": physx_closed,
            "initialization_type": f"isaac_{args.teacher}_dagger",
            "teacher": args.teacher,
            "bc_seed": int(args.demo_seed),
            "success_only_bc": bool(args.success_only_bc),
        })
        progress_path = args.out / "dagger_progress.pt"
        actor_path = args.out / "dagger_latest.pt"
        progress = _load_dagger_progress(progress_path)
        expert_episodes = list(progress.get("expert_episodes") or [])
        expert_report = list(progress.get("expert_report") or [])
        bc_pretrained = bool(progress.get("bc_pretrained"))
        correction_episodes = list(progress.get("correction_episodes") or [])
        correction_report = list(progress.get("correction_report") or [])
        completed_rounds = int(progress.get("completed_dagger_rounds") or 0)
        open_round = progress.get("open_round")
        open_episodes = list(progress.get("open_round_episodes") or [])
        open_report = list(progress.get("open_round_report") or [])
        validation_records = list(progress.get("validation_records") or [])
        best_score = progress.get("best_score")
        if isinstance(best_score, list):
            best_score = tuple(best_score)
        best_stage = progress.get("best_stage")
        if expert_episodes or bc_pretrained or completed_rounds or open_episodes:
            print(
                f"[isaac-imitation] resume bc={len(expert_episodes)}/{args.bc_episodes} "
                f"bc_pretrained={bc_pretrained} rounds={completed_rounds}/{args.dagger_rounds} "
                f"open={0 if open_round is None else len(open_episodes)}",
                flush=True,
            )
        if bc_pretrained and actor_path.is_file():
            trainer.load_checkpoint(actor_path)

        validation_seeds = list(range(
            args.dagger_validation_seed,
            args.dagger_validation_seed + args.dagger_validation_episodes,
        ))
        trainer.demo_metadata["dagger_validation_seeds"] = validation_seeds

        def save_progress():
            _atomic_torch_save({
                "version": 2,
                "expert_episodes": expert_episodes,
                "expert_report": expert_report,
                "bc_pretrained": bc_pretrained,
                "correction_episodes": correction_episodes,
                "correction_report": correction_report,
                "completed_dagger_rounds": completed_rounds,
                "open_round": open_round,
                "open_round_episodes": open_episodes,
                "open_round_report": open_report,
                "validation_records": validation_records,
                "best_score": list(best_score) if best_score is not None else None,
                "best_stage": best_stage,
            }, progress_path)
            if bc_pretrained:
                trainer.save_checkpoint(actor_path, episode=0, history=[])

        expert = BASELINES_3D[args.teacher]()
        attempt = len(expert_report)
        while len(expert_episodes) < args.bc_episodes:
            if attempt >= bc_attempts:
                raise RuntimeError(
                    f"Only {len(expert_episodes)}/{args.bc_episodes} successful BC demos "
                    f"in {attempt} attempts; raise --bc-attempts or inspect teacher under closed loop"
                )
            seed = args.demo_seed + attempt
            attempt += 1
            observation = env.reset(seed=seed)
            expert.reset(env)
            trajectory = []
            captured = False
            for _ in range(cfg.search_steps):
                action = expert.actions(env)
                active = ~np.asarray(env.disabled_uavs, dtype=bool)
                trajectory.append(snapshot(observation, action, active))
                transition = env.step_joint(action)
                observation = transition["obs"]
                if transition["terminated"] or transition["truncated"]:
                    captured = bool(transition.get("capture_success", False))
                    break
            row = {
                "seed": seed,
                "success": captured,
                "steps": len(trajectory),
                "kept": bool(captured or not args.success_only_bc),
            }
            expert_report.append(row)
            if row["kept"]:
                expert_episodes.append(trajectory)
            save_progress()
            print(
                f"[isaac-{args.teacher}-bc] kept={len(expert_episodes)}/{args.bc_episodes} "
                f"attempt={attempt}/{bc_attempts} success={captured} steps={len(trajectory)}",
                flush=True,
            )
        trainer.demo_episodes = expert_episodes
        if not bc_pretrained:
            pretrain(trainer, expert_episodes, args.bc_updates, args.batch_size, args.seed)
            bc_pretrained = True
            save_progress()
            best_score, best_stage = _maybe_select_best(
                trainer, env, cfg, validation_seeds, "bc", args.out,
                validation_records, best_score, best_stage,
            )
            save_progress()

        for round_index in range(completed_rounds, args.dagger_rounds):
            beta = 0.5 * (1.0 - round_index / max(args.dagger_rounds - 1, 1))
            if open_round != round_index:
                open_round = round_index
                open_episodes = []
                open_report = []
                save_progress()
            trainer.actor.eval()
            while len(open_episodes) < args.dagger_episodes:
                episode = len(open_episodes)
                seed = args.demo_seed + 1_000_000 + round_index * args.dagger_episodes + episode
                rng = np.random.default_rng(seed)
                observation = env.reset(seed=seed)
                expert.reset(env)
                trajectory = []
                captured = False
                for _ in range(cfg.search_steps):
                    label = expert.actions(env)
                    active = ~np.asarray(env.disabled_uavs, dtype=bool)
                    trajectory.append(snapshot(observation, label, active))
                    learner_action = deterministic_action(trainer, observation)
                    action = label if rng.random() < beta else learner_action
                    transition = env.step_joint(action)
                    observation = transition["obs"]
                    if transition["terminated"] or transition["truncated"]:
                        captured = bool(transition.get("capture_success", False))
                        break
                open_episodes.append(trajectory)
                open_report.append({
                    "round": round_index + 1,
                    "seed": seed,
                    "teacher_probability": beta,
                    "mixed_rollout_capture": captured,
                    "steps": len(trajectory),
                })
                save_progress()
            correction_episodes.extend(open_episodes)
            correction_report.extend(open_report)
            open_round = None
            open_episodes = []
            open_report = []
            pretrain_mixed(
                trainer,
                expert_episodes,
                correction_episodes,
                args.dagger_updates,
                args.batch_size,
                args.dagger_correction_fraction,
                args.seed + round_index + 1,
            )
            completed_rounds = round_index + 1
            trainer.demo_metadata["dagger_round"] = completed_rounds
            trainer.demo_metadata["dagger_correction_seeds"] = [
                row["seed"] for row in correction_report
            ]
            stage = f"dagger_round_{completed_rounds:02d}"
            best_score, best_stage = _maybe_select_best(
                trainer, env, cfg, validation_seeds, stage, args.out,
                validation_records, best_score, best_stage,
            )
            save_progress()
            print(f"[isaac-dagger] round={completed_rounds}/{args.dagger_rounds} episodes={args.dagger_episodes}", flush=True)

        if best_stage is None:
            # Validation disabled: fall back to final weights.
            trainer.save_checkpoint(args.out / "best_imitation.pt", episode=0, history=[])
            best_stage = f"dagger_round_{completed_rounds:02d}" if completed_rounds else "bc"

        trainer.demo_metadata["demo_seeds"] = [row["seed"] for row in expert_report if row.get("kept")]
        trainer.demo_metadata["expert_report"] = expert_report
        trainer.demo_metadata["correction_report"] = correction_report
        trainer.demo_metadata["dagger_correction_fraction"] = args.dagger_correction_fraction
        trainer.demo_metadata["selected_imitation_stage"] = best_stage
        backend_tag = "isaac_physx_closed" if physx_closed else "isaac_physx"
        _atomic_torch_save({
            "execution_backend": backend_tag,
            "env_config": cfg.__dict__,
            "episodes": expert_episodes,
            "teacher": args.teacher,
            "expert_report": expert_report,
            "success_only_bc": bool(args.success_only_bc),
        }, args.out / "expert_episodes.pt")
        _atomic_torch_save({
            "execution_backend": backend_tag,
            "env_config": cfg.__dict__,
            "episodes": correction_episodes,
            "correction_report": correction_report,
            "dagger_round": args.dagger_rounds,
        }, args.out / "correction_episodes.pt")
        # Reload selected best into dagger_latest for inspectability.
        if (args.out / "best_imitation.pt").is_file():
            trainer.load_checkpoint(args.out / "best_imitation.pt")
            trainer.save_checkpoint(actor_path, episode=0, history=[])
        bc_successes = sum(1 for row in expert_report if row.get("success"))
        corr_successes = sum(1 for row in correction_report if row.get("mixed_rollout_capture"))
        (args.out / "imitation_manifest.json").write_text(json.dumps({
            "execution_backend": backend_tag,
            "environment": cfg.__dict__,
            "teacher": args.teacher,
            "bc_episodes": args.bc_episodes,
            "bc_attempts": bc_attempts,
            "bc_updates": args.bc_updates,
            "success_only_bc": bool(args.success_only_bc),
            "bc_success_rate": bc_successes / max(len(expert_report), 1),
            "bc_attempts_used": len(expert_report),
            "dagger_rounds": args.dagger_rounds,
            "dagger_episodes_per_round": args.dagger_episodes,
            "dagger_updates_per_round": args.dagger_updates,
            "dagger_validation_episodes": args.dagger_validation_episodes,
            "dagger_validation_seed": args.dagger_validation_seed,
            "selected_imitation_stage": best_stage,
            "best_imitation_score": list(best_score) if best_score is not None else None,
            "expert_report": expert_report,
            "correction_report": correction_report,
            "correction_mixed_capture_rate": corr_successes / max(len(correction_report), 1),
            "checkpoint": str(args.out / "best_imitation.pt"),
        }, indent=2), encoding="utf-8")
        print(
            f"[isaac-imitation] checkpoint={args.out / 'best_imitation.pt'} "
            f"stage={best_stage} bc_success_rate={bc_successes / max(len(expert_report), 1):.3f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
