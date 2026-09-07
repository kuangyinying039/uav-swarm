"""Run MAPPO/QMIX training over multiple random seeds and report mean/std."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from marl_trainers import (
    IPPOTrainer,
    MAPPOTrainer,
    QMIXTrainer,
    TrainConfig,
    seed_everything,
)
from train_cooperative_marl import metric_points, moving_average, write_training_scene_svgs, write_training_svg
from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig


ROOT = Path(__file__).resolve().parents[1]


def make_env(seed: int, env_overrides: dict, task_mode: str):
    def factory(episode: int = 0):
        mode_classes = {
            "search_multi_target": (WeakCommConfig, WeakCommBeliefGraphEnv),
        }
        config_cls, env_cls = mode_classes[task_mode]
        cfg = config_cls(seed=seed * 100_000 + int(episode))
        for key, value in env_overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return env_cls(cfg)

    return factory


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def aggregate(histories: list[list[dict]]) -> list[dict]:
    min_len = min(len(h) for h in histories)
    metrics = [
        "reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "belief_decisiveness",
        "belief_brier_score",
        "task_success",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "pair_conflicts",
        "collision_rate",
        "outage_rate",
        "disabled_rate",
        "safety_intervention_rate",
        "new_cells",
        "revisit_gain",
        "stagnant_agent_rate",
        "new_crashes",
        "capture_success",
        "first_onboard_contact_time",
        "first_detection_time",
        "capture_time",
        "target_loss_count",
        "target_reacquisition_count",
        "mean_reacquisition_time",
        "continuous_visibility_ratio",
        "track_position_rmse",
        "track_velocity_rmse",
        "mean_track_nees",
        "track_nees_consistency_rate",
        "mean_track_age",
        "valid_track_rate",
        "track_messages",
        "track_communication_bytes",
        "assignment_search_rate",
        "assignment_switch_rate",
        "collision_exposure_rate",
        "mpc_feasible_rate",
        "path_length",
        "control_smoothness_cost",
        "weight_target",
        "weight_exploration",
        "weight_revisit",
        "weight_connectivity",
        "weight_safety",
    ]
    rows = []
    for idx in range(min_len):
        row = {"episode": histories[0][idx]["episode"]}
        for metric in metrics:
            values = np.asarray([h[idx].get(metric, 0.0) for h in histories], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return rows


def aggregate_tail(histories: list[list[dict]], window: int = 50) -> dict:
    metrics = [
        "reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "belief_decisiveness",
        "belief_brier_score",
        "task_success",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "pair_conflicts",
        "collision_rate",
        "outage_rate",
        "disabled_rate",
        "safety_intervention_rate",
        "new_cells",
        "revisit_gain",
        "stagnant_agent_rate",
        "new_crashes",
        "capture_success",
        "first_onboard_contact_time",
        "first_detection_time",
        "capture_time",
        "target_loss_count",
        "target_reacquisition_count",
        "mean_reacquisition_time",
        "continuous_visibility_ratio",
        "track_position_rmse",
        "track_velocity_rmse",
        "mean_track_nees",
        "track_nees_consistency_rate",
        "mean_track_age",
        "valid_track_rate",
        "track_messages",
        "track_communication_bytes",
        "assignment_search_rate",
        "assignment_switch_rate",
        "collision_exposure_rate",
        "mpc_feasible_rate",
        "path_length",
        "control_smoothness_cost",
        "weight_target",
        "weight_exploration",
        "weight_revisit",
        "weight_connectivity",
        "weight_safety",
    ]
    # Keep this aggregation independent from train_cooperative_marl.tail_average:
    # the multi-seed report contains additional metrics and may resume histories
    # written by an older version that did not record every current metric.
    tail_rows = [history[-min(window, len(history)) :] for history in histories]
    summary = {
        "episodes_averaged_per_seed": min(window, min(len(history) for history in histories)),
        "n_seeds": len(histories),
    }
    for metric in metrics:
        values = np.asarray(
            [
                np.mean([float(row.get(metric, 0.0)) for row in rows])
                for rows in tail_rows
            ],
            dtype=float,
        )
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def write_mean_std_svg(path: Path, algo: str, rows: list[dict]) -> None:
    width, height = 1000, 540
    left, top, right, bottom = 85, 60, 245, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    metrics = [
        ("coverage", "#1f77b4", "Coverage mean"),
        ("target_discovery_rate", "#2ca02c", "Discovery mean"),
        ("target_tracking_rate", "#9467bd", "Tracking mean"),
        ("target_completion_rate", "#111111", "Completed mean"),
    ]
    max_ep = max(row["episode"] for row in rows)
    values = [row[f"{metric}_mean"] + row[f"{metric}_std"] for row in rows for metric, _, _ in metrics]
    min_v, max_v = 0.0, max(values + [1.0])
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{left}" y="26" font-size="22" font-family="Times New Roman">{algo.upper()} Multi-seed Mean/Std</text>')
    body.append(f'<text x="{left}" y="46" font-size="16" font-family="Times New Roman" fill="#555">Bold line: smoothed mean; thin line: raw mean; translucent band: one standard deviation.</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = top + tick * plot_height / 5
        v = max_v - tick * (max_v - min_v) / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="24" y="{y+4:.1f}" font-size="15" font-family="Times New Roman">{v:.2f}</text>')
    for tick in range(6):
        x = left + tick * plot_width / 5
        episode = tick * max_ep / 5
        body.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+24}" font-size="15" font-family="Times New Roman" text-anchor="middle">{episode:.0f}</text>')
    for metric, color, label in metrics:
        upper_pts = []
        lower_pts = []
        mean_values = [float(row[f"{metric}_mean"]) for row in rows]
        smooth_values = moving_average(mean_values)
        for row in rows:
            x = left + row["episode"] / max_ep * (width - left - right)
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            y_upper = height - bottom - (min(max_v, mean + std) - min_v) / (max_v - min_v) * (height - top - bottom)
            y_lower = height - bottom - (max(min_v, mean - std) - min_v) / (max_v - min_v) * (height - top - bottom)
            upper_pts.append((x, y_upper))
            lower_pts.append((x, y_lower))
        band = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper_pts + list(reversed(lower_pts)))
        raw_pts = metric_points(rows, f"{metric}_mean", max_ep, left, top, plot_width, plot_height, min_v, max_v, values=mean_values)
        trend_pts = metric_points(rows, f"{metric}_mean", max_ep, left, top, plot_width, plot_height, min_v, max_v, values=smooth_values)
        body.append(f'<polygon points="{band}" fill="{color}" opacity="0.12"/>')
        body.append(f'<polyline points="{raw_pts}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.28"/>')
        body.append(f'<polyline points="{trend_pts}" fill="none" stroke="{color}" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round"/>')
        lx = width - right + 30
        ly = top + 25 + metrics.index((metric, color, label)) * 30
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="3.4" stroke-linecap="round"/>')
        body.append(f'<text x="{lx+38}" y="{ly+4}" font-size="16" font-family="Times New Roman">{label}</text>')
    body.append(f'<text x="{width-right+30}" y="{top}" font-size="17" font-family="Times New Roman" font-weight="bold">Legend</text>')
    body.append(f'<text x="{left + (width-left-right)/2:.1f}" y="{height-18}" font-size="16" font-family="Times New Roman" text-anchor="middle">Episode</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix"], default="mappo")
    parser.add_argument(
        "--task-mode",
        choices=["search_multi_target"],
        default="search_multi_target",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", default="11,23,41,59,83")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-existing", action="store_true", help="Resume each MAPPO/QMIX seed from its *_latest.pt checkpoint when present.")
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--no-gat",
        action="store_true",
        help="Ablation: replace the complete graph/heterogeneous encoder with a plain per-UAV MLP.",
    )
    parser.add_argument("--no-hetero-entities", action="store_true")
    assignment_group = parser.add_mutually_exclusive_group()
    assignment_group.add_argument("--assignment-head", dest="assignment_head", action="store_true")
    assignment_group.add_argument("--no-assignment-head", dest="assignment_head", action="store_false")
    parser.set_defaults(assignment_head=None)
    parser.add_argument("--search-weights", action="store_true", help="Legacy ablation using adaptive J1..J5 search weights instead of direct categorical actions.")
    parser.add_argument("--direct-actions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end-factor", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.002)
    parser.add_argument("--disable-adaptive-entropy", action="store_true")
    parser.add_argument("--uncertainty-entropy-gain", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["team_potential", "legacy_shaped"], default=None)
    parser.add_argument("--discovery-potential-weight", type=float, default=None)
    parser.add_argument("--discovery-tail-exponent", type=float, default=None)
    parser.add_argument("--exploration-potential-weight", type=float, default=None)
    parser.add_argument("--safety-cost-weight", type=float, default=None)
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--tracking-grace-steps", type=int, default=None)
    parser.add_argument("--tracking-streak-decay", type=int, default=None)
    parser.add_argument("--tracking-streak-reward", type=float, default=None)
    parser.add_argument("--tracking-streak-loss-penalty", type=float, default=None)
    parser.add_argument("--persistent-tracking-reward", type=float, default=None)
    parser.add_argument("--tracking-stage-threshold", type=float, default=None)
    parser.add_argument("--completion-reward", type=float, default=None)
    parser.add_argument("--target-lost-penalty", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--obstacle-speed", type=float, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--decision-dt", type=float, default=None)
    parser.add_argument("--max-turn-steps", type=int, default=None, help="Maximum heading change per step in 45-degree units.")
    parser.add_argument("--allow-hover", action="store_true", help="Allow UAVs to choose the hover action even when movement is feasible.")
    parser.add_argument("--no-normalize-diagonal-speed", action="store_true", help="Keep diagonal grid moves longer than straight moves.")
    parser.add_argument("--comm-radius", type=float, default=None)
    parser.add_argument("--comm-mode", choices=["full", "outage", "none"], default=None)
    parser.add_argument("--avoidance-mode", choices=["none", "dwa", "dwa_orca"], default=None)
    parser.add_argument("--outage-base-prob", type=float, default=None)
    parser.add_argument("--outage-jammed-prob", type=float, default=None)
    parser.add_argument("--outage-recovery-prob", type=float, default=None)
    parser.add_argument("--failure-base-prob", type=float, default=None)
    parser.add_argument("--failure-jammed-prob", type=float, default=None)
    parser.add_argument("--failure-recovery-prob", type=float, default=None)
    parser.add_argument("--jammer-effect-radius", type=float, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    parser.add_argument("--disable-dynamic-dt", action="store_true", help="Use the fixed decision_dt from the config.")
    parser.add_argument("--min-decision-dt", type=float, default=None)
    parser.add_argument("--max-decision-dt", type=float, default=None)
    parser.add_argument("--avoidance-prediction-horizon", type=int, default=None)
    parser.add_argument("--obstacle-prediction-buffer", type=float, default=None)
    parser.add_argument("--pair-prediction-buffer", type=float, default=None)
    parser.add_argument("--hetero-target-nodes", type=int, default=None)
    parser.add_argument("--hetero-obstacle-nodes", type=int, default=None)
    parser.add_argument("--entity-observation-radius", type=float, default=None)
    parser.add_argument("--revisit-aging-rate", type=float, default=None)
    parser.add_argument("--revisit-diffusion", type=float, default=None)
    parser.add_argument("--revisit-gain-reward", type=float, default=None)
    parser.add_argument("--disable-revisit-map", action="store_true", help="Ablation: remove the third-layer revisit map and its policy objective.")
    parser.add_argument("--disable-revisit-reward", action="store_true", help="Ablation: retain the revisit map but remove its direct reward.")
    parser.add_argument("--disable-freshness-fusion", action="store_true", help="Ablation: ignore belief age and link confidence during map fusion.")
    parser.add_argument("--belief-fusion-mode", choices=["soft", "hard"], default=None)
    parser.add_argument("--belief-fusion-temperature", type=float, default=None)
    parser.add_argument("--belief-freshness-tau", type=float, default=None)
    parser.add_argument(
        "--history-mode",
        choices=["none", "edge_age", "full"],
        default=None,
        help="Communication-history ablation: none, edge_age, or full.",
    )
    parser.add_argument("--static-targets", action="store_true")
    parser.add_argument("--static-obstacles", action="store_true")
    parser.add_argument("--evader-policy", choices=["random", "repulsive", "occlusion"], default=None)
    parser.add_argument("--field-of-view-deg", type=float, default=None)
    parser.add_argument("--building-count", type=int, default=None)
    parser.add_argument("--capture-radius", type=float, default=None)
    parser.add_argument("--capture-required-uavs", type=int, default=None)
    parser.add_argument("--capture-hold-steps", type=int, default=None)
    parser.add_argument("--capture-angular-span-deg", type=float, default=None)
    parser.add_argument("--max-tracks-per-message", type=int, default=None)
    parser.add_argument("--communication-cost-per-kb", type=float, default=None)
    parser.add_argument("--track-freshness-tau", type=float, default=None)
    parser.add_argument("--disable-track-freshness-fusion", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / f"{args.algo}_multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_classes = {
        "search_multi_target": WeakCommConfig,
    }
    base_env_cfg = config_classes[args.task_mode]()
    assignment_head_enabled = False if args.assignment_head is None else args.assignment_head
    if args.algo == "qmix" and assignment_head_enabled:
        parser.error("Use --no-assignment-head for QMIX.")
    cfg = TrainConfig(
        episodes=args.episodes,
        max_steps=0,
        use_gat=not args.no_gat,
        use_hetero_entities=not args.no_gat and not args.no_hetero_entities,
        use_search_weights=args.search_weights and not args.direct_actions,
        use_assignment_head=assignment_head_enabled,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        lr_end_factor=args.lr_end_factor,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        adaptive_entropy=not args.disable_adaptive_entropy,
        uncertainty_entropy_gain=args.uncertainty_entropy_gain,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        reward_scale=args.reward_scale,
        normalize_returns=args.normalize_returns,
        normalize_advantages=not args.no_normalize_advantages,
        device=args.device,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        target_kl=args.target_kl,
    )
    env_overrides = {
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_obstacles": args.n_obstacles,
        "grid_size": args.grid_size,
        "search_steps": args.search_steps,
        "target_completion_steps": args.completion_steps,
        "reward_mode": args.reward_mode,
        "discovery_potential_weight": args.discovery_potential_weight,
        "discovery_tail_exponent": args.discovery_tail_exponent,
        "exploration_potential_weight": args.exploration_potential_weight,
        "safety_cost_weight": args.safety_cost_weight,
        "tracking_grace_steps": args.tracking_grace_steps,
        "tracking_streak_decay": args.tracking_streak_decay,
        "tracking_streak_reward": args.tracking_streak_reward,
        "tracking_streak_loss_penalty": args.tracking_streak_loss_penalty,
        "persistent_tracking_reward": args.persistent_tracking_reward,
        "tracking_stage_threshold": args.tracking_stage_threshold,
        "completion_reward": args.completion_reward,
        "target_lost_penalty": args.target_lost_penalty,
        "target_speed": args.target_speed,
        "obstacle_speed": args.obstacle_speed,
        "uav_speed": args.uav_speed,
        "decision_dt": args.decision_dt,
        "max_turn_steps": args.max_turn_steps,
        "allow_hover": True if args.allow_hover else None,
        "normalize_diagonal_speed": False if args.no_normalize_diagonal_speed else None,
        "comm_radius": args.comm_radius,
        "comm_mode": args.comm_mode,
        "avoidance_mode": args.avoidance_mode,
        "outage_base_prob": args.outage_base_prob,
        "outage_jammed_prob": args.outage_jammed_prob,
        "outage_recovery_prob": args.outage_recovery_prob,
        "failure_base_prob": args.failure_base_prob,
        "failure_jammed_prob": args.failure_jammed_prob,
        "failure_recovery_prob": args.failure_recovery_prob,
        "jammer_effect_radius": args.jammer_effect_radius,
        "tpm_prior_base": args.tpm_prior_base,
        "tpm_target_prior_enabled": True if args.enable_target_prior else None,
        "tpm_target_prior_strength": args.tpm_target_prior_strength,
        "tpm_target_prior_sigma": args.tpm_target_prior_sigma,
        "dynamic_dt_enabled": False if args.disable_dynamic_dt else None,
        "min_decision_dt": args.min_decision_dt,
        "max_decision_dt": args.max_decision_dt,
        "avoidance_prediction_horizon": args.avoidance_prediction_horizon,
        "obstacle_prediction_buffer": args.obstacle_prediction_buffer,
        "pair_prediction_buffer": args.pair_prediction_buffer,
        "hetero_target_nodes": args.hetero_target_nodes,
        "hetero_obstacle_nodes": args.hetero_obstacle_nodes,
        "entity_observation_radius": args.entity_observation_radius,
        "revisit_aging_rate": args.revisit_aging_rate,
        "revisit_diffusion": args.revisit_diffusion,
        "revisit_gain_reward": 0.0 if args.disable_revisit_reward else args.revisit_gain_reward,
        "revisit_map_enabled": False if args.disable_revisit_map else None,
        "freshness_fusion_enabled": False if args.disable_freshness_fusion else None,
        "belief_fusion_mode": args.belief_fusion_mode,
        "belief_fusion_temperature": args.belief_fusion_temperature,
        "belief_freshness_tau": args.belief_freshness_tau,
        "history_mode": args.history_mode,
        "dynamic_targets_enabled": False if args.static_targets else None,
        "dynamic_obstacles_enabled": False if args.static_obstacles else None,
        "evader_policy": args.evader_policy,
        "field_of_view_deg": args.field_of_view_deg,
        "building_count": args.building_count,
        "capture_radius": args.capture_radius,
        "capture_required_uavs": args.capture_required_uavs,
        "capture_hold_steps": args.capture_hold_steps,
        "capture_angular_span_deg": args.capture_angular_span_deg,
        "max_tracks_per_message": args.max_tracks_per_message,
        "communication_cost_per_kb": args.communication_cost_per_kb,
        "track_freshness_tau": args.track_freshness_tau,
        "track_freshness_fusion_enabled": False
        if args.disable_track_freshness_fusion
        else None,
    }
    effective_env_cfg = config_classes[args.task_mode](seed=seeds[0] if seeds else 0)
    for key, value in env_overrides.items():
        if value is not None:
            setattr(effective_env_cfg, key, value)
    print(
        json.dumps(
            {
                "effective_env_config": {
                    "n_uavs": effective_env_cfg.n_uavs,
                    "n_targets": effective_env_cfg.n_targets,
                    "grid_size": effective_env_cfg.grid_size,
                    "search_steps": effective_env_cfg.search_steps,
                    "n_obstacles": effective_env_cfg.n_obstacles,
                    "allow_hover": effective_env_cfg.allow_hover,
                    "avoidance_prediction_horizon": effective_env_cfg.avoidance_prediction_horizon,
                    "obstacle_prediction_buffer": effective_env_cfg.obstacle_prediction_buffer,
                    "pair_prediction_buffer": effective_env_cfg.pair_prediction_buffer,
                    "task_mode": args.task_mode,
                    "assignment_head": assignment_head_enabled,
                },
                "seeds": seeds,
            },
            indent=2,
        )
    )
    trainer_cls = {
        "mappo": MAPPOTrainer,
        "ippo": IPPOTrainer,
        "qmix": QMIXTrainer,
    }[args.algo]
    histories = []
    for seed in seeds:
        seed_everything(seed)
        latest_checkpoint = out_dir / f"{args.algo}_seed_{seed}_latest.pt"
        cfg.checkpoint_path = str(latest_checkpoint)
        trainer = trainer_cls(make_env(seed, env_overrides, args.task_mode), cfg)
        if args.resume_existing and latest_checkpoint.exists():
            trainer.load_checkpoint(latest_checkpoint)
            print(f"Resumed seed {seed} at episode {trainer.start_episode}.", flush=True)
        history = trainer.train()
        histories.append(history)
        seed_json = out_dir / f"{args.algo}_seed_{seed}.json"
        seed_json.write_text(json.dumps({"algo": args.algo, "seed": seed, "history": history}, indent=2), encoding="utf-8")
        seed_env_cfg = WeakCommConfig(seed=seed)
        for key, value in env_overrides.items():
            if value is not None:
                setattr(seed_env_cfg, key, value)
        trainer.save_checkpoint(seed_json.with_suffix(".pt"), env_config=vars(seed_env_cfg), episode=args.episodes, history=history)
        write_csv(
            seed_json.with_suffix(".csv"),
            history,
            [
                "episode",
                "reward",
                "spectrum_reward",
                "spectrum_success_rate",
                "coverage",
                "target_discovery_rate",
                "target_tracking_rate",
                "target_completion_rate",
                "belief_decisiveness",
                "belief_brier_score",
                "task_success",
                "collisions",
                "uav_collisions",
                "obstacle_conflicts",
                "pair_conflicts",
                "collision_rate",
                "outage_rate",
                "disabled_rate",
                "safety_intervention_rate",
                "new_cells",
                "revisit_gain",
                "stagnant_agent_rate",
                "new_crashes",
                "weight_target",
                "weight_exploration",
                "weight_revisit",
                "weight_connectivity",
                "weight_safety",
                "steps",
            ],
        )
        write_training_svg(seed_json.with_suffix(".svg"), f"{args.algo}_seed_{seed}", history)
        write_training_scene_svgs(out_dir / f"{args.algo}_seed_{seed}_last3_scenes", f"{args.algo}_seed_{seed}", getattr(trainer, "episode_traces", []))
    rows = aggregate(histories)
    write_csv(
        out_dir / f"{args.algo}_mean_std.csv",
        rows,
        [
            "episode",
            "reward_mean",
            "reward_std",
            "spectrum_reward_mean",
            "spectrum_reward_std",
            "spectrum_success_rate_mean",
            "spectrum_success_rate_std",
            "coverage_mean",
            "coverage_std",
            "target_discovery_rate_mean",
            "target_discovery_rate_std",
            "target_tracking_rate_mean",
            "target_tracking_rate_std",
            "target_completion_rate_mean",
            "target_completion_rate_std",
            "belief_decisiveness_mean",
            "belief_decisiveness_std",
            "belief_brier_score_mean",
            "belief_brier_score_std",
            "task_success_mean",
            "task_success_std",
            "collisions_mean",
            "collisions_std",
            "uav_collisions_mean",
            "uav_collisions_std",
            "obstacle_conflicts_mean",
            "obstacle_conflicts_std",
            "pair_conflicts_mean",
            "pair_conflicts_std",
            "collision_rate_mean",
            "collision_rate_std",
            "outage_rate_mean",
            "outage_rate_std",
            "disabled_rate_mean",
            "disabled_rate_std",
            "safety_intervention_rate_mean",
            "safety_intervention_rate_std",
            "new_cells_mean",
            "new_cells_std",
            "revisit_gain_mean",
            "revisit_gain_std",
            "stagnant_agent_rate_mean",
            "stagnant_agent_rate_std",
            "new_crashes_mean",
            "new_crashes_std",
            "weight_target_mean",
            "weight_target_std",
            "weight_exploration_mean",
            "weight_exploration_std",
            "weight_revisit_mean",
            "weight_revisit_std",
            "weight_connectivity_mean",
            "weight_connectivity_std",
            "weight_safety_mean",
            "weight_safety_std",
        ],
    )
    (out_dir / f"{args.algo}_mean_std.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_mean_std_svg(out_dir / f"{args.algo}_mean_std.svg", args.algo, rows)
    print(json.dumps(aggregate_tail(histories, 50), indent=2))


if __name__ == "__main__":
    main()
