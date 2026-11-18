"""在个人 PEG 的三维场景里并行评测 APF、ORCA、CBF-QP 和滚动时域 MPC。

场景、速度和终局用 PEG 原来的规则：圆柱和运动障碍、分层出生、
任意一架追捕机与逃逸机几何接触即捕获。飞行不锁高度。

    source ./activate.sh
    python -m parallel.examples.radar5_baseline_eval --episodes 50 --output runtime/baseline-50-peg3d
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from parallel.env.radar5_contract import physics_substeps
from parallel.examples.radar5_guidance import PursuerGuidance, evader_command
from parallel.training.mappo import _radar5_settings


def _settings():
    return _radar5_settings()


def _build(episodes: int, seed: int, output: Path, perception: str | None, formation: str | None):
    from omegaconf import OmegaConf
    from omni_drones import init_simulation_app
    from parallel.env.config_loader import configure_isaac_app, load_visualization
    from parallel.env.dependencies import resolve_source, select_source
    from parallel.strategies.config import load_config

    output.mkdir(parents=True, exist_ok=False)
    source = resolve_source(None)
    select_source(source)
    settings = _settings()
    visualization = load_visualization()
    visualization["enabled"] = False
    cfg = OmegaConf.load(source / "examples/demo.yaml")
    configure_isaac_app(cfg, settings["simulation"])
    torch.set_num_threads(4)
    cache = Path(__file__).resolve().parents[2] / "runtime/isaac-kit"
    cache.mkdir(parents=True, exist_ok=True)
    sys.argv.extend(["--portable-root", str(cache)])
    app = init_simulation_app(cfg)
    from parallel.env.od_env import ParallelEnv

    env = ParallelEnv(
        cfg,
        range(seed, seed + episodes),
        visualize=False,
        environment_config=settings,
        pursuer_config=load_config()[0],
        visualization_config=visualization,
        perception=perception,
        formation=formation,
    )
    return app, env


def run_method(env, method: str, seeds: list[int]) -> dict:
    device = env.device
    guidance = PursuerGuidance(method, env.n, device)
    env.reset(seeds=seeds)
    guidance.reset()
    outcome = torch.zeros(env.n, dtype=torch.long, device=device)
    end_step = torch.full((env.n,), -1, device=device)
    min_height = torch.full((env.n,), float("inf"), device=device)
    track_hits = 0
    track_slots = 0
    dt = float(env.rules.settings["simulation"]["dt"])
    substeps = physics_substeps(dt)
    max_steps = int(env.rules.settings["task"]["max_episode_steps"])
    steps = 0
    while steps < max_steps and not bool((outcome != 0).all()):
        state = env.state()
        command = torch.zeros(env.n, state.shape[1], 4, device=device, dtype=state.dtype)
        command[:, 0] = evader_command(env)
        command[:, 1:] = guidance.command(env)
        command[outcome != 0] = 0
        for _ in range(substeps):
            if steps >= max_steps:
                break
            _, _, terminated, truncated, info = env.step(command)
            steps += 1
            done = (terminated | truncated) & (outcome == 0)
            outcome = torch.where(done, info["outcome"], outcome)
            end_step = torch.where(done & (end_step < 0), end_step.new_tensor(steps), end_step)
            min_height = torch.minimum(min_height, info["state"][..., 2].amin(-1))
            if env.mid360 is not None:
                track_hits += int(env.mid360.valid.sum())
                track_slots += int(env.mid360.valid.numel())
            if bool((outcome != 0).all()):
                break
        if steps % (50 * substeps) < substeps:
            print(
                f"[baseline] {method} step={steps} "
                f"captured={int((outcome == 1).sum())} goal={int((outcome == 6).sum())} "
                f"open={int((outcome == 0).sum())}",
                flush=True,
            )
    captured = outcome == 1
    capture_times = end_step[captured].float() * dt
    episodes = env.n
    summary = {
        "episodes": episodes,
        "captures": int(captured.sum()),
        "goals": int((outcome == 6).sum()),
        "timeouts": int((outcome == 7).sum()),
        "out_of_bounds": int((outcome == 4).sum()),
        "static_collisions": int((outcome == 2).sum()),
        "dynamic_collisions": int((outcome == 3).sum()),
        "physical_failures": int((outcome == 5).sum()),
        "capture_rate": float(captured.float().mean()),
        "goal_rate": float((outcome == 6).float().mean()),
        "timeout_rate": float((outcome == 7).float().mean()),
        "mean_capture_seconds": float(capture_times.mean()) if int(captured.sum()) else None,
        "min_height": float(min_height.min()),
        "track_valid_rate": None if track_slots == 0 else track_hits / track_slots,
    }
    print(f"[baseline] {method} {json.dumps(summary)}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["apf", "frpn", "orca", "cbf", "mpc"])
    parser.add_argument("--perception", choices=("truth", "mid360"), default="mid360")
    parser.add_argument("--formation", choices=("stratified", "one_side"), default="one_side")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1024:
        parser.error("--episodes must be 1..1024")
    app, env = _build(
        args.episodes, args.seed, args.output,
        None if args.perception == "truth" else "mid360",
        None if args.formation == "stratified" else args.formation,
    )
    seeds = list(range(args.seed, args.seed + args.episodes))
    try:
        report = {
            "episodes": args.episodes,
            "seed_start": args.seed,
            "decision_dt": 0.2,
            "flight_mode": "free_3d",
            "capture": "3D swept contact between the evader and any living pursuer",
            "goal": "evader inside the 3D goal ball and no failure",
            "evader": "goal_avoid with vertical evasion up to 0.88 m/s",
            "evader_speed_limit": 2.0,
            "spawn": "goal side of the evader, 8–10 m, initially visible" if args.formation == "one_side" else "peg stratified",
            "perception": "per-pursuer MID-360 track with smoothed velocity, coast limit and teammate track sharing; no track searches the goal; cylinders occlude" if args.perception == "mid360" else "simulator truth",
            "obstacles": "16 static cylinders and 3 dynamic obstacles; any living aircraft that touches one ends the episode and is disabled",
            "guidance": "APF, FRPN, ORCA, CBF-QP and receding-horizon MPC each propose a velocity. The issued command of every method then passes the same brake half-space shield. Obstacle geometry and the target point both come from each pursuer's MID-360 track. Unseen cylinders are ignored. Tracked cylinders use a 0.55 m assumed radius.",
            "aircraft_collision": "3D body contact disables both aircraft; evader-pursuer contact is capture",
            "termination": "peg: capture, collision, out_of_bounds, physical_failure, success, timeout",
            "methods": {},
        }
        for method in args.methods:
            started = time.time()
            summary = run_method(env, method, seeds)
            summary["seconds"] = time.time() - started
            report["methods"][method] = summary
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        print(f"[baseline] wrote {args.output / 'summary.json'}", flush=True)
    except BaseException:
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
