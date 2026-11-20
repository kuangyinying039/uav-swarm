"""ORCA 教师的独立行为克隆。只采集成功和接近成功的局，不进入 DAgger。

接近成功：本局没有捕获，但某架追捕机与逃逸机的最小三维距离不超过 1.5 m。
标签是刹车前的四维意图。验证误差按水平速度、垂直速度、偏航角速度分开记录。
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch
from torch import nn

from parallel.env.radar5_contract import MAX_YAW_RATE, PURSUER_HORIZONTAL, PURSUER_VERTICAL
from parallel.examples.radar5_baseline_eval import _build
from parallel.examples.radar5_guidance import PursuerGuidance, evader_command, shield_pursuer_command
from parallel.training.aeg import LEVEL_SPEED
from parallel.training.epn import FUTURE_STEPS, HISTORY_STEPS, XY_SCALE, Z_SCALE, BatchEPN, radar_track
from parallel.training.hetero import graph_observation
from parallel.training.mappo import Actor, command_to_unit, hold_decision, pursuer_command

NEAR_DISTANCE_M = 1.5
TRAIN_EPISODES = 300
BC_EPOCHS = 40
PLATEAU_EPOCHS = 5


def _set_medium(env) -> None:
    horizontal, vertical = LEVEL_SPEED["medium"]
    env.evader_horizontal = torch.full((env.n,), horizontal, device=env.device)
    env.evader_vertical = torch.full((env.n,), vertical, device=env.device)


def _rows(rows: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([row[0] for row in rows]),
        torch.stack([row[1] for row in rows]),
        torch.stack([row[2] for row in rows]),
    )


def _component_errors(prediction: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> dict:
    gap = prediction - command
    weight = alive.float()
    count = weight.sum().clamp_min(1.0)

    def rmse(error: torch.Tensor) -> float:
        return float((error * weight).sum().div(count).sqrt().item())

    unit_gap = command_to_unit(prediction) - command_to_unit(command)
    return {
        "horizontal_rmse_m_s": rmse(gap[..., :2].square().mean(-1)),
        "vertical_rmse_m_s": rmse(gap[..., 2].square()),
        "yaw_rmse_rad_s": rmse(gap[..., 3].square()),
        "normalized_mse": float((unit_gap.square().mean(-1) * weight).sum().div(count).item()),
    }


def _masked_mse(prediction: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    error = (command_to_unit(prediction) - command_to_unit(command)).square().mean(-1)
    weight = alive.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def _dataset_errors(actor: Actor, obs: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> dict:
    pieces = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], 1024):
            batch = slice(start, start + 1024)
            mean, _std = actor(obs[batch])
            pieces.append(_component_errors(pursuer_command(mean), command[batch], alive[batch]))
    keys = pieces[0]
    return {key: sum(piece[key] for piece in pieces) / len(pieces) for key in keys}


def _consistency(obs: torch.Tensor, command: torch.Tensor, executed: torch.Tensor, alive: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(obs).all() and torch.isfinite(command).all())
    horizontal = torch.linalg.vector_norm(command[..., :2], dim=-1)
    inside = (
        (horizontal <= PURSUER_HORIZONTAL + 1e-4)
        & (command[..., 2].abs() <= PURSUER_VERTICAL + 1e-4)
        & (command[..., 3].abs() <= MAX_YAW_RATE + 1e-4)
    )
    changed = torch.linalg.vector_norm(command[..., :3] - executed[..., :3], dim=-1) > 0.05
    weight = alive.float()
    return {
        "labels_finite": finite,
        "labels_inside_actuator_limits": bool(inside.all()),
        "alive_fraction": float(weight.mean().item()),
        "shield_changed_label_rate": float((changed.float() * weight).sum().div(weight.sum().clamp_min(1.0)).item()),
        "observation_dim": int(obs.shape[-1]),
        "action": ["vx", "vy", "vz", "yaw_rate"],
        "label": "pre-brake ORCA intent; the shield is applied only at execution",
    }


def collect(env, teacher: PursuerGuidance, forecaster: BatchEPN, observe, train_goal: int):
    device = env.device
    max_steps = int(env.rules.settings["task"]["max_episode_steps"])
    teacher.reset()
    _set_medium(env)
    env.reset(seeds=list(range(1000, 1000 + env.n)))
    forecaster.reset(torch.arange(env.n, device=device))
    pending = [[] for _ in range(env.n)]
    executed_rows = [[] for _ in range(env.n)]
    min_distance = torch.full((env.n,), float("inf"), device=device)
    episode_ids = torch.zeros(env.n, dtype=torch.long, device=device)
    next_seed = 1000 + env.n
    train_obs, train_cmd, train_alive, train_exec = [], [], [], []
    val_obs, val_cmd, val_alive, val_exec = [], [], [], []
    success_episodes = near_episodes = discarded = 0
    sequence_buf, pursuer_buf, evader_buf, id_buf = [], [], [], []
    seen = 0
    while len(train_obs) < train_goal:
        state = env.state()
        progress = 1.0 - env.rules.steps.float() / max_steps
        obs = observe(state, progress)
        preferred, executed = teacher.propose(env)
        alive = (~env.disabled[:, 1:]).detach()
        distance = torch.linalg.vector_norm(state[:, 1:, :3] - state[:, :1, :3], dim=-1)
        distance = distance.masked_fill(~alive, float("inf"))
        min_distance = torch.minimum(min_distance, distance.amin(-1))
        for index in range(env.n):
            pending[index].append((obs[index].detach(), preferred[index].detach(), alive[index].detach()))
            executed_rows[index].append(executed[index].detach())
        command = torch.zeros(env.n, state.shape[1], 4, device=device, dtype=state.dtype)
        command[:, 0] = evader_command(env)
        command[:, 1:] = executed
        sequence_buf.append(forecaster.history.detach().clone())
        pursuer_buf.append(state[:, 1:, :3].detach())
        evader_buf.append(state[:, 0, :3].detach())
        id_buf.append(episode_ids.clone())
        _reward, done, captured, _info, _outcome = hold_decision(env, command)
        if len(sequence_buf) >= 32:
            forecaster.update(torch.stack(sequence_buf), torch.stack(pursuer_buf), torch.stack(evader_buf), torch.stack(id_buf))
            sequence_buf, pursuer_buf, evader_buf, id_buf = [], [], [], []
        finished = done.nonzero(as_tuple=False).flatten()
        if not finished.numel():
            continue
        kept_ids = []
        for index in finished.tolist():
            rows = pending[index]
            flown = executed_rows[index]
            pending[index] = []
            executed_rows[index] = []
            close = float(min_distance[index].item())
            min_distance[index] = float("inf")
            qualified = bool(captured[index].item()) or close <= NEAR_DISTANCE_M
            if not rows or not qualified:
                discarded += 0 if not rows else 1
                continue
            obs_i, cmd_i, alive_i = _rows(rows)
            exec_i = torch.stack(flown)
            bucket = val_obs if seen % 8 == 0 else train_obs
            (val_cmd if bucket is val_obs else train_cmd).append(cmd_i)
            (val_alive if bucket is val_obs else train_alive).append(alive_i)
            (val_exec if bucket is val_obs else train_exec).append(exec_i)
            bucket.append(obs_i)
            seen += 1
            if bool(captured[index].item()):
                success_episodes += 1
            else:
                near_episodes += 1
            kept_ids.append(index)
        episode_ids[finished] += 1
        forecaster.reset(finished)
        teacher.previous[finished] = 0
        seeds = list(range(next_seed, next_seed + int(finished.numel())))
        next_seed += int(finished.numel())
        env.reset(finished, seeds=seeds)
        _set_medium(env)
        if len(train_obs) % 25 == 0 and kept_ids:
            print(
                f"[bc] train_episodes={len(train_obs)} val_episodes={len(val_obs)} "
                f"success={success_episodes} near={near_episodes} discarded={discarded}",
                flush=True,
            )
    def cat(chunks):
        return torch.cat(chunks, 0)

    return {
        "train": (cat(train_obs), cat(train_cmd), cat(train_alive), cat(train_exec)),
        "val": (cat(val_obs), cat(val_cmd), cat(val_alive), cat(val_exec)),
        "counts": {
            "train_episodes": len(train_obs),
            "val_episodes": len(val_obs),
            "success_episodes": success_episodes,
            "near_success_episodes": near_episodes,
            "discarded_episodes": discarded,
            "near_distance_m": NEAR_DISTANCE_M,
            "train_rows": int(cat(train_obs).shape[0]),
            "val_rows": int(cat(val_obs).shape[0]),
        },
    }


def fit(actor: Actor, train, val) -> list[dict]:
    obs, command, alive, _executed = train
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    history = []
    best = None
    best_state = None
    quiet = 0
    for epoch in range(1, BC_EPOCHS + 1):
        order = torch.randperm(obs.shape[0], device=obs.device)
        for start in range(0, obs.shape[0], 1024):
            batch = order[start:start + 1024]
            if batch.numel() < 8:
                continue
            mean, _std = actor(obs[batch])
            loss = _masked_mse(pursuer_command(mean), command[batch], alive[batch])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
        row = {"epoch": epoch, "train": _dataset_errors(actor, *train[:3]), "val": _dataset_errors(actor, *val[:3])}
        history.append(row)
        score = row["val"]["normalized_mse"]
        if best is None or score < best - 1e-4:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
            quiet = 0
        else:
            quiet += 1
        print(
            f"[bc] epoch={epoch} val_h={row['val']['horizontal_rmse_m_s']:.4f} "
            f"val_v={row['val']['vertical_rmse_m_s']:.4f} val_yaw={row['val']['yaw_rmse_rad_s']:.4f} "
            f"val_mse={score:.4f}",
            flush=True,
        )
        if quiet >= PLATEAU_EPOCHS:
            break
    actor.load_state_dict(best_state)
    return history


def _fly(env, command_fn, episodes: int, seed: int, label: str) -> dict:
    """同一批种子上闭环飞行。已结束的局不再发指令。"""
    from parallel.env.radar5_contract import physics_substeps

    device = env.device
    width = min(episodes, env.n)
    ids = torch.arange(width, device=device)
    _set_medium(env)
    env.reset(ids, seeds=list(range(seed, seed + width)))
    outcome = torch.zeros(env.n, dtype=torch.long, device=device)
    dt = float(env.rules.settings["simulation"]["dt"])
    substeps = physics_substeps(dt)
    max_steps = int(env.rules.settings["task"]["max_episode_steps"])
    steps = 0
    while steps < max_steps and not bool((outcome[:width] != 0).all()):
        state = env.state()
        executed = command_fn(state)
        command = torch.zeros(env.n, state.shape[1], 4, device=device, dtype=state.dtype)
        command[:, 0] = evader_command(env)
        command[:, 1:] = executed
        command[outcome != 0] = 0
        for _ in range(substeps):
            if steps >= max_steps:
                break
            _obs, _reward, terminated, truncated, info = env.step(command)
            steps += 1
            fresh = (terminated | truncated) & (outcome == 0)
            outcome = torch.where(fresh, info["outcome"].to(outcome.dtype), outcome)
            if bool((outcome[:width] != 0).all()):
                break
        if steps % 500 == 0:
            print(f"[bc] {label} physics_steps={steps} finished={int((outcome[:width] != 0).sum())}/{width}", flush=True)
    counts = outcome[:width]
    return {
        "episodes": width,
        "seed_start": seed,
        "captures": int((counts == 1).sum()),
        "capture_rate": float((counts == 1).float().mean().item()),
        "goals": int((counts == 6).sum()),
        "static_collisions": int((counts == 2).sum()),
        "dynamic_collisions": int((counts == 3).sum()),
        "out_of_bounds": int((counts == 4).sum()),
        "physical_failures": int((counts == 5).sum()),
        "timeouts": int((counts == 7).sum()),
        "ongoing": int((counts == 0).sum()),
        "label": label,
    }


def main() -> None:
    output = Path("/srv/data/projects/kyy/xieton/repro/results/bc-orca-300")
    app, env = _build(100, 1000, output, "mid360", "one_side")
    try:
        _set_medium(env)
        device = env.device
        state = env.state()
        forecaster = BatchEPN(env.n, 3, device)
        teacher = PursuerGuidance("orca", env.n, device)

        def observe(state, progress):
            return graph_observation(state, env, forecaster.observe(state, progress, *radar_track(env)))

        gathered = collect(env, teacher, forecaster, observe, TRAIN_EPISODES)
        actor = Actor(gathered["train"][0].shape[-1], 4).to(device)
        history = fit(actor, gathered["train"], gathered["val"])
        final = {
            "train": _dataset_errors(actor, *gathered["train"][:3]),
            "val": _dataset_errors(actor, *gathered["val"][:3]),
        }
        train_obs, train_cmd, train_alive, train_exec = gathered["train"]
        consistency = _consistency(train_obs, train_cmd, train_exec, train_alive)
        torch.save(
            {
                "actor": actor.state_dict(),
                "epn": forecaster.net.state_dict(),
                "teacher": "orca",
            },
            output / "bc_orca.pt",
        )
        print("[bc] checkpoint saved, student closed loop", flush=True)
        forecaster.reset(torch.arange(env.n, device=device))

        def student_command(state):
            progress = 1.0 - env.rules.steps.float() / int(env.rules.settings["task"]["max_episode_steps"])
            with torch.no_grad():
                mean, _std = actor(observe(state, progress))
                return shield_pursuer_command(env, pursuer_command(mean))

        student = _fly(env, student_command, 100, 2_000_000, "student")
        teacher.reset()
        forecaster.reset(torch.arange(env.n, device=device))

        def teacher_command(_state):
            _preferred, executed = teacher.propose(env)
            return executed

        reference = _fly(env, teacher_command, 100, 2_000_000, "teacher")
        report = {
            "teacher": "orca",
            "stage": "behavior_cloning",
            "dagger": False,
            "evader": "medium 1.46 m/s horizontal, 0.84 m/s vertical",
            "counts": gathered["counts"],
            "consistency": consistency,
            "errors": final,
            "history": history,
            "closed_loop": student,
            "teacher_reference": reference,
            "preprocess": {
                "perception": "mid360",
                "formation": "one_side",
                "graph": "hetero local graph, peers shared, top-3 radar obstacles, 15-d EPN forecast",
                "epn_history": HISTORY_STEPS,
                "epn_future": FUTURE_STEPS,
                "epn_scales_m": [XY_SCALE, XY_SCALE, Z_SCALE],
                "action_scales": [PURSUER_HORIZONTAL, PURSUER_HORIZONTAL, PURSUER_VERTICAL, MAX_YAW_RATE],
                "bc_loss": "normalized four-dimensional MSE of the pre-brake intent",
                "stop": "validation normalized MSE plateau, not a 0.35 threshold",
            },
        }
        torch.save(
            {
                "actor": actor.state_dict(),
                "epn": forecaster.net.state_dict(),
                "preprocess": report["preprocess"],
                "teacher": "orca",
            },
            output / "bc_orca.pt",
        )
        (output / "bc_report.json").write_text(json.dumps(report, indent=2))
        print(f"[bc] wrote {output / 'bc_report.json'}", flush=True)
        print(json.dumps({"errors": final, "closed_loop": student, "teacher_reference": reference, "counts": gathered["counts"]}, indent=2), flush=True)
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
