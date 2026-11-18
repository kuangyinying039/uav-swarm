"""Shared-parameter MAPPO for three pursuers on the personal PEG copy.

The plant accepts the Radar5 command (vx, vy, vz, yaw_rate). One policy
decision is held for 0.2 s. The evader uses goal_avoid plus vertical evasion.
Capture is one living pursuer in 3D body contact. The shared reward weights
approach and cooperative enclosure more heavily than separation or obstacles,
then adds the capture bonus, a collision penalty, and a per-step time penalty.

Run from environments/configurable-pursuit after source ./activate.sh:

    python -m parallel.training.mappo --num-envs 128 --updates 20 --rollout-steps 32 --output runtime/mappo-radar5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from parallel.env.mid360 import sensed_target
from parallel.env.radar5_contract import (
    BODY_CONTACT,
    BODY_RADIUS,
    COLLISION_PENALTY,
    GAMMA,
    MAX_YAW_RATE,
    PHYSICS_DT,
    PURSUER_HORIZONTAL,
    PURSUER_VERTICAL,
    TIME_PENALTY,
    body_yaw,
    decision_reward,
    physics_substeps,
)


def _radar5_settings():
    from parallel.env.config_loader import load_settings

    settings = load_settings()
    settings["simulation"]["headless"] = True
    settings["simulation"]["flight_mode"] = "free_3d"
    settings["simulation"]["target_height"] = 5.0
    settings["simulation"]["dt"] = PHYSICS_DT
    # 平面配置把高度卡在 1.2–1.8 米。三维仍用 PEG 的场景和终局，但高度窗口打开。
    failure = settings["task"]["physical_failure"]
    failure["min_height"] = 0.5
    failure["max_height"] = 14.0
    failure["max_horizontal_speed"] = 4.0
    return settings


def _training_settings():
    """训练用的稀疏场景。正式 PEG 配置仍是 16 根圆柱和 3 个运动障碍。"""
    settings = _radar5_settings()
    settings["scene"]["static_obstacles"]["count"] = 4
    settings["scene"]["dynamic_obstacles"]["count"] = 0
    return settings


def bc_stalled(values: list[float], fit: float, window: int = 16, tolerance: float = 0.02) -> bool:
    """最近一段时间的最好克隆误差没有再下降，并且还没降到拟合线。"""
    if len(values) < window + 8 or min(values) <= fit:
        return False
    return min(values[-window:]) > min(values[:-window]) - tolerance


def still_training(env_steps: int, imitation_max_steps: int, rl_steps: int, teacher_on: bool, rl_at_steps: int | None) -> bool:
    """预算按环境决策步数。模仿阶段有自己的上限，强化学习从进入时再计 rl_steps。"""
    if not teacher_on:
        return env_steps < rl_steps
    if rl_at_steps is None:
        return env_steps < imitation_max_steps
    return env_steps < rl_at_steps + rl_steps


def reference_scale(env_steps: int, rl_at_steps: int | None, rl_steps: int) -> float:
    """强化学习前 20% 把参照约束保持在 1，之后线性降到 0。"""
    if rl_at_steps is None:
        return 1.0
    progress = (env_steps - rl_at_steps) / float(max(rl_steps, 1))
    if progress <= 0.2:
        return 1.0
    return max(0.0, 1.0 - (progress - 0.2) / 0.8)


def imitation_phase(
    update: int,
    bc_min: int,
    bc_max: int,
    decay: int,
    origin: int | None,
    stalled: bool,
    fitted: bool,
) -> tuple[float, int | None, str]:
    """先行为克隆，再正好 decay 次 DAgger。最后一次 DAgger 的教师概率是 0，克隆完成之后下一次才进入强化学习。"""
    if origin is None:
        if update < bc_min or (update < bc_max and not stalled and not fitted):
            return 1.0, None, "bc"
        return 1.0, update, "dagger"
    elapsed = update - origin
    if elapsed >= decay:
        return 0.0, origin, "rl"
    beta = 1.0 - elapsed / float(max(decay - 1, 1))
    return beta, origin, "dagger"


def _build_app(num_envs: int, seed: int, output: Path, scene: str = "full", perception: str = "mid360"):
    from omegaconf import OmegaConf
    from omni_drones import init_simulation_app
    from parallel.env.config_loader import configure_isaac_app, load_visualization
    from parallel.env.dependencies import resolve_source, select_source
    from parallel.strategies.config import load_config

    output.mkdir(parents=True, exist_ok=False)
    source = resolve_source(None)
    select_source(source)
    settings = _training_settings() if scene == "simple" else _radar5_settings()
    pursuer = load_config()[0]
    visualization = load_visualization()
    visualization["enabled"] = False
    cfg = OmegaConf.load(source / "examples/demo.yaml")
    configure_isaac_app(cfg, settings["simulation"])
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    private_cache = Path(__file__).resolve().parents[2] / "runtime/isaac-kit"
    private_cache.mkdir(parents=True, exist_ok=True)
    sys.argv.extend(["--portable-root", str(private_cache)])
    app = init_simulation_app(cfg)
    from parallel.env.od_env import ParallelEnv

    env = ParallelEnv(
        cfg,
        range(seed, seed + num_envs),
        visualize=False,
        environment_config=settings,
        pursuer_config=pursuer,
        visualization_config=visualization,
        perception=None if perception == "truth" else perception,
        formation="one_side",
    )
    return app, env


def evader_world_velocity(env) -> torch.Tensor:
    """Public goal_avoid baseline, mapped back to world XY m/s."""
    from parallel.examples.evader_baselines import action

    rules = env.rules
    local = action(env, "goal_avoid")
    world = (local.unsqueeze(-1) * rules.frame).sum(1)
    speed = getattr(env, "evader_horizontal", None)
    if speed is None:
        speed = float(rules.settings["simulation"]["evader_speed_limit"])
        return world * speed
    return world * speed.to(device=world.device, dtype=world.dtype).unsqueeze(-1)


def scene_observation(state: torch.Tensor, env) -> torch.Tensor:
    """最近的本机障碍航迹和目标点。当前策略不读这个向量，图观测才是执行入口。"""
    from parallel.env.mid360 import sensed_cylinders

    pursuers = state[:, 1:, :3]
    n, count, _ = pursuers.shape
    xy = pursuers[..., :2]
    centers, radii, _motion, valid, dynamic = sensed_cylinders(env)
    pieces = []
    if centers.shape[1] != count or centers.shape[2] == 0:
        pieces.append(xy.new_zeros(n, count, 12))
        pieces.append(xy.new_zeros(n, count, 9))
    else:
        delta = xy[:, :, None, :] - centers
        gap = torch.linalg.vector_norm(delta, dim=-1) - radii - BODY_RADIUS
        gap = gap.masked_fill(~valid, 1.0e6)

        def _slots(masked_gap, slots: int) -> torch.Tensor:
            take = min(slots, int(masked_gap.shape[-1]))
            gaps, index = torch.topk(masked_gap, take, dim=-1, largest=False)
            chosen = delta.gather(2, index.unsqueeze(-1).expand(n, count, take, 2))
            seen = gaps < 1.0e5
            near = torch.cat((chosen / 20.0, (gaps / 5.0).unsqueeze(-1)), dim=-1)
            near = near.masked_fill(~seen.unsqueeze(-1), 0.0)
            if take < slots:
                near = torch.cat((near, near.new_zeros(n, count, slots - take, 3)), dim=2)
            return near.reshape(n, count, slots * 3)

        pieces.append(_slots(gap.masked_fill(dynamic, 1.0e6), 4))
        pieces.append(_slots(gap.masked_fill(~dynamic, 1.0e6), 3))
    height = float(env.rules.settings["simulation"]["target_height"])
    goal = torch.cat((env.rules.goal, pursuers.new_full((n, 1), height)), dim=-1)
    pieces.append((goal[:, None, :] - pursuers) / pursuers.new_tensor([20.0, 20.0, 12.0]))
    return torch.cat(pieces, dim=-1)


def _peer_offsets(points: torch.Tensor) -> torch.Tensor:
    """其余追捕机的相对位置，近的排在前面。教师用它决定谁去拦截、谁去堵。"""
    relative = points[:, None, :, :] - points[:, :, None, :]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    eye = torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
    distance = distance.masked_fill(eye, float("inf"))
    order = distance.argsort(dim=-1)[..., : max(points.shape[1] - 1, 1)]
    chosen = relative.gather(2, order.unsqueeze(-1).expand(*order.shape, 3))
    return (chosen / 20.0).reshape(points.shape[0], points.shape[1], -1)


def pursuer_observation(state: torch.Tensor, env=None) -> torch.Tensor:
    """Local observation [N, P, 12]. 目标相对位置和速度来自 MID-360 航迹。"""
    pursuers = state[:, 1:]
    if env is not None and getattr(env, "mid360", None) is not None:
        target, target_vel, valid = sensed_target(env)
        rel = torch.where(valid.unsqueeze(-1), pursuers[..., :3] - target, torch.zeros_like(pursuers[..., :3]))
        evader_vel = torch.where(valid.unsqueeze(-1), target_vel, torch.zeros_like(target_vel))
    else:
        evader = state[:, 0]
        rel = pursuers[..., :3] - evader[:, None, :3]
        evader_vel = evader[:, None, 7:10].expand(pursuers.shape[0], pursuers.shape[1], 3)
    rel = rel / rel.new_tensor([20.0, 20.0, 12.0])
    own_vel = pursuers[..., 7:10] / pursuers.new_tensor(
        [PURSUER_HORIZONTAL, PURSUER_HORIZONTAL, PURSUER_VERTICAL]
    )
    evader_vel = evader_vel / evader_vel.new_tensor([1.52, 1.52, 0.88])
    others = pursuers[..., :2].mean(dim=1, keepdim=True) - pursuers[..., :2]
    yaw = body_yaw(pursuers[..., 3:7]).unsqueeze(-1) / torch.pi
    peer = _peer_offsets(pursuers[..., :3])
    if env is not None and getattr(env, "mid360", None) is not None:
        distance = torch.linalg.vector_norm(pursuers[..., :3] - target, dim=-1)
        distance = torch.where(valid, distance, torch.zeros_like(distance))
    else:
        distance = torch.linalg.vector_norm(pursuers[..., :3] - state[:, 0, :3][:, None, :], dim=-1)
    farthest = distance.argmax(dim=-1, keepdim=True)
    index = torch.arange(pursuers.shape[1], device=pursuers.device).view(1, -1)
    blocker = (farthest == index).unsqueeze(-1).to(rel.dtype)
    from parallel.examples.radar5_guidance import aim_points
    aim = (aim_points(env) - pursuers[..., :3]) / pursuers.new_tensor([20.0, 20.0, 12.0]) if env is not None else rel[..., :3]
    return torch.cat((rel, own_vel, evader_vel, others / 20.0, yaw, peer, blocker, aim), dim=-1)


def joint_state(state: torch.Tensor) -> torch.Tensor:
    """Centralized critic input: xyz, world velocity and yaw of every aircraft."""
    yaw = body_yaw(state[..., 3:7]).unsqueeze(-1)
    return torch.cat((state[..., :3], state[..., 7:10], yaw), dim=-1).flatten(1)


def project_unit(unit: torch.Tensor) -> torch.Tensor:
    """把动作压进水平单位圆，垂直和偏航各自留在 [-1, 1]。"""
    horizontal = unit[..., :2]
    norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(1.0 / norm, max=1.0)
    return torch.cat((horizontal * scale, unit[..., 2:].clamp(-1.0, 1.0)), dim=-1)


def unit_to_command(unit: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 动作变成追捕机的 (vx, vy, vz, yaw_rate)，水平速度再按模长截断。"""
    horizontal = unit[..., :2] * PURSUER_HORIZONTAL
    speed = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    scale = torch.clamp(PURSUER_HORIZONTAL / speed.clamp_min(1e-12), max=1.0)
    return torch.cat(
        (horizontal * scale, unit[..., 2:3] * PURSUER_VERTICAL, unit[..., 3:4] * MAX_YAW_RATE),
        dim=-1,
    )


def command_to_unit(command: torch.Tensor) -> torch.Tensor:
    """速度指令映回 [-1, 1]。水平分量已经在圆盘内，除以速度上限即可还原。"""
    return torch.cat(
        (
            command[..., :2] / PURSUER_HORIZONTAL,
            command[..., 2:3] / PURSUER_VERTICAL,
            command[..., 3:4] / MAX_YAW_RATE,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def pursuer_command(raw: torch.Tensor) -> torch.Tensor:
    """Tanh 样本变成追捕机的 (vx, vy, vz, yaw_rate)。"""
    return unit_to_command(torch.tanh(raw))


class SuccessSteps:
    """按环境暂存本局样本。局结束时全部写入，失败局也保留碰撞前的状态。"""

    def __init__(self, n: int):
        self.pending: list[list] = [[] for _ in range(n)]
        self.kept = 0
        self.success_rows = 0
        self.failure_rows = 0

    def push_env(self, index: int, row) -> None:
        self.pending[index].append(row)

    def commit(self, finished, captured, accept) -> None:
        for index in finished.tolist():
            rows = self.pending[index]
            self.pending[index] = []
            if not rows:
                continue
            accept(index, rows)
            self.kept += len(rows)
            if bool(captured[index].item()):
                self.success_rows += len(rows)
            else:
                self.failure_rows += len(rows)


def _stack_rows(rows: list, keys: tuple[str, ...] | None = None):
    if keys is None:
        return tuple(torch.stack([row[i] for row in rows]) for i in range(len(rows[0])))
    return {key: torch.stack([row[key] for row in rows]) for key in keys}


class CommandReplay:
    """教师意图的累计缓存。标签是刹车前意图，失败局同样保留。"""

    def __init__(self, capacity: int, n_pursuers: int, obs_dim: int, action_dim: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros(self.capacity, n_pursuers, obs_dim, device=device)
        self.command = torch.zeros(self.capacity, n_pursuers, action_dim, device=device)
        self.alive = torch.zeros(self.capacity, n_pursuers, device=device)
        self.pos = 0
        self.size = 0
        self.demo = None

    def freeze(self) -> None:
        """留下教师自己飞出来的演示，后面的学习者数据不能把它覆盖掉。"""
        if self.demo is not None or self.size < 8:
            return
        self.demo = (
            self.obs[: self.size].detach().clone(),
            self.command[: self.size].detach().clone(),
            self.alive[: self.size].detach().clone(),
        )

    def add(self, obs: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> None:
        obs = obs.reshape(-1, *self.obs.shape[1:])
        command = command.reshape(-1, *self.command.shape[1:])
        alive = alive.reshape(-1, self.alive.shape[1]).float()
        count = int(obs.shape[0])
        if count > self.capacity:
            obs, command, alive = obs[-self.capacity:], command[-self.capacity:], alive[-self.capacity:]
            count = self.capacity
        first = min(count, self.capacity - self.pos)
        self._place(self.pos, obs[:first], command[:first], alive[:first])
        if count > first:
            self._place(0, obs[first:], command[first:], alive[first:])
        self.pos = (self.pos + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def _place(self, index: int, obs, command, alive) -> None:
        width = int(obs.shape[0])
        self.obs[index:index + width] = obs
        self.command[index:index + width] = command
        self.alive[index:index + width] = alive

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.demo is None:
            index = torch.randint(0, self.size, (batch,), device=self.device)
            return self.obs[index], self.command[index], self.alive[index]
        demo_obs, demo_command, demo_alive = self.demo
        demo_count = max(batch // 2, 1)
        online_count = batch - demo_count
        demo_index = torch.randint(0, demo_obs.shape[0], (demo_count,), device=self.device)
        online_index = torch.randint(0, self.size, (online_count,), device=self.device)
        return (
            torch.cat((demo_obs[demo_index], self.obs[online_index]), 0),
            torch.cat((demo_command[demo_index], self.command[online_index]), 0),
            torch.cat((demo_alive[demo_index], self.alive[online_index]), 0),
        )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        from parallel.training.hetero import HeteroEncoder
        self.encoder = HeteroEncoder(obs_dim)
        self.head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.head(self.encoder(obs))
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def squash_log_prob(normal: Normal, raw: torch.Tensor) -> torch.Tensor:
    squashed = torch.tanh(raw)
    log_det = torch.log(1.0 - squashed.square() + 1e-6).sum(-1)
    return normal.log_prob(raw).sum(-1) - log_det


def gae(reward, value, done, last_value, gamma=GAMMA, lam=0.95):
    advantage = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for t in reversed(range(reward.shape[0])):
        next_value = last_value if t == reward.shape[0] - 1 else value[t + 1]
        next_mask = 1.0 - done[t]
        delta = reward[t] + gamma * next_value * next_mask - value[t]
        running = delta + gamma * lam * next_mask * running
        advantage[t] = running
    return advantage, advantage + value


def hold_decision(env, command: torch.Tensor):
    """同一四维指令保持一个决策周期，并在周期末结算一次奖励。"""
    before = env.state()
    active_before = ~env.disabled[:, 1:]
    static = env.rules.static
    dynamic_before = env.rules.bg.clone()
    dynamic_radii = env.rules.radii[:, env.rules.layout.dynamic_objects]
    captured = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    collision = torch.zeros_like(captured)
    terminated = torch.zeros_like(captured)
    truncated = torch.zeros_like(captured)
    outcome = torch.zeros(env.n, dtype=torch.long, device=env.device)
    live = command
    info = None
    for _ in range(physics_substeps(float(env.rules.settings["simulation"]["dt"]))):
        live = live.masked_fill(env.disabled.unsqueeze(-1), 0)
        _, _, term, trunc, info = env.step(live)
        captured |= info["captured"]
        collision |= info["pursuer_obstacle"] | info["pursuer_peer"]
        terminated |= term
        truncated |= trunc
        stop = terminated | truncated | captured
        fresh = stop & (outcome == 0)
        outcome = torch.where(fresh, info["outcome"].to(outcome.dtype), outcome)
        live = live.clone()
        live[stop] = 0
        if bool(stop.all()):
            break
    after = info["state"]
    reward = decision_reward(
        before,
        after,
        captured=captured,
        truncated=truncated,
        terminated=terminated,
        collision=collision,
        active_before=active_before,
        active_after=~env.disabled[:, 1:],
        static=static,
        dynamic_before=dynamic_before,
        dynamic_after=env.rules.bg,
        dynamic_radii=dynamic_radii,
    )
    done = terminated | truncated | captured
    return reward, done, captured, info, outcome


def _apply_level(env, index: int, level: str) -> None:
    from parallel.training.aeg import LEVEL_SPEED

    horizontal, vertical = LEVEL_SPEED[level]
    env.evader_horizontal[index] = horizontal
    env.evader_vertical[index] = vertical


def _outcome_report(outcomes: torch.Tensor) -> dict:
    total = max(int(outcomes.numel()), 1)

    def count(code: int) -> int:
        return int((outcomes == code).sum())

    captures = count(1)
    static_collisions = count(2)
    dynamic_collisions = count(3)
    return {
        "episodes": int(outcomes.numel()),
        "captures": captures,
        "capture_rate": captures / total,
        "static_collisions": static_collisions,
        "dynamic_collisions": dynamic_collisions,
        "collision_rate": (static_collisions + dynamic_collisions) / total,
        "out_of_bounds": count(4),
        "physical_failures": count(5),
        "goals": count(6),
        "timeouts": count(7),
        "ongoing": count(0),
    }


def _evaluate_batches(env, forecaster, observe, evader_command, max_steps, pursue, seed_start: int, episodes: int) -> dict:
    """按环境数分批评测。监控、确认和最终测试使用不同的种子段。"""
    width = int(env.n)
    batches = max(1, (int(episodes) + width - 1) // width)
    combined = None
    keys = ("episodes", "captures", "static_collisions", "dynamic_collisions", "out_of_bounds", "physical_failures", "goals", "timeouts", "ongoing")
    for batch in range(batches):
        seeds = list(range(seed_start + batch * width, seed_start + (batch + 1) * width))
        report = _evaluate_controller(env, forecaster, observe, evader_command, max_steps, pursue, seeds)
        if combined is None:
            combined = report
            continue
        for key in keys:
            combined[key] += report[key]
    total = max(int(combined["episodes"]), 1)
    combined["capture_rate"] = combined["captures"] / total
    combined["collision_rate"] = (combined["static_collisions"] + combined["dynamic_collisions"]) / total
    return combined


def _intent_nll(mean: torch.Tensor, log_std: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """高斯负对数似然。回归的是刹车前意图在 tanh 之前的坐标。"""
    unit = command_to_unit(command).clamp(-0.999, 0.999)
    target = torch.atanh(unit)
    std = log_std.exp()
    nll = (0.5 * ((target - mean) / std).square() + log_std).sum(-1)
    return (nll * alive).sum() / alive.sum().clamp_min(1.0)


def _intent_mse(prediction: torch.Tensor, command: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    error = (prediction - command).square().sum(-1)
    return (error * alive).sum() / alive.sum().clamp_min(1.0)


def _evaluate_controller(env, forecaster, observe, evader_command, max_steps, pursue, seeds) -> dict:
    """pursue(state, progress) 返回已经过安全过滤的追捕机指令。"""
    device = env.device
    ids = torch.arange(env.n, device=device)
    forecaster.reset(ids)
    env.reset(ids, seeds=list(seeds))
    decisions = max(int(max_steps) // physics_substeps(float(env.rules.settings["simulation"]["dt"])), 1)
    pending = torch.ones(env.n, dtype=torch.bool, device=device)
    outcomes = torch.zeros(env.n, dtype=torch.long, device=device)
    for _ in range(decisions):
        if not bool(pending.any()):
            break
        state = env.state()
        progress = 1.0 - env.rules.steps.float() / max_steps
        with torch.no_grad():
            command = torch.zeros(env.n, state.shape[1], 4, device=device)
            command[:, 0] = evader_command(env)
            command[:, 1:] = pursue(state, progress)
            command[~pending] = 0
        _reward, done, _got, _info, outcome = hold_decision(env, command)
        fresh = pending & done
        outcomes = torch.where(fresh, outcome, outcomes)
        pending &= ~done
    return _outcome_report(outcomes)


def train(args):
    from parallel.examples.radar5_guidance import PursuerGuidance, evader_command, shield_pursuer_command
    from parallel.training.aeg import AEGConfig, DiscreteAEG, LEVEL_SPEED, LayoutCurriculum, reset_with_curriculum
    from parallel.training.hetero import graph_observation
    from parallel.training.epn import FUTURE_STEPS, BatchEPN, radar_track

    app, env = _build_app(args.num_envs, args.seed, args.output, args.scene, args.perception)
    device = env.device
    try:
        state = env.state()
        n_pursuers = state.shape[1] - 1
        if n_pursuers != 3:
            raise RuntimeError(f"Radar5 expects 3 pursuers, found {n_pursuers}")
        static_count = int(env.rules.static.shape[1])
        dynamic_count = int(env.rules.bg.shape[1])
        if args.scene == "simple":
            evader_level = "medium"
            horizontal, vertical = LEVEL_SPEED[evader_level]
            env.evader_horizontal = torch.full((env.n,), horizontal, device=device)
            env.evader_vertical = torch.full((env.n,), vertical, device=device)
            if (static_count, dynamic_count) != (4, 0):
                raise RuntimeError(f"Expected 4 static and 0 dynamic obstacles, found {static_count}, {dynamic_count}")
        else:
            if (static_count, dynamic_count) != (16, 3):
                raise RuntimeError(f"Expected the PEG scene, found {static_count} static and {dynamic_count} dynamic")
            evader_level = "aeg"
            aeg = DiscreteAEG(AEGConfig(cold_start_level="medium"))
            aeg.set_seed(args.seed)
            level_of = [aeg.cfg.cold_start_level] * env.n
            env.evader_horizontal = torch.empty(env.n, device=device)
            env.evader_vertical = torch.empty(env.n, device=device)
            for index, level in enumerate(level_of):
                _apply_level(env, index, level)
        teacher = None if args.teacher == "none" else PursuerGuidance(args.teacher, env.n, device)
        forecaster = BatchEPN(env.n, n_pursuers, device)
        max_steps = int(env.rules.settings["task"]["max_episode_steps"])
        episode_ids = torch.zeros(env.n, dtype=torch.long, device=device)
        next_seed = int(args.seed) * 100_000
        episode_seed = list(range(args.seed, args.seed + env.n))
        layout = LayoutCurriculum()
        layout.set_seed(args.seed)
        obs_dim = graph_observation(state, env, torch.zeros(env.n, n_pursuers, 15, device=device)).shape[-1]
        critic_dim = n_pursuers * obs_dim
        action_dim = 4
        actor = Actor(obs_dim, action_dim).to(device)
        if teacher is not None:
            actor.log_std.data.fill_(-2.5)
        reference = None
        anchor = None
        teacher_capture = 0.0
        rollbacks = 0
        rollback_streak = 0
        critic = Critic(critic_dim).to(device)
        # 演员和评论家分开反传、分开裁剪。价值损失的梯度不再把策略和 KL 锚一起缩小。
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
        bc_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        history = []
        best_capture = -1.0
        best_snap = None

        def emit(message: str) -> None:
            try:
                print(message, flush=True)
            except BrokenPipeError:
                pass
            with (args.output / "train.log").open("a") as handle:
                handle.write(message + "\n")

        def observe(state, progress):
            return graph_observation(state, env, forecaster.observe(state, progress, *radar_track(env)))

        def fit_teacher(obs_f, teacher_f, alive_f, epochs: int) -> float:
            mse = 0.0
            flat_count = obs_f.shape[0]
            for _epoch in range(epochs):
                order = torch.randperm(flat_count, device=device)
                for start in range(0, flat_count, args.minibatch):
                    batch = order[start:start + args.minibatch]
                    mean, _std = actor(obs_f[batch])
                    error = (pursuer_command(mean) - teacher_f[batch]).square().sum(-1)
                    mask = alive_f[batch]
                    bc_loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
                    bc_optimizer.zero_grad()
                    bc_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    bc_optimizer.step()
                    mse = float(bc_loss.detach())
            return mse

        replay = CommandReplay(524_288, n_pursuers, obs_dim, action_dim, device)
        val_replay = CommandReplay(65_536, n_pursuers, obs_dim, action_dim, device)
        successes = SuccessSteps(env.n)
        matched = False
        teacher_report = None
        reverts = 0

        def snapshot() -> dict:
            return {
                "actor": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
                "critic": {key: value.detach().cpu().clone() for key, value in critic.state_dict().items()},
            }

        def restore(snap: dict) -> None:
            actor.load_state_dict(snap["actor"])
            critic.load_state_dict(snap["critic"])
            actor_optimizer.state.clear()
            critic_optimizer.state.clear()
            bc_optimizer.state.clear()

        def fit_replay(steps: int) -> float:
            if replay.size < 8:
                return 0.0
            loss_value = 0.0
            batch_size = min(args.minibatch, replay.size)
            for _step in range(steps):
                obs_b, command_b, alive_b = replay.sample(batch_size)
                mean, _std = actor(obs_b)
                loss = _intent_nll(mean, actor.log_std, command_b, alive_b)
                bc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                bc_optimizer.step()
                actor.log_std.data.clamp_(-3.0, -2.2)
                loss_value = float(loss.detach())
            return loss_value

        def replay_mse(buffer: CommandReplay) -> float | None:
            if buffer.size < 8:
                return None
            with torch.no_grad():
                obs_b, command_b, alive_b = buffer.sample(min(1024, buffer.size))
                mean, _std = actor(obs_b)
                return float(_intent_mse(pursuer_command(mean), command_b, alive_b))

        def pursue_actor(_state, progress):
            mean, _std = actor(observe(_state, progress))
            return shield_pursuer_command(env, pursuer_command(mean))

        def pursue_teacher(_state, _progress):
            return teacher.command(env)

        def scored(seed_start: int, episodes: int) -> dict:
            nonlocal teacher_report
            saved_h = env.evader_horizontal.clone()
            saved_v = env.evader_vertical.clone()
            if evader_level == "aeg":
                horizontal, vertical = LEVEL_SPEED["medium"]
                env.evader_horizontal.fill_(horizontal)
                env.evader_vertical.fill_(vertical)
            actor_report = _evaluate_batches(
                env, forecaster, observe, evader_command, max_steps, pursue_actor, seed_start, episodes,
            )
            if teacher is not None:
                teacher_report = _evaluate_batches(
                    env, forecaster, observe, evader_command, max_steps, pursue_teacher, seed_start, episodes,
                )
                teacher.previous.zero_()
            env.evader_horizontal.copy_(saved_h)
            env.evader_vertical.copy_(saved_v)
            return actor_report

        completed = 0
        env_steps = 0
        update = 0
        dagger_origin = None
        rl_at_steps = None
        extend_note = False
        bc_trace: list[float] = []
        stalled = False
        last_bc = 1.0e9
        info = None
        rl_ready = False
        init_capture = None
        reference = None
        low_streak = 0
        best_monitor = -1.0
        while still_training(env_steps, args.imitation_max_steps, args.rl_steps, teacher is not None, rl_at_steps):
            obs_buf, raw_buf, logp_buf = [], [], []
            state_buf, reward_buf, done_buf, value_buf = [], [], [], []
            sequence_buf, pursuer_buf, evader_buf, episode_buf = [], [], [], []
            teacher_buf, alive_buf = [], []
            captures = 0
            static_hits = 0
            dynamic_hits = 0
            finished_steps = 0
            if teacher is None:
                beta, phase = 0.0, "rl"
            else:
                beta, started, phase = imitation_phase(
                    update, args.bc_warmup, args.bc_max, args.dagger_decay, dagger_origin, stalled, last_bc <= args.dagger_fit,
                )
                if started is not None and dagger_origin is None:
                    replay.freeze()
                    dagger_origin = started
                    emit(
                        f"[mappo] dagger steps={env_steps} bc={last_bc:.4f} "
                        f"rows={successes.kept} failure_rows={successes.failure_rows}"
                    )
            if phase == "rl" and teacher is not None and not rl_ready:
                if best_snap is not None:
                    restore(best_snap)
                import copy
                reference = copy.deepcopy(actor).eval()
                for parameter in reference.parameters():
                    parameter.requires_grad_(False)
                actor_optimizer.state.clear()
                critic_optimizer.state.clear()
                bc_optimizer.state.clear()
                rl_at_steps = env_steps
                rl_ready = True
                opening = scored(9_000_000, args.confirm_episodes)
                init_capture = opening["capture_rate"]
                best_capture = init_capture
                best_snap = snapshot()
                emit(f"[mappo] rl start steps={env_steps} init_capture={init_capture:.3f}")
            if teacher is not None and rl_at_steps is None and env_steps >= args.imitation_max_steps and not extend_note:
                extend_note = True
                emit(f"[mappo] imitation step ceiling steps={env_steps} bc={last_bc:.4f}")
            learner_buf = []
            for _ in range(args.rollout_steps):
                state = env.state()
                progress = 1.0 - env.rules.steps.float() / max_steps
                obs = observe(state, progress)
                joint = obs.flatten(1)
                with torch.no_grad():
                    mean, std = actor(obs)
                    normal = Normal(mean, std)
                    raw = mean if phase != "rl" else normal.sample()
                    logp = squash_log_prob(normal, raw)
                    value = critic(joint)
                preferred = executed = None
                if teacher is not None:
                    preferred, executed = teacher.propose(env)
                use_teacher = torch.rand(env.n, device=device) < beta if teacher is not None else torch.zeros(env.n, dtype=torch.bool, device=device)
                learner_intent = pursuer_command(mean if phase != "rl" else raw)
                learner_cmd = shield_pursuer_command(env, learner_intent)
                env_steps += int(env.n)
                command = torch.zeros(env.n, state.shape[1], action_dim, device=device)
                command[:, 0] = evader_command(env)
                if teacher is None:
                    command[:, 1:] = learner_cmd
                else:
                    command[:, 1:] = torch.where(use_teacher.view(env.n, 1, 1), executed, learner_cmd)
                learner_buf.append((~use_teacher).detach())
                if preferred is not None:
                    teacher_buf.append(preferred.detach())
                    alive_buf.append((~env.disabled[:, 1:]).detach())
                    for index in range(env.n):
                        successes.push_env(index, (
                            obs[index].detach(),
                            preferred[index].detach(),
                            (~env.disabled[index, 1:]).detach(),
                        ))
                team_reward, done, captured, info, outcome = hold_decision(env, command)
                obs_buf.append(obs)
                raw_buf.append(raw)
                logp_buf.append(logp)
                state_buf.append(joint)
                reward_buf.append(team_reward)
                done_buf.append(done.float())
                value_buf.append(value)
                sequence_buf.append(forecaster.history.detach().clone())
                pursuer_buf.append(state[:, 1:, :3].detach())
                evader_buf.append(state[:, 0, :3].detach())
                episode_buf.append(episode_ids.clone())
                captures += int(captured.sum())
                finished = done.nonzero(as_tuple=False).flatten()
                finished_steps += int(finished.numel())
                if finished.numel():
                    static_hits += int((outcome[finished] == 2).sum())
                    dynamic_hits += int((outcome[finished] == 3).sum())
                if finished.numel():
                    if preferred is not None:
                        successes.commit(
                            finished,
                            captured,
                            lambda index, rows: (val_replay if index % 8 == 0 else replay).add(*_stack_rows(rows)),
                        )
                    completed += int(finished.numel())
                    episode_ids[finished] += 1
                    forecaster.reset(finished)
                    if teacher is not None:
                        teacher.previous[finished] = 0
                    if evader_level == "aeg":
                        for index in finished.tolist():
                            aeg.observe(level_of[index], bool(captured[index].item()))
                            level_of[index] = aeg.next_level()
                            _apply_level(env, index, level_of[index])
                    next_seed = reset_with_curriculum(
                        env, finished, episode_seed, next_seed, captured, outcome, layout
                    )
            with torch.no_grad():
                boot_state = env.state()
                boot_progress = 1.0 - env.rules.steps.float() / max_steps
                last_value = critic(graph_observation(
                    boot_state, env, forecaster.forecast(boot_state, boot_progress, *radar_track(env))
                ).flatten(1))
                reward = torch.stack(reward_buf)
                value = torch.stack(value_buf)
                done_t = torch.stack(done_buf)
                advantage, returns = gae(reward, value, done_t, last_value)
                advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
            obs_all = torch.stack(obs_buf)
            raw_all = torch.stack(raw_buf)
            old_logp = torch.stack(logp_buf)
            state_all = torch.stack(state_buf)
            flat = args.rollout_steps * env.n
            obs_f = obs_all.reshape(flat, n_pursuers, obs_dim)
            raw_f = raw_all.reshape(flat, n_pursuers, action_dim)
            old_logp_f = old_logp.reshape(flat, n_pursuers)
            adv_f = advantage.reshape(flat, 1).expand_as(old_logp_f)
            ret_f = returns.reshape(flat)
            state_f = state_all.reshape(flat, critic_dim)
            bc_weight = 0.0
            bc_mse = 0.0
            learner_f = torch.stack(learner_buf).reshape(flat) if learner_buf else torch.zeros(flat, dtype=torch.bool, device=device)
            ppo_kl = 0.0
            ref_kl = 0.0
            if teacher is not None and phase != "rl" and replay.size >= 8:
                bc_weight = 1.0
                fit_replay(args.bc_steps if phase == "bc" else 16)
                held_val = replay_mse(val_replay)
                bc_mse = held_val if held_val is not None else replay_mse(replay) or 0.0
                last_bc = bc_mse
                bc_trace.append(last_bc)
                if phase == "bc" and not stalled and bc_stalled(bc_trace, args.dagger_fit):
                    stalled = True
                    emit(
                        f"[mappo] bc plateau val_mse={last_bc:.4f} best={min(bc_trace):.4f} "
                        f"rows={successes.kept} failure_rows={successes.failure_rows}"
                    )
            if phase == "rl" and bool(learner_f.any()):
                indexes = torch.randperm(flat, device=device)
                stop_early = False
                consecutive = 0
                kl_scale = reference_scale(env_steps, rl_at_steps, args.rl_steps)
                action_width = float(action_dim)
                for _epoch in range(args.epochs):
                    if stop_early:
                        break
                    for start in range(0, flat, args.minibatch):
                        batch = indexes[start:start + args.minibatch]
                        batch = batch[learner_f[batch]]
                        if batch.numel() < 8:
                            continue
                        mean, std = actor(obs_f[batch])
                        normal = Normal(mean, std)
                        new_logp = squash_log_prob(normal, raw_f[batch])
                        log_ratio = new_logp - old_logp_f[batch]
                        ratio = log_ratio.exp()
                        surrogate = torch.min(
                            ratio * adv_f[batch],
                            ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * adv_f[batch],
                        )
                        policy_loss = -surrogate.mean()
                        entropy = normal.entropy().sum(-1).mean()
                        actor_loss = policy_loss - args.entropy * entropy
                        approx = ((ratio - 1.0) - log_ratio).mean() / action_width
                        if reference is not None:
                            ref_mean, ref_std = reference(obs_f[batch])
                            ref_term = torch.distributions.kl_divergence(
                                normal, Normal(ref_mean.detach(), ref_std.detach())
                            ).mean()
                            actor_loss = actor_loss + args.kl_coef * kl_scale * ref_term
                            ref_kl = float(ref_term.detach())
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        actor_optimizer.step()
                        actor.log_std.data.clamp_(-3.0, -0.5)
                        value_scale = ret_f.detach().var().clamp_min(1.0)
                        value_loss = (critic(state_f[batch]) - ret_f[batch]).square().mean() / value_scale
                        critic_optimizer.zero_grad()
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                        critic_optimizer.step()
                        ppo_kl = float(approx.detach())
                        consecutive = consecutive + 1 if ppo_kl > args.kl_target else 0
                        if consecutive >= 2:
                            stop_early = True
                            break
            epn_mse = forecaster.update(
                torch.stack(sequence_buf),
                torch.stack(pursuer_buf),
                torch.stack(evader_buf),
                torch.stack(episode_buf),
            )
            row = {
                "update": update,
                "episodes": completed,
                "finished": finished_steps,
                "captures": captures,
                "capture_rate": captures / finished_steps if finished_steps else 0.0,
                "mean_reward": float(reward.mean()),
                "epn_mse": epn_mse,
                "evader_level": evader_level,
                "static_obstacles": static_count,
                "dynamic_obstacles": dynamic_count,
                "teacher": args.teacher,
                "phase": phase,
                "dagger_beta": beta,
                "bc_weight": bc_weight,
                "bc_mse": bc_mse,
                "reference_kl": ref_kl,
                "ppo_kl": ppo_kl,
                "failure_rows": successes.failure_rows,
                "dataset": replay.size,
                "env_steps": env_steps,
                "rollbacks": rollbacks,
                "static_collisions": static_hits,
                "dynamic_collisions": dynamic_hits,
                "collision_rate": (static_hits + dynamic_hits) / finished_steps if finished_steps else 0.0,
                "matched": matched,
                "aeg": evader_level == "aeg",
            }
            if evader_level == "aeg":
                rates = aeg.stats()["capture_rates"]
                row["aeg_level"] = aeg.current_level
                row["aeg_rates"] = {name: None if value != value else value for name, value in rates.items()}
            history.append(row)
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            emit(
                f"[mappo] update={update} episodes={completed}/{args.episodes} "
                f"phase={row['phase']} capture_rate={row['capture_rate']:.3f} reward={row['mean_reward']:.3f} "
                f"epn_mse={epn_mse:.4f} level={evader_level} obstacles={static_count}+{dynamic_count} "
                f"teacher={args.teacher} beta={beta:.2f} bc={bc_mse:.4f} rows={successes.kept} "
                f"failure_rows={successes.failure_rows} ppo_kl={ppo_kl:.4f} ref_kl={ref_kl:.4f} "
                f"aeg={row.get('aeg_level', evader_level)} "
                f"collisions={static_hits}+{dynamic_hits} matched={int(matched)}"
            )
            if (update + 1) % args.gate_interval == 0:
                report = scored(8_000_000, env.n)
                emit(
                    f"[mappo] monitor capture={report['capture_rate']:.3f} "
                    f"episodes={report['episodes']} collisions={report['static_collisions']}+{report['dynamic_collisions']}"
                )
                dropped = phase == "rl" and init_capture is not None and report["capture_rate"] + 0.08 < init_capture
                low_streak = low_streak + 1 if dropped else 0
                if low_streak >= 2 and best_snap is not None:
                    check = scored(9_000_000, args.confirm_episodes)
                    if check["capture_rate"] + 0.08 < init_capture:
                        restore(best_snap)
                        reverts += 1
                        low_streak = 0
                        emit(
                            f"[mappo] restore capture={check['capture_rate']:.3f} "
                            f"init={init_capture:.3f} best={best_capture:.3f}"
                        )
                    else:
                        low_streak = 0
                elif report["capture_rate"] > best_monitor:
                    best_monitor = report["capture_rate"]
                    check = scored(9_000_000, args.confirm_episodes)
                    if check["capture_rate"] > best_capture:
                        best_capture = check["capture_rate"]
                        best_snap = snapshot()
                        emit(f"[mappo] confirmed capture={best_capture:.3f} phase={phase} steps={env_steps}")
            update += 1
            if update % 10 == 0:
                torch.save(_payload(actor, critic, forecaster, history, evader_level, static_count, dynamic_count, args.teacher, args.scene, evader_level == "aeg"), args.output / "mappo.pt")
        torch.save(_payload(actor, critic, forecaster, history, evader_level, static_count, dynamic_count, args.teacher, args.scene, evader_level == "aeg"), args.output / "mappo.pt")
        emit(f"[mappo] wrote {args.output / 'mappo.pt'} episodes={completed}")
        if best_snap is not None:
            restore(best_snap)
        if evader_level == "aeg":
            for index in range(env.n):
                _apply_level(env, index, "medium")
        actor_report = _evaluate_batches(
            env, forecaster, observe, evader_command, max_steps, pursue_actor, 10_000_000, args.final_episodes,
        )
        held_teacher = None
        if teacher is not None:
            held_teacher = _evaluate_batches(
                env, forecaster, observe, evader_command, max_steps, pursue_teacher, 10_000_000, args.final_episodes,
            )
        evaluation = {
            **actor_report,
            "teacher": args.teacher,
            "perception": args.perception,
            "dagger": True,
            "teacher_eval": held_teacher,
            "control": "learner mean action, no teacher in the action, safety shield on",
            "eval_evader_level": "medium" if evader_level == "aeg" else evader_level,
            "checkpoint": "best capture",
            "best_capture": best_capture,
            "matched": matched,
        }
        (args.output / "eval.json").write_text(json.dumps(evaluation, indent=2))
        teacher_rate = 0.0 if held_teacher is None else held_teacher["capture_rate"]
        emit(
            f"[mappo] eval capture_rate={evaluation['capture_rate']:.3f} "
            f"captures={evaluation['captures']}/{evaluation['episodes']} "
            f"collisions={evaluation['static_collisions']}+{evaluation['dynamic_collisions']} "
            f"teacher={teacher_rate:.3f}"
        )
    except BaseException:
        import traceback
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        env.close()
        app.close()


def _payload(actor, critic, forecaster, history, evader_level, static_count, dynamic_count, teacher, scene, use_aeg=False) -> dict:
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "epn": forecaster.net.state_dict(),
        "history": history,
        "imitation": teacher != "none",
        "contract": {
            "action": ["vx", "vy", "vz", "yaw_rate"],
            "pursuer_horizontal": PURSUER_HORIZONTAL,
            "pursuer_vertical": PURSUER_VERTICAL,
            "yaw_rate": MAX_YAW_RATE,
            "decision_dt": 0.2,
            "gamma": GAMMA,
            "capture_radius": BODY_CONTACT,
            "capture_required": 1,
            "collision_penalty": COLLISION_PENALTY,
            "time_penalty": TIME_PENALTY,
            "epn": True,
            "evader_level": evader_level,
            "static_obstacles": static_count,
            "dynamic_obstacles": dynamic_count,
            "aeg": bool(use_aeg),
            "teacher": None if teacher == "none" else teacher,
            "scene": scene,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=5000, help="保留参数，停止条件改用环境步数")
    parser.add_argument("--max-episodes", type=int, default=20000, help="保留参数，停止条件改用环境步数")
    parser.add_argument("--imitation-max-steps", type=int, default=600_000)
    parser.add_argument("--rl-steps", type=int, default=5_000_000)
    parser.add_argument("--confirm-episodes", type=int, default=1024)
    parser.add_argument("--final-episodes", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.08)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--scene", choices=("full", "simple"), default="full")
    parser.add_argument("--teacher", choices=("none", "apf", "frpn", "orca", "cbf", "mpc"), default="none")
    parser.add_argument("--bc-warmup", type=int, default=20)
    parser.add_argument("--bc-max", type=int, default=48)
    parser.add_argument("--bc-steps", type=int, default=64)
    parser.add_argument("--gate-interval", type=int, default=8)
    parser.add_argument("--dagger-warmup", type=int, default=8)
    parser.add_argument("--dagger-decay", type=int, default=24)
    parser.add_argument("--dagger-fit", type=float, default=0.35)
    parser.add_argument("--dagger-latest", type=int, default=32, help="保留参数，阶段不再由它推动")
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--kl-target", type=float, default=0.03)
    parser.add_argument("--perception", choices=("truth", "mid360"), default="mid360")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 1024:
        parser.error("num-envs must be 1..1024")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    started = time.time()
    train(args)
    print(f"[mappo] elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
