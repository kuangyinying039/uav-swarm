"""固定 ORCA 行为克隆参照，并检查学员独有的动态碰撞。

仿真真值只写进诊断字段，不进入策略观测。同一批场景种子分别飞学员和教师。
原先那 100 局和 19 局丢弃轨迹没有落盘，这里用已保存权重重新飞，并记下随机种子。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from collections import deque
from pathlib import Path

import torch

from parallel.env.mid360 import cylinder_visible, sensed_cylinders
from parallel.env.radar5_contract import BODY_RADIUS, MAX_YAW_RATE, PURSUER_HORIZONTAL, PURSUER_VERTICAL
from parallel.examples.bc_orca import NEAR_DISTANCE_M, TRAIN_EPISODES, _set_medium
from parallel.examples.radar5_baseline_eval import _build
from parallel.examples.radar5_guidance import (
    BODY_CONTACT,
    BRAKE_ACCEL,
    DECISION_DT,
    MAX_ALTITUDE,
    MIN_ALTITUDE,
    RESPONSE_TIME,
    SAFETY_BUFFER,
    PursuerGuidance,
    _brake_cylinders,
    _clear_columns,
    _clear_peers,
    _stay_inside,
    clip_speed,
    evader_command,
    shield_pursuer_command,
)
from parallel.training.epn import FUTURE_STEPS, HISTORY_STEPS, XY_SCALE, Z_SCALE, BatchEPN, radar_track
from parallel.training.hetero import (
    BLIND,
    K_OBSTACLES,
    OBSTACLE_DIM,
    PEER_DIM,
    SELF_DIM,
    TARGET_DIM,
    graph_observation,
    observation_slices,
)
from parallel.training.mappo import Actor, hold_decision, pursuer_command

SEED_START = 2_000_000
EPISODES = 100
TORCH_SEED = 20261119
COVERAGE_SEED = 20261120
HAZARD_M = 1.0
TIGHT_M = 0.5
WINDOW = 8
REPO = Path("/srv/data/projects/kyy/xieton/repro")
CHECKPOINT = REPO / "results/bc-orca-300/bc_orca.pt"
OUTPUT = REPO / "results/bc-orca-inspect"
CONSTRAINTS = (
    "none",
    "dynamic_track",
    "static_track",
    "boundary",
    "peer",
    "sweep_column",
    "sweep_peer",
    "stay_inside",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _round(value) -> float | None:
    number = float(value)
    if number != number or abs(number) > 1.0e5:
        return None
    return round(number, 4)


def _halfspace(velocity, normal, clearance, offset):
    limited = velocity.clone()
    delay = RESPONSE_TIME + DECISION_DT
    gap = (clearance - SAFETY_BUFFER).clamp(min=0.0)
    allowed = torch.sqrt((BRAKE_ACCEL * delay) ** 2 + 2.0 * BRAKE_ACCEL * gap) - BRAKE_ACCEL * delay
    push = ((SAFETY_BUFFER - clearance).clamp(min=0.0) / delay).clamp(max=0.3)
    lower = offset - allowed + push
    for _ in range(4):
        closing = (normal * limited[:, :, None, :]).sum(-1)
        violation = (lower - closing).clamp(min=0.0)
        limited = limited + (violation.unsqueeze(-1) * normal).sum(2)
        limited = clip_speed(limited, PURSUER_HORIZONTAL, PURSUER_VERTICAL)
    return limited


def _constraint(env, nominal: torch.Tensor) -> torch.Tensor:
    """每个追捕机上，安全层改动最大的那一类约束。0 表示几乎没改。"""
    state = env.state()
    points = state[:, 1:, :3]
    base = clip_speed(nominal[..., :3], PURSUER_HORIZONTAL, PURSUER_VERTICAL)
    centers, radii, motion, valid = _brake_cylinders(env, points.shape[1])
    codes = torch.zeros(points.shape[:2], dtype=torch.long, device=points.device)
    best = torch.zeros(points.shape[:2], device=points.device)
    if centers.shape[2]:
        delta = points[:, :, None, :2] - centers
        radial = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        outward = delta / radial.unsqueeze(-1)
        normal = torch.cat((outward, torch.zeros_like(points[:, :, None, 2:3]).expand(*outward.shape[:3], 1)), dim=-1)
        clearance = radial - radii - BODY_RADIUS
        if valid is not None:
            clearance = torch.where(valid, clearance, torch.full_like(clearance, 1.0e3))
        offset = (outward * motion).sum(-1)
        static_count = int(env.rules.static.shape[1])
        families = []
        if centers.shape[2] > static_count:
            keep = torch.zeros_like(clearance, dtype=torch.bool)
            keep[:, :, static_count:] = True
            families.append((1, clearance.masked_fill(~keep, 1.0e3)))
        if static_count:
            keep = torch.zeros_like(clearance, dtype=torch.bool)
            keep[:, :, :static_count] = True
            families.append((2, clearance.masked_fill(~keep, 1.0e3)))
        for code, masked in families:
            updated = _halfspace(base, normal, masked, offset)
            delta_v = torch.linalg.vector_norm(updated - base, dim=-1)
            replace = delta_v > best
            best = torch.where(replace, delta_v, best)
            codes = torch.where(replace, torch.full_like(codes, code), codes)
    half_x = float(env.rules.cfg["boundary_half_extent_x"])
    half_y = float(env.rules.cfg["boundary_half_extent_y"])
    walls = (
        ((1.0, 0.0, 0.0), points[..., 0] + half_x - BODY_RADIUS),
        ((-1.0, 0.0, 0.0), half_x - BODY_RADIUS - points[..., 0]),
        ((0.0, 1.0, 0.0), points[..., 1] + half_y - BODY_RADIUS),
        ((0.0, -1.0, 0.0), half_y - BODY_RADIUS - points[..., 1]),
        ((0.0, 0.0, 1.0), points[..., 2] - MIN_ALTITUDE),
        ((0.0, 0.0, -1.0), MAX_ALTITUDE - points[..., 2]),
    )
    wall_normals, wall_clear, wall_offset = [], [], []
    for direction, gap in walls:
        wall_normals.append(points.new_tensor(direction).view(1, 1, 1, 3).expand(*points.shape[:2], 1, 3))
        wall_clear.append(gap.unsqueeze(-1))
        wall_offset.append(torch.zeros_like(gap).unsqueeze(-1))
    updated = _halfspace(base, torch.cat(wall_normals, 2), torch.cat(wall_clear, 2), torch.cat(wall_offset, 2))
    delta_v = torch.linalg.vector_norm(updated - base, dim=-1)
    replace = delta_v > best
    best = torch.where(replace, delta_v, best)
    codes = torch.where(replace, torch.full_like(codes, 3), codes)
    if points.shape[1] > 1:
        delta = points[:, :, None, :] - points[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        normal = delta / distance.unsqueeze(-1)
        eye = torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
        clearance = torch.where(eye, torch.full_like(distance, 10.0), distance - BODY_CONTACT)
        updated = _halfspace(base, normal, clearance, torch.zeros_like(distance))
        delta_v = torch.linalg.vector_norm(updated - base, dim=-1)
        replace = delta_v > best
        best = torch.where(replace, delta_v, best)
        codes = torch.where(replace, torch.full_like(codes, 4), codes)
    sequential = base
    stages = []
    if centers.shape[2]:
        swept = _clear_columns(sequential, points, centers, radii, valid, PURSUER_HORIZONTAL, PURSUER_VERTICAL)
        stages.append((5, swept))
        sequential = swept
    if points.shape[1] > 1:
        swept = _clear_peers(sequential, points, PURSUER_HORIZONTAL, PURSUER_VERTICAL)
        stages.append((6, swept))
        sequential = swept
    inside = _stay_inside(sequential, points, env)
    stages.append((7, inside))
    previous = base
    for code, updated in stages:
        delta_v = torch.linalg.vector_norm(updated - previous, dim=-1)
        replace = delta_v > best
        best = torch.where(replace, delta_v, best)
        codes = torch.where(replace, torch.full_like(codes, code), codes)
        previous = updated
    return torch.where(best > 0.05, codes, torch.zeros_like(codes))


def _true_disks(env):
    state = env.state()
    xy = state[:, 1:, :2]
    centers = env.rules.bg
    radii = env.rules.radii[:, env.rules.layout.dynamic_objects]
    velocity = env.rules.bgvel
    if velocity.shape[-1] != 2:
        velocity = velocity[..., :2]
    delta = xy[:, :, None, :] - centers[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    gap = distance - radii[:, None, :] - BODY_RADIUS
    return state, gap, centers, radii, velocity


def _closing(velocity_xy: torch.Tensor, origin_xy: torch.Tensor, center_xy: torch.Tensor) -> torch.Tensor:
    delta = center_xy - origin_xy
    return (velocity_xy * delta).sum(-1) / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)


def _observation_table() -> dict:
    slices = observation_slices(3)
    return {
        "width": slices["obstacle"][1],
        "nodes": {
            "self": ["vx/1.48", "vy/1.48", "vz/0.8", "yaw/pi", "height/12", "alive", "target_detected", "id_0", "id_1", "id_2"],
            "target": ["relative_xyz/20_20_12 or -5 if blind", "target_velocity/1.52_1.52_0.88", "epn_forecast_15", "detected"],
            "peer": "two teammates: relative_xyz/20_20_12, velocity/own limits, alive",
            "obstacle": "three nearest valid tracks: relative_xy/20, gap/5, radius/2, moving_flag, seen",
        },
        "slices": {name: {"start": start, "stop": stop, "node_width": width} for name, (start, stop, width) in slices.items()},
        "not_in_policy_input": [
            "obstacle velocity",
            "raw world position",
            "mission goal",
            "true future evader position",
            "unseen cylinders",
        ],
        "seen_by_teacher_and_shield_but_not_student": ["obstacle track velocity", "all valid tracks, not only the nearest three, inside the shield"],
        "scales": {"self": SELF_DIM, "target": TARGET_DIM, "peer": PEER_DIM, "obstacle": OBSTACLE_DIM, "obstacle_slots": K_OBSTACLES, "blind": BLIND},
        "epn": {"history": HISTORY_STEPS, "future": FUTURE_STEPS, "xy_scale_m": XY_SCALE, "z_scale_m": Z_SCALE},
    }


def _snapshot(env, nominal, executed, teacher_preferred, previous, gap, nearest, centers, radii, velocity, codes) -> list[list[dict]]:
    state, _, _, _, _ = _true_disks(env)
    tracker = env.mid360
    sensed_center, sensed_radius, sensed_velocity, sensed_valid, _moving = sensed_cylinders(env)
    static_count = int(env.rules.static.shape[1])
    origin = state[:, 1:, :3]
    visible = cylinder_visible(origin, torch.cat((env.rules.static[..., :2], centers), 1), torch.cat((env.rules.static[..., 2], radii), 1))
    age = tracker.time[:, None, None] - tracker.obstacle_last if tracker.obstacle_count else None
    body = state[:, 1:, 7:10]
    alive = ~env.disabled[:, 1:]
    sensed_gap = None
    top3 = None
    if sensed_center.shape[2]:
        radial = torch.linalg.vector_norm(origin[:, :, None, :2] - sensed_center, dim=-1)
        sensed_gap = radial - sensed_radius - BODY_RADIUS
        ranked = sensed_gap.masked_fill(~sensed_valid, 1.0e6)
        take = min(3, int(sensed_center.shape[2]))
        top3 = torch.topk(ranked, take, dim=-1, largest=False).indices
    rows = []
    gap_cpu = gap.detach().cpu()
    near_cpu = nearest.detach().cpu()
    origin_cpu = origin.detach().cpu()
    center_cpu = centers.detach().cpu()
    radius_cpu = radii.detach().cpu()
    velocity_cpu = velocity.detach().cpu()
    body_cpu = body.detach().cpu()
    alive_cpu = alive.detach().cpu()
    nominal_cpu = nominal.detach().cpu()
    shield_cpu = executed.detach().cpu()
    teacher_cpu = teacher_preferred.detach().cpu()
    previous_cpu = previous.detach().cpu()
    code_cpu = codes.detach().cpu()
    visible_cpu = visible.detach().cpu()
    valid_cpu = sensed_valid.detach().cpu()
    age_cpu = age.detach().cpu() if age is not None else None
    sensed_center_cpu = sensed_center.detach().cpu()
    sensed_velocity_cpu = sensed_velocity.detach().cpu()
    sensed_gap_cpu = sensed_gap.detach().cpu() if sensed_gap is not None else None
    top3_cpu = top3.detach().cpu() if top3 is not None else None
    for env_index in range(env.n):
        pursuers = []
        for agent in range(3):
            disk = int(near_cpu[env_index, agent])
            slot = static_count + disk
            center = center_cpu[env_index, disk]
            true_gap = float(gap_cpu[env_index, agent, disk])
            nominal_v = nominal_cpu[env_index, agent]
            shield_v = shield_cpu[env_index, agent]
            teacher_v = teacher_cpu[env_index, agent]
            body_v = body_cpu[env_index, agent]
            previous_v = previous_cpu[env_index, agent]
            tracked = bool(valid_cpu[env_index, agent, slot].item()) if valid_cpu.shape[2] > slot else False
            in_three = bool((top3_cpu[env_index, agent] == slot).any().item()) if top3_cpu is not None and tracked else False
            track_age = None
            sensed_position = None
            sensed_motion = None
            sensed_gap_m = None
            if tracked and age_cpu is not None:
                track_age = _round(age_cpu[env_index, agent, slot])
                sensed_position = [_round(sensed_center_cpu[env_index, agent, slot, 0]), _round(sensed_center_cpu[env_index, agent, slot, 1])]
                sensed_motion = [_round(sensed_velocity_cpu[env_index, agent, slot, 0]), _round(sensed_velocity_cpu[env_index, agent, slot, 1])]
                sensed_gap_m = _round(sensed_gap_cpu[env_index, agent, slot])
            origin_xy = origin_cpu[env_index, agent, :2]
            pursuers.append({
                "alive": bool(alive_cpu[env_index, agent].item()),
                "true_gap_m": _round(true_gap),
                "true_disk_index": disk,
                "true_disk_radius_m": _round(radius_cpu[env_index, disk]),
                "true_disk_velocity_m_s": [_round(velocity_cpu[env_index, disk, 0]), _round(velocity_cpu[env_index, disk, 1])],
                "true_relative_position_m": [_round(center[0] - origin_xy[0]), _round(center[1] - origin_xy[1])],
                "geometrically_visible": bool(visible_cpu[env_index, agent, slot].item()),
                "track_valid": tracked,
                "track_age_s": track_age,
                "in_nearest_three": in_three,
                "sensed_position_m": sensed_position,
                "sensed_velocity_m_s": sensed_motion,
                "sensed_gap_m": sensed_gap_m,
                "nominal_velocity_m_s": [_round(nominal_v[0]), _round(nominal_v[1]), _round(nominal_v[2])],
                "shield_velocity_m_s": [_round(shield_v[0]), _round(shield_v[1]), _round(shield_v[2])],
                "body_velocity_m_s": [_round(body_v[0]), _round(body_v[1]), _round(body_v[2])],
                "teacher_velocity_m_s": [_round(teacher_v[0]), _round(teacher_v[1]), _round(teacher_v[2])],
                "nominal_yaw_rate_rad_s": _round(nominal_v[3]),
                "teacher_yaw_rate_rad_s": _round(teacher_v[3]),
                "shield_change_m_s": _round(torch.linalg.vector_norm(shield_v[:3] - nominal_v[:3])),
                "constraint": CONSTRAINTS[int(code_cpu[env_index, agent].item())],
                "nominal_closing_m_s": _round(_closing(nominal_v[:2], origin_xy, center)),
                "shield_closing_m_s": _round(_closing(shield_v[:2], origin_xy, center)),
                "body_closing_m_s": _round(_closing(body_v[:2], origin_xy, center)),
                "teacher_closing_m_s": _round(_closing(teacher_v[:2], origin_xy, center)),
                "previous_shield_closing_m_s": _round(_closing(previous_v[:2], origin_xy, center)),
            })
        rows.append(pursuers)
    return rows


def _classify(window: list[dict]) -> dict:
    last = min(window[-1], key=lambda row: row["true_gap_m"] if row["true_gap_m"] is not None else 1.0e6)
    tracked = bool(last["track_valid"])
    top3 = bool(last["in_nearest_three"])
    nominal = last["nominal_closing_m_s"] or 0.0
    teacher = last["teacher_closing_m_s"] or 0.0
    shield = last["shield_closing_m_s"] or 0.0
    body = last["body_closing_m_s"] or 0.0
    previous = last["previous_shield_closing_m_s"] or 0.0
    if not tracked or not top3:
        label = "input_missing"
    elif nominal > teacher + 0.2:
        label = "policy_shift"
    elif shield > nominal + 0.15 or body > previous + 0.2:
        label = "execution"
    elif teacher > 0.25:
        label = "teacher_unreliable"
    else:
        label = "not_separated"
    return {"primary": label, "closest_pursuer_at_last_decision": last}


def _error_bucket(store, name, nominal, teacher, alive, gap):
    error = nominal[..., :4] - teacher[..., :4]
    mask = alive & torch.isfinite(gap)
    if name == "tight":
        mask = mask & (gap < TIGHT_M)
    elif name == "approach":
        mask = mask & (gap >= TIGHT_M) & (gap < HAZARD_M)
    else:
        mask = mask & (gap >= HAZARD_M)
    weight = mask.float()
    store[name]["count"] += float(weight.sum())
    store[name]["square"] += (error.square() * weight.unsqueeze(-1)).sum(dim=(0, 1))


def _roll(env, forecaster, observe, actor, teacher: PursuerGuidance, seeds: list[int], detailed: bool):
    device = env.device
    _set_medium(env)
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    env.reset(torch.arange(env.n, device=device), seeds=seeds)
    forecaster.reset(torch.arange(env.n, device=device))
    teacher.reset()
    outcome = torch.zeros(env.n, dtype=torch.long, device=device)
    ever_dynamic = torch.zeros(env.n, dtype=torch.bool, device=device)
    ever_static = torch.zeros(env.n, dtype=torch.bool, device=device)
    min_gap = torch.full((env.n,), float("inf"), device=device)
    max_steps = int(env.rules.settings["task"]["max_episode_steps"])
    from parallel.env.radar5_contract import physics_substeps

    substeps = physics_substeps(float(env.rules.settings["simulation"]["dt"]))
    rings = [deque(maxlen=WINDOW) for _ in range(env.n)]
    saved = {}
    buckets = {
        name: {"count": 0.0, "square": torch.zeros(4, device=device)}
        for name in ("tight", "approach", "ordinary")
    }
    buckets["hazard_tracked"] = 0.0
    buckets["hazard_top3"] = 0.0
    buckets["hazard_steps"] = 0.0
    buckets["hazard_yaw_large"] = 0.0
    buckets["hazard_yaw_small"] = 0.0
    buckets["hazard_yaw_large_tracked"] = 0.0
    buckets["hazard_yaw_small_tracked"] = 0.0
    previous = torch.zeros(env.n, 3, 4, device=device)
    steps = 0
    label = "student" if detailed else "teacher"
    while steps < max_steps and not bool((outcome != 0).all()):
        open_now = outcome == 0
        state, gap, centers, radii, disk_velocity = _true_disks(env)
        nearest = gap.argmin(-1)
        nearest_gap = gap.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        min_gap = torch.where(open_now, torch.minimum(min_gap, nearest_gap.amin(-1)), min_gap)
        progress = 1.0 - env.rules.steps.float() / max_steps
        with torch.no_grad():
            if detailed:
                mean, _std = actor(observe(state, progress))
                nominal = pursuer_command(mean)
                executed = shield_pursuer_command(env, nominal)
                preferred, _teacher_executed = teacher.propose(env)
            else:
                preferred, executed = teacher.propose(env)
                nominal = preferred
        if detailed and bool(open_now.any()):
            alive = ~env.disabled[:, 1:]
            _error_bucket(buckets, "tight", nominal, preferred, alive & open_now[:, None], nearest_gap)
            _error_bucket(buckets, "approach", nominal, preferred, alive & open_now[:, None], nearest_gap)
            _error_bucket(buckets, "ordinary", nominal, preferred, alive & open_now[:, None], nearest_gap)
            codes = _constraint(env, nominal)
            rows = _snapshot(env, nominal, executed, preferred, previous, gap, nearest, centers, radii, disk_velocity, codes)
            hazard = open_now & (nearest_gap < HAZARD_M).any(-1)
            for index in hazard.nonzero(as_tuple=False).flatten().tolist():
                closest = min(range(3), key=lambda agent: rows[index][agent]["true_gap_m"] or 1.0e6)
                row = rows[index][closest]
                buckets["hazard_steps"] += 1.0
                buckets["hazard_tracked"] += float(row["track_valid"])
                buckets["hazard_top3"] += float(row["in_nearest_three"])
                large = abs((row["nominal_yaw_rate_rad_s"] or 0.0) - (row["teacher_yaw_rate_rad_s"] or 0.0)) > 0.5
                buckets["hazard_yaw_large" if large else "hazard_yaw_small"] += 1.0
                key = "hazard_yaw_large_tracked" if large else "hazard_yaw_small_tracked"
                buckets[key] += float(row["track_valid"] and row["in_nearest_three"])
            for index in open_now.nonzero(as_tuple=False).flatten().tolist():
                rings[index].append({"decision": int(env.rules.steps[index].item()) // 10, "pursuers": rows[index]})
        command = torch.zeros(env.n, state.shape[1], 4, device=device, dtype=state.dtype)
        command[:, 0] = evader_command(env)
        command[:, 1:] = executed
        command[outcome != 0] = 0
        previous = executed.detach()
        for _ in range(substeps):
            if steps >= max_steps:
                break
            _obs, _reward, terminated, truncated, info = env.step(command)
            steps += 1
            fresh = (terminated | truncated) & (outcome == 0)
            ever_dynamic |= info["dynamic_collision"] & (outcome == 0)
            ever_static |= info["static_collision"] & (outcome == 0)
            ended = fresh & (info["outcome"].to(outcome.dtype) == 3)
            if detailed and bool(ended.any()):
                for index in ended.nonzero(as_tuple=False).flatten().tolist():
                    saved[index] = list(rings[index])
            outcome = torch.where(fresh, info["outcome"].to(outcome.dtype), outcome)
            if bool((outcome != 0).all()):
                break
        if steps % 500 == 0:
            print(f"[inspect] {label} physics_steps={steps} finished={int((outcome != 0).sum())}/{env.n}", flush=True)
    return {
        "outcome": outcome.detach().cpu(),
        "ever_dynamic": ever_dynamic.detach().cpu(),
        "ever_static": ever_static.detach().cpu(),
        "min_gap": min_gap.detach().cpu(),
        "windows": saved,
        "buckets": buckets,
    }


def _rmse(square: torch.Tensor, count: float) -> dict:
    if count <= 0:
        return {"samples": 0}
    mean = square / count
    return {
        "samples": int(count),
        "horizontal_rmse_m_s": round(float(mean[:2].mean().sqrt()), 4),
        "vertical_rmse_m_s": round(float(mean[2].sqrt()), 4),
        "yaw_rmse_rad_s": round(float(mean[3].sqrt()), 4),
    }


def _coverage(env, teacher: PursuerGuidance, forecaster: BatchEPN):
    device = env.device
    torch.manual_seed(COVERAGE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(COVERAGE_SEED)
    _set_medium(env)
    teacher.reset()
    env.reset(seeds=list(range(1000, 1000 + env.n)))
    forecaster.reset(torch.arange(env.n, device=device))
    min_distance = torch.full((env.n,), float("inf"), device=device)
    min_disk = torch.full((env.n,), float("inf"), device=device)
    hazard_steps = torch.zeros(env.n, device=device)
    tight_steps = torch.zeros(env.n, device=device)
    decisions = torch.zeros(env.n, device=device)
    unseen_hazard = torch.zeros(env.n, device=device)
    next_seed = 1000 + env.n
    kept = discarded = 0
    rows = []
    while kept < TRAIN_EPISODES:
        state = env.state()
        _state, gap, _centers, _radii, _velocity = _true_disks(env)
        nearest_gap = gap.amin(-1)
        alive = ~env.disabled[:, 1:]
        nearest_gap = nearest_gap.masked_fill(~alive, float("inf"))
        episode_gap = nearest_gap.amin(-1)
        min_disk = torch.minimum(min_disk, episode_gap)
        distance = torch.linalg.vector_norm(state[:, 1:, :3] - state[:, :1, :3], dim=-1).masked_fill(~alive, float("inf"))
        min_distance = torch.minimum(min_distance, distance.amin(-1))
        open_hazard = episode_gap < HAZARD_M
        decisions += 1
        hazard_steps += open_hazard.float()
        tight_steps += (episode_gap < TIGHT_M).float()
        centers, _radius, _motion, valid, _moving = sensed_cylinders(env)
        static_count = int(env.rules.static.shape[1])
        disk_index = gap.argmin(-1)
        slot = static_count + disk_index
        tracked = valid.gather(2, slot.unsqueeze(-1)).squeeze(-1) if valid.shape[2] else torch.zeros_like(alive)
        sensed_gap = torch.linalg.vector_norm(state[:, 1:, None, :2] - centers, dim=-1) - _radius - BODY_RADIUS
        if int(centers.shape[2]) == 0:
            in_three = torch.zeros_like(alive)
        else:
            ranked = sensed_gap.masked_fill(~valid, 1.0e6)
            top = torch.topk(ranked, k=min(3, int(centers.shape[2])), dim=-1, largest=False).indices
            in_three = (top == slot.unsqueeze(-1)).any(-1) & tracked
        relevant = alive & (nearest_gap < HAZARD_M)
        unseen_hazard += (relevant & ~in_three).any(-1).float()
        preferred, executed = teacher.propose(env)
        command = torch.zeros(env.n, state.shape[1], 4, device=device, dtype=state.dtype)
        command[:, 0] = evader_command(env)
        command[:, 1:] = executed
        _reward, done, captured, _info, _outcome = hold_decision(env, command)
        finished = done.nonzero(as_tuple=False).flatten()
        if not finished.numel():
            continue
        for index in finished.tolist():
            close = float(min_distance[index])
            qualified = bool(captured[index].item()) or close <= NEAR_DISTANCE_M
            kind = "discarded"
            if qualified:
                kind = "success" if bool(captured[index].item()) else "near_success"
                kept += 1
            else:
                discarded += 1
            rows.append({
                "kind": kind,
                "min_evader_distance_m": round(close, 4),
                "min_dynamic_gap_m": _round(min_disk[index]),
                "decisions": int(decisions[index]),
                "hazard_decisions": int(hazard_steps[index]),
                "tight_decisions": int(tight_steps[index]),
                "hazard_decisions_without_disk_in_top3": int(unseen_hazard[index]),
            })
            min_distance[index] = float("inf")
            min_disk[index] = float("inf")
            hazard_steps[index] = 0
            tight_steps[index] = 0
            decisions[index] = 0
            unseen_hazard[index] = 0
        teacher.previous[finished] = 0
        seeds = list(range(next_seed, next_seed + int(finished.numel())))
        next_seed += int(finished.numel())
        env.reset(finished, seeds=seeds)
        _set_medium(env)
        if kept % 50 == 0 and kept:
            print(f"[inspect] coverage kept={kept} discarded={discarded}", flush=True)
    return rows


def _summarize_coverage(rows: list[dict]) -> dict:
    def pack(chosen):
        count = len(chosen)
        hazard_eps = sum(row["hazard_decisions"] > 0 for row in chosen)
        tight_eps = sum(row["tight_decisions"] > 0 for row in chosen)
        decisions = sum(row["decisions"] for row in chosen)
        hazard = sum(row["hazard_decisions"] for row in chosen)
        tight = sum(row["tight_decisions"] for row in chosen)
        unseen = sum(row["hazard_decisions_without_disk_in_top3"] for row in chosen)
        return {
            "episodes": count,
            "episodes_with_dynamic_gap_below_1m": hazard_eps,
            "episodes_with_dynamic_gap_below_0_5m": tight_eps,
            "hazard_decision_ratio": round(hazard / decisions, 4) if decisions else None,
            "tight_decision_ratio": round(tight / decisions, 4) if decisions else None,
            "hazard_decisions_missing_top3_ratio": round(unseen / hazard, 4) if hazard else None,
        }

    qualified = [row for row in rows if row["kind"] != "discarded"]
    discarded = [row for row in rows if row["kind"] == "discarded"]
    return {"qualified": pack(qualified), "discarded": pack(discarded), "episodes": rows}


def main() -> None:
    app, env = _build(EPISODES, SEED_START, OUTPUT, "mid360", "one_side")
    try:
        device = env.device
        forecaster = BatchEPN(env.n, 3, device)
        blob = torch.load(CHECKPOINT, map_location=device)
        actor = Actor(64, 4).to(device)
        actor.load_state_dict(blob["actor"])
        forecaster.net.load_state_dict(blob["epn"])
        actor.eval()
        teacher = PursuerGuidance("orca", env.n, device)

        def observe(state, progress):
            return graph_observation(state, env, forecaster.observe(state, progress, *radar_track(env)))

        manifest = {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "git_head": _git_head(),
            "git_note": "bc_orca.py is an untracked local file and is not inside this commit",
            "bc_script_sha256": _sha256(REPO / "configurable-pursuit/parallel/examples/bc_orca.py"),
            "inspect_script_sha256": _sha256(Path(__file__).resolve()),
            "scene_seeds": list(range(SEED_START, SEED_START + EPISODES)),
            "torch_seed_each_rollout": TORCH_SEED,
            "coverage_torch_seed": COVERAGE_SEED,
            "evader": "medium 1.46/0.84",
            "perception": "mid360",
            "formation": "one_side",
            "scene": "full",
            "observation": _observation_table(),
            "ground_truth_fields": [
                "true_gap_m",
                "true_disk_index",
                "true_disk_radius_m",
                "true_disk_velocity_m_s",
                "true_relative_position_m",
                "geometrically_visible",
                "nominal_closing_m_s",
                "shield_closing_m_s",
                "body_closing_m_s",
                "teacher_closing_m_s",
            ],
            "note": "The original 64/59 paired run and the 19 discarded episodes were not saved. This folder is the traceable replay.",
        }
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        seeds = manifest["scene_seeds"]
        print("[inspect] student rollout", flush=True)
        student = _roll(env, forecaster, observe, actor, teacher, seeds, detailed=True)
        print("[inspect] teacher rollout", flush=True)
        reference = _roll(env, forecaster, observe, actor, teacher, seeds, detailed=False)
        names = {0: "ongoing", 1: "captured", 2: "static", 3: "dynamic", 4: "boundary", 5: "physical", 6: "goal", 7: "timeout"}
        paired = []
        cases = []
        for index in range(EPISODES):
            row = {
                "seed": SEED_START + index,
                "student_outcome": names[int(student["outcome"][index])],
                "teacher_outcome": names[int(reference["outcome"][index])],
                "student_ever_dynamic": bool(student["ever_dynamic"][index]),
                "teacher_ever_dynamic": bool(reference["ever_dynamic"][index]),
                "student_ever_static": bool(student["ever_static"][index]),
                "teacher_ever_static": bool(reference["ever_static"][index]),
                "student_min_dynamic_gap_m": _round(student["min_gap"][index]),
                "teacher_min_dynamic_gap_m": _round(reference["min_gap"][index]),
            }
            paired.append(row)
            teacher_clean = not row["teacher_ever_dynamic"] and not row["teacher_ever_static"]
            if row["student_outcome"] == "dynamic" and teacher_clean and index in student["windows"]:
                window = student["windows"][index]
                cases.append({"seed": row["seed"], "classification": _classify([step["pursuers"] for step in window][-1:]), "window": window})
        segments = {
            name: _rmse(student["buckets"][name]["square"], student["buckets"][name]["count"])
            for name in ("tight", "approach", "ordinary")
        }
        hazard_steps = student["buckets"]["hazard_steps"]
        yaw_large = student["buckets"]["hazard_yaw_large"]
        yaw_small = student["buckets"]["hazard_yaw_small"]
        segments["hazard_visibility"] = {
            "decisions_with_dynamic_gap_below_1m": int(hazard_steps),
            "nearest_disk_tracked_ratio": round(student["buckets"]["hazard_tracked"] / hazard_steps, 4) if hazard_steps else None,
            "nearest_disk_in_top3_ratio": round(student["buckets"]["hazard_top3"] / hazard_steps, 4) if hazard_steps else None,
            "yaw_error_above_0_5": {
                "samples": int(yaw_large),
                "tracked_and_top3_ratio": round(student["buckets"]["hazard_yaw_large_tracked"] / yaw_large, 4) if yaw_large else None,
            },
            "yaw_error_at_most_0_5": {
                "samples": int(yaw_small),
                "tracked_and_top3_ratio": round(student["buckets"]["hazard_yaw_small_tracked"] / yaw_small, 4) if yaw_small else None,
            },
        }
        print("[inspect] coverage collection", flush=True)
        coverage_rows = _coverage(env, teacher, forecaster)
        coverage = _summarize_coverage(coverage_rows)
        labels = {}
        for case in cases:
            labels[case["classification"]["primary"]] = labels.get(case["classification"]["primary"], 0) + 1
        report = {
            "paired_definition": "student final outcome is dynamic collision, and the teacher never hit a static or dynamic obstacle on the same scene seed",
            "paired_cases": len(cases),
            "classification_counts": labels,
            "student_dynamic_outcomes": sum(row["student_outcome"] == "dynamic" for row in paired),
            "teacher_dynamic_outcomes": sum(row["teacher_outcome"] == "dynamic" for row in paired),
            "student_captures": sum(row["student_outcome"] == "captured" for row in paired),
            "teacher_captures": sum(row["teacher_outcome"] == "captured" for row in paired),
            "segments": segments,
            "coverage_summary": {"qualified": coverage["qualified"], "discarded": coverage["discarded"]},
        }
        (OUTPUT / "paired_episodes.json").write_text(json.dumps(paired, indent=2))
        (OUTPUT / "pre_collision.json").write_text(json.dumps({
            "schema": {
                "policy_input": ["track_valid", "track_age_s", "in_nearest_three", "sensed_position_m", "sensed_velocity_m_s", "sensed_gap_m"],
                "commands": ["nominal_velocity_m_s", "shield_velocity_m_s", "body_velocity_m_s", "teacher_velocity_m_s"],
                "shield": ["shield_change_m_s", "constraint"],
                "ground_truth_not_policy_input": manifest["ground_truth_fields"],
            },
            "cases": cases,
        }, indent=2))
        (OUTPUT / "segment_errors.json").write_text(json.dumps(segments, indent=2))
        (OUTPUT / "coverage.json").write_text(json.dumps(coverage, indent=2))
        (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        (OUTPUT / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
