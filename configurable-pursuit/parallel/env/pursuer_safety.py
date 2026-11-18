"""Pursuer velocity-level safety and diagnostics. Never writes physical positions/velocities.

Braking parameters are engineering hypotheses pending Hummingbird flight tests,
not a proof of invariance. Constraint residuals are explicit control failures.
"""
import torch
from parallel.env.geometry import segment_clearance


def unit(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-9)


class SafetyController:
    braking_acceleration = 1.5
    lag_seconds = .25
    boundary_margin = .25
    escape_seconds = 2.
    escape_speed = .6
    hard_static_margin = .1

    def __init__(self, n, device, dtype, *, count=3, static_count=16, radius=.3, north_boundary=10., dt=None, speed_limit=None):
        if dt is None:
            from parallel.env.config_loader import load_settings
            dt=load_settings()['simulation']['dt']
        self.dt=dt
        if speed_limit is None:
            from parallel.strategies.config import load_config
            speed_limit=load_config()[0]['spawn']['speed_max']
        self.speed_limit=speed_limit
        self.radius=radius
        self.north_boundary=north_boundary
        self.age = torch.zeros(n, count, static_count, device=device, dtype=torch.long)
        self.entry = torch.zeros(n, count, static_count, 2, device=device, dtype=dtype)
        self.previous_offset = torch.zeros_like(self.entry)
        self.active = torch.zeros_like(self.age, dtype=torch.bool)
        self.metrics = {}

    def reset(self, ids):
        for field in (self.age, self.entry, self.previous_offset, self.active):
            field[ids] = 0
        # Replace cached diagnostics rather than editing tensors already returned
        # in an older step's info; preserve unselected environments and snapshots.
        for name,value in self.metrics.items():
            cleared=value.clone()
            cleared[ids]=0
            self.metrics[name]=cleared

    def commands(self, pos, actual_velocity, nominal, static):
        offset = pos[:, :, None] - static[:, None, :, :2]
        distance = torch.linalg.vector_norm(offset, dim=-1)
        normal = unit(offset)
        normal = torch.where((distance < 1e-8)[..., None],
                             torch.tensor([1., 0.], device=pos.device), normal)
        clearance = distance - static[:, None, :, 2] - self.radius
        overlap = clearance <= 0
        entered = overlap & ~self.active
        old_normal = unit(self.previous_offset)
        old_normal = torch.where((self.previous_offset.square().sum(-1) < 1e-12)[..., None], normal, old_normal)
        self.entry = torch.where(entered[..., None], old_normal, self.entry)
        released = self.active & (clearance > .15)
        crossing = released & ((normal * self.entry).sum(-1) < 0)
        self.active = (self.active | entered) & ~released
        self.age = torch.where(self.active, self.age + 1, 0)
        timed_out = self.active & (self.age * self.dt > self.escape_seconds)
        escape = self.active.any(-1)
        # Return toward the entry side. Bounded duration: after timeout stop
        # pursuing and retain safety constraints; report the unresolved episode.
        escape_direction = unit((self.entry * self.active[..., None]).sum(-2))
        desired = torch.where(escape[..., None], escape_direction * self.escape_speed, nominal)
        desired = torch.where(timed_out.any(-1)[..., None], torch.zeros_like(desired), desired)
        cap = torch.where(escape, min(self.escape_speed,self.speed_limit), self.speed_limit)
        inward_speed = (-(normal * actual_velocity[:, :, None]).sum(-1)).clamp_min(0)
        braking = self.lag_seconds * inward_speed + inward_speed.square() / (2 * self.braking_acceleration)
        # Original .7m CBF remains the nominal preference. Dense scenes can
        # have overlapping preference bands: hard constraints use .1m clearance
        # plus measured-velocity braking, not mutually impossible soft margins.
        safe_gap = clearance - self.hard_static_margin - braking
        lower = (-safe_gap * 2.).clamp(-self.speed_limit, min(.6,self.speed_limit))
        near = clearance < (2.2 + braking)
        lower = torch.where(near, lower, torch.full_like(lower, -self.speed_limit))
        outward = actual_velocity[..., 1].clamp_min(0)
        gap = self.north_boundary - pos[..., 1] - self.boundary_margin
        available = gap - self.lag_seconds * outward - outward.square() / (2 * self.braking_acceleration)
        upper_y = (available * 2.).clamp(-self.speed_limit, self.speed_limit)
        # Joint cyclic projection onto ALL static halfplanes, boundary and disk.
        # Residual after finite iterations is exposed rather than called feasible.
        v = desired.clone()
        for _ in range(12):
            for j in range(static.shape[1]):
                a = normal[:, :, j]
                deficit = (lower[:, :, j] - (a * v).sum(-1)).clamp_min(0)
                v = v + deficit[..., None] * a
            v = v * (cap / torch.linalg.vector_norm(v, dim=-1).clamp_min(cap).clamp_min(1e-9))[..., None]
            # Enforce the boundary and speed disk jointly. Normalizing after
            # clamping Y can undo braking at Y=-cap when X is nonzero.
            # If upper_y < -cap the intersection is empty; keep the speed cap
            # and expose the unavoidable boundary residual below.
            feasible_upper = torch.maximum(upper_y, -cap)
            v[..., 1] = torch.minimum(v[..., 1], feasible_upper)
            x_limit = (cap.square()-v[..., 1].square()).clamp_min(0).sqrt()
            v[..., 0] = torch.maximum(torch.minimum(v[..., 0], x_limit), -x_limit)
        residual = (lower - (normal * v[:, :, None]).sum(-1)).clamp_min(0).amax(-1)
        residual = torch.maximum(residual, (v[..., 1] - upper_y).clamp_min(0))
        self.metrics = dict(overlap=overlap, overlap_entered=entered, overlap_depth=(-clearance).clamp_min(0),
            overlap_duration=self.age * self.dt, opposite_side_exit=crossing,
            escape_timeout=timed_out, escaping=escape, constraint_residual=residual,
            constraint_failure=residual > 1e-3)
        self.previous_offset = offset.clone()
        return v

    def observe_transition(self, previous, current, static):
        """Post-physics diagnostics also catch an entire pillar crossed in one step."""
        start = previous[:, 1:, None, :2] - static[:, None, :, :2]
        end = current[:, 1:, None, :2] - static[:, None, :, :2]
        radius = static[:, None, :, 2] + self.radius
        swept = segment_clearance(start, end, radius)
        outside = (torch.linalg.vector_norm(start, dim=-1) > radius) & (torch.linalg.vector_norm(end, dim=-1) > radius)
        through = outside & (swept <= 0) & ((start * end).sum(-1) < 0)
        return dict(p_boundary_violation=current[:, 1:, 1] > self.north_boundary,
                    p_boundary_excess=(current[:, 1:, 1] - self.north_boundary).clamp_min(0),
                    p_swept_overlap=swept <= 0, p_swept_depth=(-swept).clamp_min(0),
                    p_through_crossing=through, **self.metrics)


@torch.jit.script
def cbf(pos: torch.Tensor, velocity: torch.Tensor, static: torch.Tensor, radius: float = .3, max_speed: float = 2.) -> torch.Tensor:
    # TorchScript CUDA fusion cannot allocate its empty P intermediates.
    # Empty P has no constraints and keeps the exact empty command shape.
    if pos.shape[1] == 0: return velocity.clone()
    # Same sequential projection and clipping order using the configured constraint ordering.
    v = velocity.clone()
    for iteration in range(4):
        for j in range(static.shape[1]):
            offset = pos - static[:, j:j+1, :2]
            safe = static[:, j:j+1, 2] + radius + .7  # configured P radius + preferred margin
            d2 = (offset * offset).sum(-1)
            a = 2.0 * offset
            a2 = (a * a).sum(-1)
            b = -2.0 * (d2 - safe * safe)
            residual = b - (a * v).sum(-1)
            active = (d2 <= (safe + 1.5).square()) & (a2 > 1e-12) & (residual > 0)
            v = v + torch.where(active, residual / a2.clamp_min(1e-12), torch.zeros_like(residual)).unsqueeze(-1) * a
            length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            v = v * (max_speed / length.clamp_min(max_speed).clamp_min(1e-9))
    return v


