"""当前逃逸任务的事件检测与终局规则；不计算奖励、不持有步数。"""
from dataclasses import dataclass
import torch
from parallel.env.geometry import segment_clearance
from parallel.env.config_loader import load_settings, EVENT_NAMES


def norm(x):
    return torch.linalg.vector_norm(x,dim=-1,keepdim=True)


def _column_contact(previous_xy, xy, static, body):
    """每架飞机的水平轨迹是否扫过竖直圆柱。圆柱贯穿飞行高度，从上方绕不过去。"""
    if static.shape[1] == 0:
        return torch.zeros(xy.shape[:2], dtype=torch.bool, device=xy.device)
    start = previous_xy[:, :, None, :] - static[:, None, :, :2]
    end = xy[:, :, None, :] - static[:, None, :, :2]
    return (segment_clearance(start, end, static[:, None, :, 2] + body) <= 0).any(-1)


def _moving_contact(previous_xy, xy, previous_centers, centers, radii, body):
    """每架飞机是否在这一步扫到运动障碍。"""
    if centers.shape[1] == 0:
        return torch.zeros(xy.shape[:2], dtype=torch.bool, device=xy.device)
    start = previous_xy[:, :, None, :] - previous_centers[:, None, :, :]
    end = xy[:, :, None, :] - centers[:, None, :, :]
    return (segment_clearance(start, end, radii[:, None, :] + body) <= 0).any(-1)


def _sphere_sweep(previous, current, other_previous, other_current, radius):
    """两段三维轨迹的最小中心距减去半径和。previous [N,3]，other [N,P,3]。"""
    start = previous[:, None, :] - other_previous
    end = current[:, None, :] - other_current
    delta = end - start
    denom = delta.square().sum(-1).clamp_min(1e-12)
    closest = start + ((-(start * delta).sum(-1) / denom).clamp(0, 1).unsqueeze(-1) * delta)
    return torch.linalg.vector_norm(closest, dim=-1) - radius


def _body_contact(previous, current, active, radius):
    """仍在飞行的飞机之间，这一步的三维扫掠是否贴到机身。"""
    start = previous[:, :, None, :] - previous[:, None, :, :]
    end = current[:, :, None, :] - current[:, None, :, :]
    delta = end - start
    denom = delta.square().sum(-1).clamp_min(1e-12)
    closest = start + ((-(start * delta).sum(-1) / denom).clamp(0, 1).unsqueeze(-1) * delta)
    count = current.shape[1]
    eye = torch.eye(count, device=current.device, dtype=torch.bool)
    hit = torch.linalg.vector_norm(closest, dim=-1) <= radius
    hit = hit & ~eye & active[:, :, None] & active[:, None, :]
    return hit.any(-1)


@dataclass
class TaskResult:
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict
    distance: torch.Tensor
    static_clearance: torch.Tensor
    dynamic_clearance: torch.Tensor


def detect_events(state, *, goal_position, static, background, radii,
                  previous_state, previous_bg, diagnostic_state, flight_mode,
                  config=None, evader_radius=.3, boundary_half_extent=(10.5,13.),
                  active=None, goal_height=None):
    """纯计算：使用相对扫掠圆检测，不修改输入张量。"""
    if not torch.isfinite(state).all() or not torch.isfinite(diagnostic_state).all():
        raise RuntimeError('Nonfinite physics state, including terminal states')
    config=load_settings()["task"] if config is None else config
    if active is None:
        active=torch.ones(state.shape[:2],dtype=torch.bool,device=state.device)
    pos=state[:,0,:2]
    distance=norm(goal_position-pos).squeeze(-1)
    bounds=torch.tensor(boundary_half_extent,device=state.device,dtype=static.dtype)-evader_radius-pos.abs()
    if static.shape[1]:
        clearance=norm(pos[:,None]-static[:,:,:2]).squeeze(-1)-static[:,:,2]-evader_radius
        static_clear=torch.minimum(clearance.amin(-1),bounds.amin(-1))
    else:
        static_clear=bounds.amin(-1)
    dynamic_pos=torch.cat((background,state[:,1:,:2]),1)
    dynamic_clear=norm(pos[:,None]-dynamic_pos).squeeze(-1)-radii-evader_radius
    oldpos=previous_state[:,0,:2]
    num_dynamic=background.shape[1]
    if flight_mode=='free_3d':
        static_craft=_column_contact(previous_state[:,:,:2],state[:,:,:2],static,evader_radius)&active
        dynamic_craft=_moving_contact(previous_state[:,:,:2],state[:,:,:2],previous_bg,background,radii[:,:num_dynamic],evader_radius)&active
        capture_mask=_sphere_sweep(previous_state[:,0,:3],state[:,0,:3],previous_state[:,1:,:3],state[:,1:,:3],2*evader_radius)<=0
        capture_mask=capture_mask&active[:,1:]&active[:,:1]
        body=_body_contact(previous_state[:,:,:3],state[:,:,:3],active,2*evader_radius)
        extent=torch.tensor(boundary_half_extent,device=state.device,dtype=state.dtype)
        outside=state[:,:,:2].abs().gt(extent-evader_radius).any(-1)|previous_state[:,:,:2].abs().gt(extent-evader_radius).any(-1)
        pursuer_out=outside.clone()
        pursuer_out[:,0]=False
        aircraft_failure=static_craft|dynamic_craft|body|(pursuer_out&active)
        static_hit=static_craft.any(-1)
        dynamic_hit=dynamic_craft.any(-1)
        pursuer_obstacle=(static_craft[:,1:]|dynamic_craft[:,1:]).any(-1)
        pursuer_peer=_body_contact(previous_state[:,1:,:3],state[:,1:,:3],active[:,1:],2*evader_radius).any(-1)
    else:
        if static.shape[1]:
            static_hit=(segment_clearance(oldpos[:,None]-static[:,:,:2],
                pos[:,None]-static[:,:,:2],static[:,:,2]+evader_radius)<=0).any(-1)
        else:
            static_hit=torch.zeros(state.shape[0],dtype=torch.bool,device=state.device)
        old_dynamic=torch.cat((previous_bg,previous_state[:,1:,:2]),1)
        swept=segment_clearance(oldpos[:,None]-old_dynamic,pos[:,None]-dynamic_pos,radii+evader_radius)
        dynamic_hit=(swept[:,:num_dynamic]<=0).any(-1)
        capture_mask=swept[:,num_dynamic:]<=0
        aircraft_failure=torch.zeros(state.shape[:2],dtype=torch.bool,device=state.device)
        pursuer_obstacle=torch.zeros(state.shape[0],dtype=torch.bool,device=state.device)
        pursuer_peer=torch.zeros_like(pursuer_obstacle)
    captured=capture_mask.any(-1)
    captor=torch.where(captured,capture_mask.long().argmax(-1),-1) if capture_mask.shape[1] else torch.full_like(captured,-1,dtype=torch.long)
    boundary_hit=((bounds<=0).any(-1) |
        ((torch.tensor(boundary_half_extent,device=state.device,dtype=static.dtype)-evader_radius-oldpos.abs())<=0).any(-1))
    collision=static_hit|dynamic_hit|boundary_hit|captured
    if flight_mode=='free_3d' and goal_height is not None:
        offset=state[:,0,2]-goal_height
        goal_reached=torch.sqrt(distance.square()+offset.square())<config['goal_radius']
    else:
        goal_reached=distance<config['goal_radius']
    # Engineering failure is distinct from task collisions/capture. Compare
    # height to the configured interval directly so representable endpoints
    # are accepted without subtractive roundoff at the boundary.
    height_low = diagnostic_state[..., 2] < config['physical_failure']['min_height']
    height_high = diagnostic_state[..., 2] > config['physical_failure']['max_height']
    excessive_speed = torch.linalg.vector_norm(diagnostic_state[..., 7:9], dim=-1) > config['physical_failure']['max_horizontal_speed']
    instability = height_low | height_high | excessive_speed
    if flight_mode=='free_3d':
        instability=instability&active
    physical_failure = instability.any(-1) if flight_mode == 'free_3d' else torch.zeros_like(collision)
    info=dict(goal_reached=goal_reached,captured=captured,captor_id=captor,
        capture_mask=capture_mask,static_collision=static_hit,dynamic_collision=dynamic_hit,
        boundary_collision=boundary_hit,collision=collision,
        physical_failure=physical_failure,physical_instability=instability,
        physical_height_low=height_low,physical_height_high=height_high,
        physical_excessive_speed=excessive_speed,
        aircraft_failure=aircraft_failure,
        pursuer_obstacle=pursuer_obstacle,pursuer_peer=pursuer_peer,
        clearance=torch.minimum(static_clear,(dynamic_clear.amin(-1) if dynamic_clear.shape[1] else torch.full_like(static_clear,float('inf')))))
    return info,distance,static_clear,dynamic_clear


def resolve_termination(events, *, steps, max_steps, config=None):
    """事件优先级与奖励启用无关；分别保留所有事实与唯一结算结果。"""
    config=load_settings()['task'] if config is None else config
    event_flags=(events['captured'],events['static_collision'],events['dynamic_collision'],
                 events['boundary_collision'],events['physical_failure'])
    failure=events['collision']|events['physical_failure']
    success=events['goal_reached'] & ~failure
    terminated=torch.zeros_like(success)
    for name,flag in zip(EVENT_NAMES[:5],event_flags):
        if config['terminate_on'][name]:terminated |= flag
    if config['terminate_on']['success']:terminated |= success
    truncated=(steps>=max_steps)&~terminated
    # 0 ongoing, 1 captured, 2 static, 3 dynamic, 4 boundary, 5 physical, 6 success, 7 timeout
    outcome=torch.zeros_like(steps)
    for index,flag in enumerate((*event_flags,success,truncated),1):
        outcome=torch.where((outcome==0)&flag,index,outcome)
    return terminated,truncated,success,outcome


def evaluate_task(state, *, goal_position, static, background, radii, steps,
                  max_steps, previous_state, previous_bg, diagnostic_state, flight_mode,
                  config=None, evader_radius=.3, boundary_half_extent=(10.5,13.),
                  active=None, goal_height=None):
    events,distance,static_clear,dynamic_clear=detect_events(state,
        goal_position=goal_position,static=static,background=background,radii=radii,
        previous_state=previous_state,previous_bg=previous_bg,
        diagnostic_state=diagnostic_state,flight_mode=flight_mode,config=config,
        evader_radius=evader_radius,boundary_half_extent=boundary_half_extent,
        active=active,goal_height=goal_height)
    term,trunc,success,outcome=resolve_termination(events,steps=steps,max_steps=max_steps,config=config)
    events.update(success=success,timeout=trunc,outcome=outcome)
    return TaskResult(term,trunc,events,distance,static_clear,dynamic_clear)
