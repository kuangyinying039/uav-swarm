"""Configurable-count Hummingbirds with default ideal altitude constraint and CUDA geometry."""
import time
import torch
from parallel.env.core import BatchRules
from parallel.env.config_loader import load_visualization
from parallel.env.isaac_setup import IsaacSetup
from parallel.env.interface import TensorSpec
from parallel.env.observation import observation_layout, observation_scale
from parallel.env.mid360 import Mid360Tracks
from parallel.env.radar5_contract import body_yaw, clip_velocity_yaw


class ParallelEnv:
    def __init__(self, cfg, seeds, max_steps=None, visualize=None, sampler_backend='cpu', profile_reset=False, flight_mode=None, environment_config=None, pursuer_config=None, visualization_config=None, formation=None, perception=None):
        self._closed = False
        self._device = str(cfg.sim.device)
        if self.device != 'cuda:0' or not cfg.sim.use_gpu_pipeline:
            raise ValueError('v1 requires direct CUDA tensors and GPU pipeline')
        self._n, self.max_steps = len(seeds), max_steps
        self.reset_count = 0
        self.rules = BatchRules(seeds, self.device, sampler_backend=sampler_backend, flight_mode=flight_mode, environment_config=environment_config, pursuer_config=pursuer_config, formation=formation)
        if perception not in (None, 'mid360'):
            raise ValueError('perception must be mid360 or omitted')
        self.perception = 'mid360' if formation == 'one_side_triangle' or perception == 'mid360' else None
        self.mid360 = Mid360Tracks(self.n, self.rules.layout.num_pursuers, self.device) if self.perception == 'mid360' else None
        self.flight_mode = self.rules.flight_mode
        simulation=self.rules.settings['simulation']
        if float(cfg.sim.dt)!=simulation['dt'] or bool(cfg.headless)!=simulation['headless']:
            raise ValueError('Configure Isaac app dt/headless from environment simulation settings BEFORE starting the app')
        self.visualization_config=load_visualization(visualization_config)
        if visualize is not None:
            if type(visualize) is not bool: raise ValueError('visualize must be bool or None')
            self.visualization_config['enabled']=visualize
        self.visualize=self.visualization_config['enabled']
        if max_steps is not None:
            from parallel.env.config_loader import positive_integer
            positive_integer(max_steps,"max_steps")
            self.rules.settings["task"]["max_episode_steps"]=max_steps
        self.max_steps=self.rules.settings["task"]["max_episode_steps"]
        self.set_reset_profiling(profile_reset)
        self.last_reset_profile = {}
        setup = IsaacSetup(cfg, self.rules.scene, self.rules.bg, simulation=simulation, visualization=self.visualization_config)
        self.sim, self.stage = setup.sim, setup.stage
        self.drone, self.controller = setup.drone, setup.controller
        self.offsets, self.tensor_sim = setup.offsets, setup.tensor_sim
        self.joints, self.gpu_physics = setup.joints, setup.gpu_physics
        self.visualizer = setup.visualizer
        self.quat = torch.zeros(self.n, self.rules.layout.num_aircraft, 4, device=self.device)
        self.return_sum = torch.zeros(self.n, device=self.device)
        self.disabled = torch.zeros(self.n, self.rules.layout.num_aircraft, dtype=torch.bool, device=self.device)
        self.reset(torch.arange(self.n, device=self.device), resample=False)

    def _lidar_cylinders(self):
        static = self.rules.static
        centers = [static[..., :2]]
        radii = [static[..., 2]]
        if self.rules.bg.shape[1]:
            centers.append(self.rules.bg)
            radii.append(self.rules.radii[:, self.rules.layout.dynamic_objects])
        return torch.cat(centers, 1), torch.cat(radii, 1)

    @property
    def n(self):
        """并行局数（只读）。"""
        return self._n

    @property
    def num_envs(self):
        return self._n

    @property
    def device(self):
        return self._device

    @property
    def unwrapped(self):
        return self

    @property
    def closed(self):
        return self._closed

    @property
    def action_spec(self):
        return TensorSpec((self.n,self.rules.layout.num_aircraft,2), 'float32', 'm/s', 'world_xy',
            '平面包装仍用世界系水平速度。step 另接受 [N,aircraft,4]：vx,vy,vz,yaw_rate')

    @property
    def observation_spec(self):
        width=self.observation_layout[-1]['stop']
        return TensorSpec((self.n,width), 'float32', 'mixed', 'see observation_layout',
            'E自身/目标、激光、背景对象、配置允许的固定P槽位及mask')

    @property
    def observation_layout(self):
        return observation_layout(self.rules.settings['observation'])

    @property
    def observation_scale(self):
        return observation_scale(self.rules.settings['observation'],environment=self.rules.settings,pursuer=self.rules.pursuer_config)

    @property
    def state_spec(self):
        return TensorSpec((self.n,self.rules.layout.num_aircraft,13), 'float32', 'm, quaternion, m/s, rad/s', 'environment_world',
            '每架飞机：位置3、四元数wxyz4、线速度3、角速度3；位置扣除并行场景偏移')

    @property
    def profile_reset(self):
        return self._profile_reset

    @profile_reset.setter
    def profile_reset(self, enabled):
        self.set_reset_profiling(enabled)

    def set_reset_profiling(self, enabled):
        self._ensure_open()
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be bool')
        self._profile_reset = enabled
        self.rules.profile_reset = enabled

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError('Environment is closed')

    def close(self):
        """幂等释放本对象持有的物理引用；不停止/关闭调用方拥有的 Isaac app。

        Isaac 场景和全局 GPU 资源由创建 app 的运行入口最终释放。
        close 后不允许继续交互，也不支持重新打开此对象。
        """
        if self.closed:
            return
        self._closed = True
        self.visualizer.close()
        self.tensor_sim = None
        self.drone = None
        self.controller = None
        self.sim = None
        self.stage = None

    def sync_visuals(self):
        if self.visualize:
            self.visualizer.update(self.rules.static, self.rules.bg, self.rules.radii[:, self.rules.layout.dynamic_objects])

    def reset(self, ids=None, resample=True, seeds=None):
        self._ensure_open()
        if ids is None:
            ids = torch.arange(self.n,device=self.device)
        if not isinstance(ids,torch.Tensor) or ids.dtype != torch.long:
            raise TypeError("ids must be an int64 tensor or None")
        ids = ids.to(device=self.device, dtype=torch.long)
        if self.profile_reset: torch.cuda.synchronize()
        started = time.perf_counter()
        self.rules.reset(ids, resample=resample, seeds=seeds)
        if not ids.numel(): return self.observe()
        if self.profile_reset: torch.cuda.synchronize()
        rules_done = time.perf_counter()
        self.quat[ids] = 0.
        self.quat[ids, :, 0] = (self.rules.initial_yaw[ids]/2).cos()
        self.quat[ids, :, 3] = (self.rules.initial_yaw[ids]/2).sin()
        self.reset_count += ids.numel() if resample else 0
        self.drone._reset_idx(ids)
        for name in ('forces', 'thrusts', 'torques', 'vel', 'vel_w', 'vel_b', 'acc', 'acc_w', 'acc_b'):
            getattr(self.drone, name)[ids] = 0.
        self.drone.set_world_poses(self.rules.initial_pos[ids]+self.offsets[ids, None], self.quat[ids], env_indices=ids)
        self.drone.set_velocities(self.rules.initial_velocity[ids], env_indices=ids)
        self.drone._view.set_joint_positions(self.joints[ids], env_indices=ids)
        self.drone._view.set_joint_velocities(torch.zeros_like(self.joints[ids]), env_indices=ids)
        self.tensor_sim.update_articulations_kinematic()
        if self.flight_mode == 'planar': self.apply_altitude_constraint(ids)
        self.return_sum[ids] = 0.
        self.disabled[ids] = False
        if self.mid360 is not None:
            self.mid360.reset(ids, self.state(), self.disabled[:, 1:], self.rules.buildings, self._lidar_cylinders())
        self.sync_visuals()

    def nudge_starts(self, ids, noise: float = 1.5):
        """同一套障碍上，给各机水平出生位置加噪声。"""
        if ids is None or not int(ids.numel()) or noise <= 0:
            return
        delta = (torch.rand(int(ids.numel()), self.rules.initial_pos.shape[1], 2, device=self.device) * 2 - 1) * float(noise)
        position = self.rules.initial_pos[ids].clone()
        position[..., :2] = position[..., :2] + delta
        half_x = float(self.rules.cfg["boundary_half_extent_x"]) - 1.0
        half_y = float(self.rules.cfg["boundary_half_extent_y"]) - 1.0
        position[..., 0].clamp_(-half_x, half_x)
        position[..., 1].clamp_(-half_y, half_y)
        self.rules.initial_pos[ids] = position
        self.drone.set_world_poses(position + self.offsets[ids, None], self.quat[ids], env_indices=ids)
        self.drone.set_velocities(self.rules.initial_velocity[ids], env_indices=ids)
        self.tensor_sim.update_articulations_kinematic()
        if self.mid360 is not None:
            self.mid360.reset(ids, self.state(), self.disabled[:, 1:], self.rules.buildings, self._lidar_cylinders())
        if self.profile_reset: torch.cuda.synchronize()
        physical_done = time.perf_counter()
        obs = self.observe()
        if self.profile_reset:
            torch.cuda.synchronize()
            self.last_reset_profile = dict(rules_seconds=rules_done-started, physical_write_seconds=physical_done-rules_done, observe_seconds=time.perf_counter()-physical_done, **getattr(self.rules, 'last_reset_profile', {}))
        return obs

    def apply_altitude_constraint(self, ids=None, raw_state=None):
        """Ideal altitude constraint, not a physical altitude controller.

        GPU root tensor writes retain XY, orientation, angular velocity and
        joints. Reading state afterwards refreshes upstream vel/vel_w caches.
        """
        raw = self.state() if raw_state is None else raw_state
        if not torch.isfinite(raw).all(): raise RuntimeError('Nonfinite constraint input')
        selected = torch.arange(self.n, device=self.device) if ids is None else ids
        positions = raw[selected, :, :3].clone() + self.offsets[selected, None]
        positions[..., 2] = self.offsets[selected, None, 2] + self.rules.settings['simulation']['target_height']
        velocity = raw[selected, :, 7:13].clone(); velocity[..., 2] = 0.
        self.drone.set_world_poses(positions, env_indices=selected)
        self.drone.set_velocities(velocity, env_indices=selected)
        self.tensor_sim.update_articulations_kinematic()
        return self.state()

    def _keep_flight_band(self, raw):
        """把仍在飞行、但掉出 1–12 米的机体托回飞行带。已经相碰失效的机体不再托起。"""
        correct = ((raw[..., 2] < 1.0) | (raw[..., 2] > 12.0)) & ~self.disabled
        if not correct.any():
            return raw
        positions = raw[:, :, :3].clone() + self.offsets[:, None]
        floor = self.offsets[:, None, 2] + 1.0
        ceiling = self.offsets[:, None, 2] + 12.0
        clamped = torch.minimum(torch.maximum(positions[..., 2], floor), ceiling)
        positions[..., 2] = torch.where(correct, clamped, positions[..., 2])
        velocity = raw[:, :, 7:13].clone()
        velocity[..., 2] = torch.where(correct & (raw[..., 2] < 1.0), velocity[..., 2].clamp(min=0), velocity[..., 2])
        velocity[..., 2] = torch.where(correct & (raw[..., 2] > 12.0), velocity[..., 2].clamp(max=0), velocity[..., 2])
        ids = torch.arange(self.n, device=self.device)
        self.drone.set_world_poses(positions, env_indices=ids)
        self.drone.set_velocities(velocity, env_indices=ids)
        self.tensor_sim.update_articulations_kinematic()
        return self.state()

    def state(self):
        self._ensure_open()
        state = self.drone.get_state()[..., :13].clone()
        if not torch.isfinite(state).all(): raise RuntimeError('Nonfinite physical state')
        return state

    def motor_state(self):
        """执行中的归一化旋翼速度，不是控制器输出；返回副本 [N,1+P,4]。"""
        self._ensure_open()
        value=self.drone.throttle.clone()
        if value.shape!=(self.n,self.rules.layout.num_aircraft,4) or not torch.isfinite(value).all():
            raise RuntimeError('Invalid actual rotor throttle state')
        return value

    def _observation(self,state):
        fields=self.rules.settings['observation']['pursuers']['fields']
        motors=self.motor_state() if 'motor_throttle' in fields else None
        return self.rules.observation(state,motor_state=motors)

    def observe(self): return self._observation(self.state())

    def _velocity_yaw_command(self, action, state):
        """二维平面指令保持原样；四维指令是世界系速度加偏航角速度。"""
        aircraft = state.shape[1]
        if action.shape not in ((self.n, aircraft, 2), (self.n, aircraft, 4)) or not torch.isfinite(action).all():
            raise ValueError(
                f'Expected finite [N,aircraft,2] XY velocity or [N,aircraft,4] '
                f'(vx, vy, vz, yaw_rate), got {tuple(action.shape)}'
            )
        if action.shape[-1] == 2:
            commands = torch.zeros_like(state[:, :, :3])
            commands[:, :, :2] = action
            target = state[:, :, :3].clone()
            target[:, :, 2] = self.rules.settings['simulation']['target_height']
            target_yaw = self.rules.initial_yaw.reshape(-1, 1)
            yaw_rate = torch.zeros(self.n, aircraft, device=state.device, dtype=state.dtype)
            return commands, target, target_yaw, yaw_rate, self.flight_mode == 'planar'
        evader_horizontal = getattr(self, "evader_horizontal", None)
        evader_vertical = getattr(self, "evader_vertical", None)
        if evader_horizontal is None:
            evader_horizontal = float(self.rules.settings['simulation']['evader_speed_limit'])
            evader_vertical = None
        commands, yaw_rate = clip_velocity_yaw(action, evader_horizontal, evader_vertical)
        target = state[:, :, :3].clone()
        dt = float(self.rules.settings['simulation']['dt'])
        yaw = body_yaw(state[:, :, 3:7])
        target_yaw = torch.atan2(torch.sin(yaw + yaw_rate * dt), torch.cos(yaw + yaw_rate * dt))
        return commands, target, target_yaw.reshape(-1, 1), yaw_rate, False

    @torch.no_grad()
    def step(self, action):
        """执行速度指令。索引 0=E，1:=P。

        [N,1+P,2] 是原平面世界系水平速度，高度锁在 target_height，偏航保持初始值。
        [N,1+P,4] 是 (vx, vy, vz, yaw_rate)：位置设定钉在当前坐标，Lee 跟踪世界系速度；
        偏航设定为当前偏航加上角速度乘物理步长。四维指令不锁高度。
        """
        state = self.state()
        previous_bg = self.rules.bg.clone()
        previous_distance = torch.linalg.vector_norm(state[:, 0, :2]-self.rules.goal, dim=-1)
        action = action.to(device=self.device, dtype=state.dtype)
        commands, target, target_yaw, yaw_rate, lock_altitude = self._velocity_yaw_command(action, state)
        self.rules.advance_dynamic_obstacles()
        rotor = self.controller.compute(state.reshape(-1, 13), target_pos=target.reshape(-1, 3),
                                        target_vel=commands.reshape(-1, 3), target_yaw=target_yaw)
        if not torch.isfinite(rotor).all(): raise RuntimeError('Nonfinite controller output')
        rotor = rotor.reshape(self.n, self.rules.layout.num_aircraft, -1)
        if not lock_altitude:
            rotor = torch.where(self.disabled.unsqueeze(-1), torch.full_like(rotor, -1.0), rotor)
        self.drone.apply_action(rotor)
        # IsaacSetup installs the local finite coplanar downwash adapter.
        # Nonfinite forces still fail explicitly before physics advances.
        for name in ('forces', 'thrusts', 'torques'):
            if not torch.isfinite(getattr(self.drone, name)).all(): raise RuntimeError(f'Nonfinite drone {name}')
        self.sync_visuals()
        # Render only after projection so the window never shows raw Z drift.
        self.sim.step(render=False)
        raw = self.state()
        current = self.apply_altitude_constraint(raw_state=raw) if lock_altitude else self._keep_flight_band(raw)
        if self.mid360 is not None and not lock_altitude:
            self.mid360.advance(current, self.disabled[:, 1:], float(self.rules.settings['simulation']['dt']), self.rules.buildings, self._lidar_cylinders())
        if self.visualize or not self.rules.settings['simulation']['headless']: self.sim.render()
        reward, term, trunc, info = self.rules.reward_done(
            current, previous_distance, self.max_steps, state, previous_bg, diagnostic_state=raw, active=~self.disabled
        )
        if not lock_altitude:
            self.disabled |= info["aircraft_failure"]
        self.return_sum += reward
        constraint_failure = ((current[..., 2] != self.rules.settings['simulation']['target_height']) | (current[..., 9] != 0.)).any(-1) if lock_altitude else torch.zeros(self.n, dtype=torch.bool, device=self.device)
        info.update(target_height=self.rules.settings['simulation']['target_height'],altitude_constraint_failure=constraint_failure,
                    yaw_rate=yaw_rate, command_width=action.shape[-1], lock_altitude=lock_altitude,
                    pre_projection_height=raw[..., 2].clone(), pre_projection_vz=raw[..., 9].clone(),
                    altitude_correction=current[..., 2]-raw[..., 2], vertical_velocity_correction=current[..., 9]-raw[..., 9],
                    disabled=self.disabled.clone(),
                    episode_return=self.return_sum.clone(),
                    length=self.rules.steps.clone(), state=current, commands=commands,
                    terminal_observation=self._observation(current))
        mask=self.observation_layout[-1]
        info['pursuer_observation_mask']=info['terminal_observation'][:,mask['start']:mask['stop']].bool()
        self.sync_visuals()
        return info['terminal_observation'], reward, term, trunc, info
