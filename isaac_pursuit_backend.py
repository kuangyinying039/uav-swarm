"""Single-scene Isaac Sim pursuit backend for MAPPO and MATD3."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import sys

import numpy as np
import torch

from quadrotor_pursuit_env import QuadrotorPursuitConfig


class IsaacPursuitEnv:
    """One complete three-UAV pursuit scene backed by Isaac rigid-body physics.

    This first backend intentionally supports one scene. The outer trainer can
    be validated against real flight dynamics before adding batched Isaac GAE
    and replay collection.
    """

    def __init__(self, cfg: QuadrotorPursuitConfig, *, seed=0, headless=True, native_lidar=False):
        self.cfg = replace(cfg, seed=int(seed))
        self._headless = headless
        self._native_lidar_enabled = bool(native_lidar)
        self._needs_reset = True
        self._closed = False
        self._steps = 0
        self._capture = False
        self._safety_reasons = [[] for _ in range(self.cfg.n_uavs)]
        self._project_root = Path(__file__).resolve().parents[2]
        self._experiment_dir = (
            self._project_root
            / "omnidrones/experiments/radar5_frozen_transfer_v1"
        )
        omnidrones_root = Path(os.environ.get("OMNIDRONES_SOURCE", "/opt/lab/src/OmniDrones-isaac41"))
        drone_assets = omnidrones_root / "omni_drones/robots/assets/usd"
        if not (drone_assets / "hummingbird.yaml").is_file():
            raise FileNotFoundError(f"Hummingbird assets not found under {drone_assets}")
        if str(omnidrones_root) in sys.path:
            sys.path.remove(str(omnidrones_root))
        sys.path.insert(0, str(omnidrones_root))
        for import_root in (self._project_root / "omnidrones",):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))

        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app

        config_dir = omnidrones_root / "cfg"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"OmniDrones Hydra config directory not found: {config_dir}")
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            isaac_cfg = compose(
                config_name="train",
                overrides=[
                    "task=Formation",
                    "task.drone_model.name=Hummingbird",
                    "task.drone_model.controller=LeePositionController",
                    "task.env.num_envs=1",
                    f"task.env.max_episode_length={self.cfg.search_steps}",
                    "task.formation=radar5_three",
                    f"headless={'true' if headless else 'false'}",
                    "++sim.device=cuda:0",
                ],
            )
        OmegaConf.resolve(isaac_cfg)
        OmegaConf.set_struct(isaac_cfg, False)
        physics_steps = max(1, round(self.cfg.decision_dt / 0.01))
        self.physics_dt = self.cfg.decision_dt / physics_steps
        isaac_cfg.sim.dt = self.physics_dt
        isaac_cfg.sim.substeps = 1
        isaac_cfg.sim.gravity = [0.0, 0.0, -self.cfg.quadrotor_gravity]
        isaac_cfg.sim.device = "cuda:0"
        isaac_cfg.sim.use_gpu_pipeline = True
        isaac_cfg.sim.replicate_physics = False
        isaac_cfg.sim.use_flatcache = True
        isaac_cfg.env.num_envs = 1
        isaac_cfg.task.env.num_envs = 1
        isaac_cfg.task.env.max_episode_length = self.cfg.search_steps

        self.simulation_app = init_simulation_app(isaac_cfg)
        from omni_drones.envs import formation as formation_task
        from omni_drones.envs.formation import Formation

        def formation_cost(points, desired_p):
            points = points - points.mean(-2, keepdim=True)
            desired_p = desired_p - desired_p.mean(-2, keepdim=True)
            return torch.cdist(points, desired_p).min(-1).values.max(-1).values.unsqueeze(-1)

        formation_task.cost_formation_hausdorff = formation_cost

        formation_task.FORMATIONS["radar5_three"] = [
            [0.0, 0.0, self.cfg.initial_altitude],
            [self.cfg.formation_spacing, 0.0, self.cfg.initial_altitude],
            [0.0, self.cfg.formation_spacing, self.cfg.initial_altitude],
        ]
        self.isaac_cfg = isaac_cfg
        try:
            self.sim_env = Formation(isaac_cfg, headless=headless)
        except Exception:
            self.simulation_app.close()
            raise
        self.sim_env.controller = self.sim_env.controller.to(self.sim_env.device)

        from experiments.radar5_frozen_transfer_v1.isaac_sensor_bridge import IsaacLidarAdapter

        self.lidar = IsaacLidarAdapter(3, n_targets=1, decision_dt=self.cfg.decision_dt)
        from experiments.radar5_frozen_transfer_v1.isaac_pursuit_observation import IsaacPursuitObservationBridge

        self.bridge = IsaacPursuitObservationBridge(self.cfg, sensor_adapter=self.lidar)
        self._create_scene_prims()
        self.sensor_paths = []
        self._sensor_creation_errors = []
        self.lidar_interface = None
        if self._native_lidar_enabled:
            from omni.isaac.range_sensor import _range_sensor

            self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
            for drone_path in self.sim_env.drone.prim_paths:
                drone_path = str(drone_path).rstrip("/")
                parent = drone_path if drone_path.endswith("/base_link") else f"{drone_path}/base_link"
                sensor_path = f"{parent}/radar5_lidar"
                try:
                    self.sensor_paths.append(self._create_lidar(sensor_path, parent))
                except Exception as exc:  # pragma: no cover - simulator-dependent fallback
                    self._sensor_creation_errors.append((sensor_path, exc))
                    self.sensor_paths.append(None)
        else:
            self.sensor_paths = [None] * self.cfg.n_uavs
        self._initialize_scene_transforms()
        self._last_native_scan = None
        self.reset(seed=seed)

    @property
    def n_agents(self):
        return self.cfg.n_uavs

    @property
    def t(self):
        return self._steps

    @property
    def captured(self):
        return self._capture

    def __getattr__(self, name):
        bridge = self.__dict__.get("bridge")
        if bridge is not None and hasattr(bridge.env, name):
            return getattr(bridge.env, name)
        raise AttributeError(name)

    def obs_dim(self):
        return self.bridge.env.obs_dim()

    def state_dim(self):
        return self.bridge.env.state_dim()

    def continuous_action_dim(self):
        return self.bridge.env.continuous_action_dim()

    def minimum_capture_gap(self):
        state = self._physical_state()
        target = self._target_position()
        active = ~np.asarray(self.bridge.env.disabled_uavs, dtype=bool)
        distances = np.linalg.norm(state[:, :3] - target[None, :], axis=1)
        required = int(self.cfg.capture_required_uavs)
        if np.count_nonzero(active) < required:
            return float(self.cfg.grid_size)
        required_distance = np.partition(distances[active], required - 1)[required - 1]
        return float(max(required_distance - self.cfg.capture_radius, 0.0))

    def configure_domain(self, cfg: QuadrotorPursuitConfig):
        """Switch among transfer profiles with an identical policy/scene shape."""
        if (
            cfg.n_uavs != self.cfg.n_uavs
            or cfg.n_targets != self.cfg.n_targets
            or cfg.building_count != self.cfg.building_count
            or cfg.building_state_capacity != self.cfg.building_state_capacity
            or cfg.capture_required_uavs != self.cfg.capture_required_uavs
            or cfg.execution_reward_version != self.cfg.execution_reward_version
            or cfg.capture_radius != self.cfg.capture_radius
        ):
            raise ValueError("Transfer domain changes the policy or capture contract")
        if not np.isclose(cfg.decision_dt, self.cfg.decision_dt):
            raise ValueError("Transfer domain must keep decision_dt fixed")
        self.cfg = replace(cfg, seed=self.cfg.seed)
        self.bridge.env.cfg = self.cfg

    def _has_physical_capture(self, state, target):
        active = ~np.asarray(self.bridge.env.disabled_uavs, dtype=bool)
        distances = np.linalg.norm(state[:, :3] - target[None, :], axis=1)
        return int(np.count_nonzero(active & (distances <= self.cfg.capture_radius))) >= int(
            self.cfg.capture_required_uavs
        )

    @staticmethod
    def _episode_end_flags(captured, task_timeout, upstream_terminated, upstream_truncated):
        terminated = bool(captured or task_timeout or upstream_terminated)
        truncated = bool(upstream_truncated and not task_timeout and not terminated)
        return terminated, truncated

    def _create_lidar(self, path, parent):
        from experiments.radar5_frozen_transfer_v1.isaac_sensor_bridge import IsaacLidarAdapter

        return IsaacLidarAdapter.create_physx_lidar(
            path,
            parent,
            min_range=self.cfg.lidar_min_range,
            max_range=self.cfg.lidar_target_detection_range,
            horizontal_fov=self.cfg.lidar_horizontal_fov_deg,
            vertical_fov=self.cfg.lidar_vertical_max_deg - self.cfg.lidar_vertical_min_deg,
            horizontal_resolution=5.0,
            vertical_resolution=5.0,
        )

    def _create_scene_prims(self):
        from pxr import Gf, UsdGeom, UsdPhysics
        import omni.usd
        from omni.isaac.core.utils.semantics import add_update_semantics

        stage = omni.usd.get_context().get_stage()
        ground = UsdGeom.Cube.Define(stage, "/World/envs/env_0/radar5_ground")
        ground.CreateDisplayColorAttr([(0.10, 0.14, 0.17)])
        ground_xform = UsdGeom.Xformable(ground.GetPrim())
        ground_translate = ground_xform.AddTranslateOp()
        ground_scale = ground_xform.AddScaleOp()
        ground_translate.Set(Gf.Vec3d(self.cfg.grid_size / 2, self.cfg.grid_size / 2, -0.08))
        ground_scale.Set(Gf.Vec3f(self.cfg.grid_size / 2, self.cfg.grid_size / 2, 0.08))
        ground_xform.SetXformOpOrder([ground_translate, ground_scale])
        self._target_path = "/World/envs/env_0/radar5_target"
        target = UsdGeom.Sphere.Define(stage, self._target_path)
        target.CreateRadiusAttr(self.cfg.target_diameter / 2.0)
        target.CreateDisplayColorAttr([(0.95, 0.22, 0.10)])
        add_update_semantics(target.GetPrim(), "radar5_target", "class")
        UsdPhysics.CollisionAPI.Apply(target.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
        rigid.CreateKinematicEnabledAttr(True)
        self._target_translate = UsdGeom.Xformable(target.GetPrim()).AddTranslateOp()
        self._building_ops = []
        for index in range(self.cfg.building_count):
            path = f"/World/envs/env_0/radar5_building_{index}"
            building = UsdGeom.Cube.Define(stage, path)
            building.CreateDisplayColorAttr([(0.10, 0.42, 0.52)])
            UsdPhysics.CollisionAPI.Apply(building.GetPrim())
            xform = UsdGeom.Xformable(building.GetPrim())
            translate = xform.AddTranslateOp()
            scale = xform.AddScaleOp()
            self._building_ops.append((translate, scale))

    def _initialize_scene_transforms(self):
        from pxr import Gf

        self._target_translate.Set(Gf.Vec3d(0.0, 0.0, self.cfg.initial_altitude))
        self._set_building_transforms()

    def _set_building_transforms(self):
        from pxr import Gf

        for (translate, scale), rect, height in zip(
            self._building_ops, self.bridge.env.buildings, self.bridge.env.building_heights
        ):
            x0, y0, x1, y1 = map(float, rect)
            translate.Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, float(height) / 2))
            scale.Set(Gf.Vec3f((x1 - x0) / 2, (y1 - y0) / 2, float(height) / 2))

    def _physical_state(self):
        from experiments.radar5_frozen_transfer_v1.state_adapter import omnidrones_state_to_radar5

        state = self.sim_env.drone.get_state(check_nan=True)
        converted = omnidrones_state_to_radar5(state)
        return np.asarray(converted).reshape(3, 13)

    def _sync_bridge(self):
        self.bridge.sync_pursuers(self._physical_state())

    def _target_position(self):
        env = self.bridge.env
        return np.asarray([*env.dynamic_targets[0], env.target_altitudes[0]], dtype=np.float64)

    def _write_target_pose(self):
        from pxr import Gf

        self._target_translate.Set(Gf.Vec3d(*self._target_position().tolist()))

    def _read_native_lidar(self):
        if not self._native_lidar_enabled:
            self._last_native_scan = None
            self.bridge.clear_native_sensor_scan()
            return
        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import quaternion_to_rotation

        physical = self._physical_state()
        positions = physical[:, :3]
        rotations = np.stack([quaternion_to_rotation(row[6:10]) for row in physical])
        scan = self.lidar.read_native_scan(
            sim_time=self._steps * self.cfg.decision_dt,
            lidar_interface=self.lidar_interface,
            sensor_paths=self.sensor_paths,
            sensor_origins=positions,
            sensor_rotations=rotations,
            target_prim_tokens=("radar5_target",),
        )
        self._last_native_scan = scan
        if scan.get("sensor_available", False):
            self.bridge.set_native_sensor_scan(scan)
        else:
            self.bridge.clear_native_sensor_scan()

    def reset(self, seed=None):
        self.lidar.reset()
        if seed is not None:
            self.cfg.seed = int(seed)
            self.bridge.env.cfg.seed = int(seed)
            self.bridge.env.rng = np.random.default_rng(int(seed))
        self.td = self.sim_env.reset()
        self.bridge.reset()
        self._steps = 0
        self._capture = False
        self._needs_reset = False
        self._set_building_transforms()
        physical = self.bridge.env.quadrotor_states
        positions = torch.as_tensor(physical[:, :3], dtype=torch.float32, device=self.sim_env.device).unsqueeze(0)
        orientations = torch.as_tensor(physical[:, 6:10], dtype=torch.float32, device=self.sim_env.device).unsqueeze(0)
        self.sim_env.drone.set_world_poses(positions, orientations)
        self.sim_env.drone.set_velocities(torch.zeros((1, 3, 6), device=self.sim_env.device))
        self._write_target_pose()
        self._sync_bridge()
        try:
            self._read_native_lidar()
        except (RuntimeError, ValueError):
            self._last_native_scan = None
            self.bridge.clear_native_sensor_scan()
        return self.bridge.observe()

    def observe_search(self):
        if self._needs_reset:
            self.reset()
        self._sync_bridge()
        return self.bridge.observe()

    def coverage(self):
        return float(getattr(self.bridge.env, "coverage", lambda: 0.0)())

    def step_joint(self, actions):
        action = np.asarray(actions, dtype=np.float32)
        if action.shape != (3, 4) or not np.isfinite(action).all():
            raise ValueError("Isaac pursuit action must be finite with shape (3, 4)")
        if self._needs_reset:
            raise RuntimeError("step_joint called after episode end; call observe_search to reset")

        from experiments.radar5_frozen_transfer_v1.sim2real_contract import policy_action_to_setpoint

        self._sync_bridge()
        measured_yaw = np.asarray(self.bridge.env.headings, dtype=np.float32) * (np.pi / 4.0)
        requested_velocity, target_yaw = policy_action_to_setpoint(
            action,
            measured_yaw,
            decision_dt=self.cfg.decision_dt,
            max_horizontal_velocity=self.cfg.max_horizontal_velocity,
            max_vertical_velocity=self.cfg.max_vertical_velocity,
            max_yaw_rate=self.cfg.max_reference_yaw_rate,
        )
        requested_velocity, safety_reasons = self.bridge.filter_references(requested_velocity)
        velocity_tensor = torch.as_tensor(requested_velocity, dtype=torch.float32, device=self.sim_env.device).unsqueeze(0)
        yaw_tensor = torch.as_tensor(target_yaw, dtype=torch.float32, device=self.sim_env.device).unsqueeze(0)

        for _ in range(round(self.cfg.decision_dt / self.physics_dt)):
            state = self.sim_env.drone.get_state(check_nan=True)
            target_pos = state[..., :3].clone()
            with torch.no_grad():
                motor_action = self.sim_env.controller.compute(
                    state[..., :13], target_pos=target_pos,
                    target_vel=velocity_tensor, target_yaw=yaw_tensor,
                ).clamp(-1.0, 1.0)
            self.td = self.sim_env.step(self.td.set(("agents", "action"), motor_action))

        self._sync_bridge()
        result = self.bridge.advance()
        self._steps += 1
        self._write_target_pose()
        self._sync_bridge()
        self._read_native_lidar()
        result["obs"] = self.bridge.observe()
        actual_gap = self.minimum_capture_gap()
        capture = self._has_physical_capture(self._physical_state(), self._target_position())
        bridge_capture = bool(result.get("capture_success", False))
        self._capture = bool(capture or bridge_capture)
        result["capture_success"] = self._capture
        result["minimum_capture_gap"] = actual_gap
        result["safety_reasons"] = safety_reasons
        task_timeout = self._steps >= self.cfg.search_steps and not self._capture
        result["terminated"], result["truncated"] = self._episode_end_flags(
            self._capture,
            task_timeout,
            result.get("terminated", False),
            result.get("truncated", False),
        )
        result["reward"] = float(result["reward"])
        result["reward_components"] = dict(result.get("reward_components", {}))
        missing_capture_event_reward = capture and not bridge_capture
        result["reward_components"]["isaac_capture_reconciliation"] = (
            float(missing_capture_event_reward) * self.cfg.capture_reward
        )
        if missing_capture_event_reward:
            result["reward"] += self.cfg.capture_reward
        self._safety_reasons = safety_reasons
        self._needs_reset = result["terminated"] or result["truncated"]
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self.sim_env, "close"):
                self.sim_env.close()
        except Exception:
            pass
        try:
            if hasattr(self.simulation_app, "close"):
                self.simulation_app.close()
        except Exception:
            pass