"""Single-scene Isaac Sim pursuit backend for MAPPO and MATD3."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import sys

import numpy as np
import torch

from quadrotor_pursuit_env import QuadrotorPursuitConfig


def _require_display_for_viewport() -> None:
    """Fail fast when GUI mode is requested without a usable display."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise RuntimeError(
        "Isaac Viewport visualization requires DISPLAY or WAYLAND_DISPLAY. "
        "Use --headless (default) on SSH training hosts, or run "
        "play_isaac_pursuit.py --visualize on a workstation with a monitor / X11."
    )


def _init_simulation_app(cfg) -> object:
    """Launch Isaac with the standard python kit.

    Always use ``omni.isaac.sim.python.kit`` (with ``headless=True`` → Kit
    ``--no-window``). The lighter ``python.gym.headless.kit`` does **not** load
    ``omni.isaac.debug_draw``, which OmniDrones imports at module import time and
    then crashes every train/eval start. Successful A0 training on this host used
    ``python.kit`` + headless; keep that combination.
    """
    from isaacsim import SimulationApp

    headless = bool(cfg["headless"])
    if not headless:
        _require_display_for_viewport()
    config = {"headless": headless, "anti_aliasing": 1}
    exp_root = os.environ.get("EXP_PATH") or os.environ.get("ISAACSIM_PATH")
    if not exp_root:
        raise RuntimeError("EXP_PATH / ISAACSIM_PATH must be set before launching Isaac")
    candidate = Path(exp_root)
    # setup_conda_env.sh sets EXP_PATH to <isaac>/apps; fall back to that layout.
    exp_path = candidate if (candidate / "omni.isaac.sim.python.kit").is_file() else candidate / "apps"
    gui_kit = exp_path / "omni.isaac.sim.python.kit"
    if not gui_kit.is_file():
        raise FileNotFoundError(f"No Isaac python kit under {exp_path}")
    app_experience = str(gui_kit)
    print(f"[isaac] SimulationApp experience={app_experience} headless={headless}", flush=True)
    return SimulationApp(config, experience=app_experience)


class IsaacPursuitEnv:
    """Three-UAV pursuit scene(s) backed by Isaac rigid-body physics.

    ``num_envs=1`` (default)
        Single-scene API: ``reset(seed=…)`` → obs dict, ``step_joint((3,4))`` → result.

    ``num_envs>1``
        One PhysX process, N GridCloner scenes. Isaac batches drone dynamics;
        Radar5 KF-CI / dual-capture reward / geometric lidar stay as N Python
        bridges (Isaac has no native KF-CI or k=2 capture). With CUDA, each
        bridge's 6-D CV predict + CI scoring use float64 GPU batches
        (``use_gpu_tracker``; same math as NumPy). Vector API matches
        ``SyncVectorPursuitEnv``: ``reset(seeds=[…])`` → list[obs],
        ``step_joint((N,3,4))`` → list[result] with per-env auto-reset.
    """

    def __init__(
        self,
        cfg: QuadrotorPursuitConfig,
        *,
        seed=0,
        headless=True,
        native_lidar=False,
        num_envs: int = 1,
        env_spacing: float | None = None,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.cfg = replace(cfg, seed=int(seed))
        self._headless = headless
        # Native LiDAR path is single-env for now; multi-env uses geometric fallback.
        self._native_lidar_enabled = bool(native_lidar) and self.num_envs == 1
        self._env_spacing = (
            float(env_spacing)
            if env_spacing is not None
            else max(float(cfg.grid_size) + 8.0, 32.0)
        )
        self._needs_reset = True
        self._closed = False
        self._steps = np.zeros(self.num_envs, dtype=np.int32)
        self._capture = np.zeros(self.num_envs, dtype=bool)
        self._physical_potential = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._safety_reasons = [[[] for _ in range(self.cfg.n_uavs)] for _ in range(self.num_envs)]
        self._auto_reset = True
        self._physics_fail_count = 0
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

        config_dir = omnidrones_root / "cfg"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"OmniDrones Hydra config directory not found: {config_dir}")
        physics_steps = max(1, round(self.cfg.decision_dt / 0.01))
        self.physics_dt = self.cfg.decision_dt / physics_steps
        # Formation counts physics steps in progress_buf; Radar5 counts decisions.
        self._max_physics_steps = int(self.cfg.search_steps) * physics_steps
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            isaac_cfg = compose(
                config_name="train",
                overrides=[
                    "task=Formation",
                    "task.drone_model.name=Hummingbird",
                    "task.drone_model.controller=LeePositionController",
                    f"task.env.num_envs={self.num_envs}",
                    f"task.env.env_spacing={self._env_spacing}",
                    f"task.env.max_episode_length={self._max_physics_steps}",
                    "task.formation=radar5_three",
                    f"headless={'true' if headless else 'false'}",
                    "++sim.device=cuda:0",
                ],
            )
        OmegaConf.resolve(isaac_cfg)
        OmegaConf.set_struct(isaac_cfg, False)
        isaac_cfg.sim.dt = self.physics_dt
        isaac_cfg.sim.substeps = 1
        isaac_cfg.sim.gravity = [0.0, 0.0, -self.cfg.quadrotor_gravity]
        isaac_cfg.sim.device = "cuda:0"
        isaac_cfg.sim.use_gpu_pipeline = True
        # Keep False so per-env Radar5 buildings/targets added after GridCloner
        # remain independently editable; GPU PhysX still batches num_envs>1.
        isaac_cfg.sim.replicate_physics = False
        isaac_cfg.sim.use_flatcache = True
        isaac_cfg.env.num_envs = self.num_envs
        isaac_cfg.env.env_spacing = self._env_spacing
        isaac_cfg.task.env.num_envs = self.num_envs
        isaac_cfg.task.env.env_spacing = self._env_spacing
        isaac_cfg.task.env.max_episode_length = self._max_physics_steps

        self.simulation_app = _init_simulation_app(isaac_cfg)
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
        from experiments.radar5_frozen_transfer_v1.isaac_pursuit_observation import IsaacPursuitObservationBridge

        # One Radar5 logic env (KF / dual-capture reward / geometric lidar) per PhysX clone.
        # Isaac has no built-in KF-CI or k=2 capture reward — those stay in Radar5 Python.
        self.lidars = [
            IsaacLidarAdapter(3, n_targets=1, decision_dt=self.cfg.decision_dt)
            for _ in range(self.num_envs)
        ]
        self.bridges = [
            IsaacPursuitObservationBridge(
                replace(self.cfg, seed=int(seed) + index),
                sensor_adapter=self.lidars[index],
            )
            for index in range(self.num_envs)
        ]
        self.bridge = self.bridges[0]  # single-env / getattr compatibility
        self.lidar = self.lidars[0]
        self.envs = [bridge.env for bridge in self.bridges]
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
            self.sensor_paths = [None] * (self.cfg.n_uavs * self.num_envs)
        self._initialize_scene_transforms()
        self._last_native_scan = None
        self._physics_initialized = False
        self._patch_multirotor_throttle_reset()
        self._patch_formation_episode_end()
        print(
            f"[isaac] num_envs={self.num_envs} env_spacing={self._env_spacing} "
            f"replicate_physics={bool(isaac_cfg.sim.replicate_physics)}",
            flush=True,
        )
        self.reset(seed=seed)

    def _patch_multirotor_throttle_reset(self):
        """Avoid OmniDrones MultirotorBase throttle broadcast crash on Formation reset.

        ``throttle[env_ids] = rotors.f_inv(gravity/KF)`` fails for n_drones=3 /
        num_envs=1 (shape ``[3,1,12]`` vs ``[3,1]``). Replace with equal-share
        hover throttle so the first ``sim_env.reset()`` and any accidental later
        Formation resets stay safe; soft-reset already bypasses ``_reset_idx``.
        """
        drone = self.sim_env.drone
        if getattr(drone, "_radar5_throttle_reset_patched", False):
            return

        def _safe_reset_idx(env_ids, train=True):
            if env_ids is None:
                env_ids = torch.arange(drone.shape[0], device=drone.device)
            drone.thrusts[env_ids] = 0.0
            drone.torques[env_ids] = 0.0
            drone.vel[env_ids] = 0.0
            drone.acc[env_ids] = 0.0
            if train and "train" in getattr(drone, "randomization", {}):
                drone._randomize(env_ids, drone.randomization["train"])
            elif "eval" in getattr(drone, "randomization", {}):
                drone._randomize(env_ids, drone.randomization["eval"])
            try:
                total = drone.KF[env_ids].sum(-1, keepdim=True).clamp_min(1e-8)
                hover = drone.gravity[env_ids] / total
                while hover.ndim < drone.throttle.ndim:
                    hover = hover.unsqueeze(-1)
                assigned = drone.throttle[env_ids]
                hover = hover.expand_as(assigned) / max(int(drone.num_rotors), 1)
                drone.throttle[env_ids] = hover.clamp(0.0, 1.0)
            except RuntimeError:
                drone.throttle[env_ids] = 0.5
            drone.throttle_difference[env_ids] = 0.0
            return env_ids

        drone._reset_idx = _safe_reset_idx  # type: ignore[method-assign]
        drone._radar5_throttle_reset_patched = True

    def _patch_formation_episode_end(self):
        """Radar5 owns episode ends; disable Formation misbehave/NaN termination.

        Formation otherwise marks envs done when z<0.2 or drones get too close,
        which fights multi-env soft-reset and can desync bridges from PhysX.
        """
        if getattr(self.sim_env, "_radar5_done_patched", False):
            return

        def _radar5_reward_and_done():
            from tensordict.tensordict import TensorDict

            reward = torch.zeros(
                self.num_envs, self.sim_env.drone.n, 1, device=self.sim_env.device
            )
            false = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.sim_env.device)
            return TensorDict(
                {
                    "agents": {"reward": reward},
                    "done": false,
                    "terminated": false,
                    "truncated": false,
                },
                self.sim_env.batch_size,
            )

        self.sim_env._compute_reward_and_done = _radar5_reward_and_done  # type: ignore[method-assign]
        self.sim_env._radar5_done_patched = True

    @property
    def n_agents(self):
        return self.cfg.n_uavs

    @property
    def t(self):
        return int(self._steps[0]) if self.num_envs == 1 else self._steps

    @property
    def captured(self):
        return bool(self._capture[0]) if self.num_envs == 1 else self._capture

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

    def _env_offsets(self):
        return self.sim_env.envs_positions  # (N, 3) world origins of GridCloner cells

    def minimum_capture_gap(self, env_index: int = 0):
        state = self._physical_state(env_index)
        target = self._target_position(env_index)
        active = ~np.asarray(self.bridges[env_index].env.disabled_uavs, dtype=bool)
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
        for bridge in self.bridges:
            bridge.env.cfg = replace(self.cfg, seed=bridge.env.cfg.seed)

    def _has_physical_capture(self, state, target, env_index: int = 0):
        active = ~np.asarray(self.bridges[env_index].env.disabled_uavs, dtype=bool)
        distances = np.linalg.norm(state[:, :3] - target[None, :], axis=1)
        return int(np.count_nonzero(active & (distances <= self.cfg.capture_radius))) >= int(
            self.cfg.capture_required_uavs
        )

    @staticmethod
    def _episode_end_flags(captured, task_timeout, upstream_terminated, upstream_truncated):
        terminated = bool(captured or task_timeout or upstream_terminated)
        truncated = bool(upstream_truncated and not task_timeout and not terminated)
        return terminated, truncated

    def _controller_feasible_rate(self, physical, env_index: int = 0):
        """Fraction of active UAVs still upright and above the ground plane."""
        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import (
            quaternion_to_rotation,
        )

        state = np.asarray(physical, dtype=np.float64).reshape(self.cfg.n_uavs, 13)
        active = ~np.asarray(self.bridges[env_index].env.disabled_uavs, dtype=bool)
        if not np.any(active):
            return 0.0
        upright = np.zeros(self.cfg.n_uavs, dtype=bool)
        for index, quaternion in enumerate(state[:, 6:10]):
            rotation = quaternion_to_rotation(quaternion)
            upright[index] = bool(rotation[2, 2] > 0.25)
        airborne = state[:, 2] > max(0.15, 0.5 * float(self.cfg.quadrotor_clearance))
        finite = np.isfinite(state).all(axis=1)
        feasible = active & upright & airborne & finite
        return float(np.count_nonzero(feasible) / max(int(np.count_nonzero(active)), 1))

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

    @staticmethod
    def _translate_scale_ops(prim):
        """Reuse existing USD translate/scale ops (Cube.Define may pre-create them)."""
        from pxr import UsdGeom

        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        translate = next(
            (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        scale = next(
            (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if translate is None:
            translate = xform.AddTranslateOp()
        if scale is None:
            scale = xform.AddScaleOp()
        xform.SetXformOpOrder([translate, scale])
        return translate, scale

    @staticmethod
    def _translate_op(prim):
        from pxr import UsdGeom

        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        translate = next(
            (op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate is None:
            translate = xform.AddTranslateOp()
        return translate

    def _create_scene_prims(self):
        """Create ground / target / buildings under every GridCloner env cell.

        Formation already cloned ``/World/envs/env_*`` before this runs, so we
        add Radar5 props per env (not only env_0). Transforms are local to each
        env xform; drone teleports add ``envs_positions`` for world poses.
        """
        from pxr import Gf, UsdGeom, UsdPhysics
        import omni.usd
        from omni.isaac.core.utils.semantics import add_update_semantics

        stage = omni.usd.get_context().get_stage()
        self._target_paths = []
        self._target_translates = []
        self._building_ops = []  # list[env][building] -> (translate, scale)
        for env_index in range(self.num_envs):
            root = f"/World/envs/env_{env_index}"
            ground = UsdGeom.Cube.Define(stage, f"{root}/radar5_ground")
            ground.CreateDisplayColorAttr([(0.10, 0.14, 0.17)])
            ground_translate, ground_scale = self._translate_scale_ops(ground.GetPrim())
            ground_translate.Set(Gf.Vec3d(self.cfg.grid_size / 2, self.cfg.grid_size / 2, -0.08))
            ground_scale.Set(Gf.Vec3f(self.cfg.grid_size / 2, self.cfg.grid_size / 2, 0.08))

            target_path = f"{root}/radar5_target"
            target = UsdGeom.Sphere.Define(stage, target_path)
            target.CreateRadiusAttr(self.cfg.target_diameter / 2.0)
            target.CreateDisplayColorAttr([(0.95, 0.22, 0.10)])
            add_update_semantics(target.GetPrim(), "radar5_target", "class")
            UsdPhysics.CollisionAPI.Apply(target.GetPrim())
            rigid = UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
            rigid.CreateKinematicEnabledAttr(True)
            self._target_paths.append(target_path)
            self._target_translates.append(self._translate_op(target.GetPrim()))

            env_buildings = []
            for building_index in range(self.cfg.building_count):
                path = f"{root}/radar5_building_{building_index}"
                building = UsdGeom.Cube.Define(stage, path)
                building.CreateDisplayColorAttr([(0.10, 0.42, 0.52)])
                UsdPhysics.CollisionAPI.Apply(building.GetPrim())
                env_buildings.append(self._translate_scale_ops(building.GetPrim()))
            self._building_ops.append(env_buildings)
        # Back-compat aliases for single-env overlays / smoke scripts.
        self._target_path = self._target_paths[0]
        self._target_translate = self._target_translates[0]

    def _initialize_scene_transforms(self):
        from pxr import Gf

        for translate in self._target_translates:
            translate.Set(Gf.Vec3d(0.0, 0.0, self.cfg.initial_altitude))
        for env_index in range(self.num_envs):
            self._set_building_transforms(env_index)

    def _set_building_transforms(self, env_index: int = 0):
        from pxr import Gf

        env = self.bridges[env_index].env
        for (translate, scale), rect, height in zip(
            self._building_ops[env_index], env.buildings, env.building_heights
        ):
            x0, y0, x1, y1 = map(float, rect)
            translate.Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, float(height) / 2))
            scale.Set(Gf.Vec3f((x1 - x0) / 2, (y1 - y0) / 2, float(height) / 2))

    def _physical_state_batch(self):
        """Radar5 pursuer states in env-local frames, shape (N, 3, 13).

        Uses ``check_nan=False`` so one bad clone cannot abort the whole batch;
        callers recover via :meth:`_corrupt_env_mask`. Non-finite OmniDrones
        rows are mapped to NaN Radar5 rows instead of raising.
        """
        from experiments.radar5_frozen_transfer_v1.state_adapter import omnidrones_state_to_radar5

        state = self.sim_env.drone.get_state(check_nan=False)
        if isinstance(state, torch.Tensor):
            raw = state.detach()
            if not torch.isfinite(raw).all():
                # Convert finite envs normally; leave corrupt envs as NaN markers.
                raw_np = raw.detach().cpu().numpy()
            else:
                raw_np = None
                converted = omnidrones_state_to_radar5(state)
                return np.asarray(converted, dtype=np.float64).reshape(
                    self.num_envs, self.cfg.n_uavs, 13
                )
        else:
            raw_np = np.asarray(state)
        raw_np = raw_np.reshape(self.num_envs, self.cfg.n_uavs, -1)
        out = np.full((self.num_envs, self.cfg.n_uavs, 13), np.nan, dtype=np.float64)
        for env_index in range(self.num_envs):
            row = raw_np[env_index]
            if not np.isfinite(row).all():
                continue
            try:
                out[env_index] = omnidrones_state_to_radar5(row)
            except ValueError:
                continue
        return out

    def _physical_state(self, env_index: int = 0):
        return self._physical_state_batch()[env_index]

    def _corrupt_env_mask(self, batch=None):
        """True for envs with non-finite state or deeply underground drones."""
        if batch is None:
            batch = self._physical_state_batch()
        finite = np.isfinite(batch).reshape(self.num_envs, -1).all(axis=1)
        # Allow brief low altitude; only treat deep underground / explode as corrupt.
        min_z = batch[:, :, 2].min(axis=1)
        sane_z = min_z > -2.0
        max_speed = np.linalg.norm(batch[:, :, 3:6], axis=-1).max(axis=1)
        sane_speed = max_speed < 80.0
        return ~(finite & sane_z & sane_speed)

    def _sync_bridge(self, env_index: int | None = None, batch=None):
        if batch is None:
            batch = self._physical_state_batch()
        corrupt = self._corrupt_env_mask(batch)
        if env_index is None:
            for index, bridge in enumerate(self.bridges):
                if corrupt[index]:
                    continue
                bridge.sync_pursuers(batch[index])
        else:
            if not corrupt[env_index]:
                self.bridges[env_index].sync_pursuers(batch[env_index])
        return batch

    def _target_position(self, env_index: int = 0):
        env = self.bridges[env_index].env
        return np.asarray([*env.dynamic_targets[0], env.target_altitudes[0]], dtype=np.float64)

    def _write_target_pose(self, env_index: int | None = None):
        from pxr import Gf

        indices = range(self.num_envs) if env_index is None else (env_index,)
        for index in indices:
            self._target_translates[index].Set(Gf.Vec3d(*self._target_position(index).tolist()))

    def _read_native_lidar(self):
        if not self._native_lidar_enabled:
            self._last_native_scan = None
            self.bridges[0].clear_native_sensor_scan()
            return
        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import quaternion_to_rotation

        physical = self._physical_state(0)
        positions = physical[:, :3]
        rotations = np.stack([quaternion_to_rotation(row[6:10]) for row in physical])
        scan = self.lidars[0].read_native_scan(
            sim_time=float(self._steps[0]) * self.cfg.decision_dt,
            lidar_interface=self.lidar_interface,
            sensor_paths=self.sensor_paths[: self.cfg.n_uavs],
            sensor_origins=positions,
            sensor_rotations=rotations,
            target_prim_tokens=("radar5_target",),
        )
        self._last_native_scan = scan
        if scan.get("sensor_available", False):
            self.bridges[0].set_native_sensor_scan(scan)
        else:
            self.bridges[0].clear_native_sensor_scan()

    def _clear_drone_dynamics(self, env_ids=None):
        """Zero drone buffers without MultirotorBase._reset_idx (throttle shape bug)."""
        drone = self.sim_env.drone
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_env.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.sim_env.device, dtype=torch.long)
        for name in ("thrusts", "torques", "forces", "vel", "vel_w", "vel_b", "acc", "acc_w"):
            value = getattr(drone, name, None)
            if value is not None and torch.is_tensor(value):
                value[env_ids] = 0.0
        if hasattr(drone, "throttle_difference") and torch.is_tensor(drone.throttle_difference):
            drone.throttle_difference[env_ids] = 0.0
        if (
            hasattr(drone, "throttle")
            and hasattr(drone, "gravity")
            and hasattr(drone, "KF")
            and torch.is_tensor(drone.throttle)
        ):
            try:
                total = drone.KF[env_ids].sum(-1, keepdim=True).clamp_min(1e-8)
                hover = drone.gravity[env_ids] / total
                while hover.ndim < drone.throttle.ndim:
                    hover = hover.unsqueeze(-1)
                assigned = drone.throttle[env_ids]
                hover = hover.expand_as(assigned) / max(int(drone.num_rotors), 1)
                drone.throttle[env_ids] = hover.clamp(0.0, 1.0)
            except RuntimeError:
                drone.throttle[env_ids] = 0.5
        if hasattr(self.sim_env, "progress_buf") and torch.is_tensor(self.sim_env.progress_buf):
            self.sim_env.progress_buf[env_ids] = 0.0
        if hasattr(self.sim_env, "stats"):
            try:
                self.sim_env.stats[env_ids] = 0.0
            except Exception:
                pass

    def _teleport_pursuers_from_bridge(self, env_indices=None):
        # World poses = Radar5 local xyz + GridCloner env origin.
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        env_ids = torch.as_tensor(env_indices, device=self.sim_env.device, dtype=torch.long)
        offsets = self._env_offsets()[env_ids].unsqueeze(1)  # (K, 1, 3)
        locals_pos = []
        locals_quat = []
        for index in env_indices:
            physical = np.asarray(self.bridges[index].env.quadrotor_states, dtype=np.float64)
            locals_pos.append(physical[:, :3])
            locals_quat.append(physical[:, 6:10])
        positions = (
            torch.as_tensor(np.stack(locals_pos), dtype=torch.float32, device=self.sim_env.device)
            + offsets
        )
        orientations = torch.as_tensor(
            np.stack(locals_quat), dtype=torch.float32, device=self.sim_env.device
        )
        self.sim_env.drone.set_world_poses(positions, orientations, env_ids)
        self.sim_env.drone.set_velocities(
            torch.zeros((len(env_indices), self.cfg.n_uavs, 6), device=self.sim_env.device),
            env_ids,
        )
        self._clear_drone_dynamics(env_ids)

    def _reset_env_indices(self, env_indices, seeds=None):
        """Soft-reset selected parallel scenes (bridges + teleports + USD)."""
        for offset, env_index in enumerate(env_indices):
            self.lidars[env_index].reset()
            bridge = self.bridges[env_index]
            if seeds is not None:
                seed = int(seeds[offset])
                bridge.env.cfg.seed = seed
                bridge.env.rng = np.random.default_rng(seed)
            bridge.reset()
            self._steps[env_index] = 0
            self._capture[env_index] = False
            self._episode_return[env_index] = 0.0
            self._set_building_transforms(env_index)
            self._write_target_pose(env_index)
        self._teleport_pursuers_from_bridge(env_indices)
        # Flush teleported poses into the physics view before the next Lee step.
        try:
            if hasattr(self.sim_env, "sim") and hasattr(self.sim_env.sim, "forward"):
                self.sim_env.sim.forward()
        except Exception:
            pass
        self._sync_bridge()
        if hasattr(self, "_physical_potential"):
            for env_index in env_indices:
                try:
                    self._physical_potential[env_index] = self._gap_potential(
                        self.minimum_capture_gap(env_index)
                    )
                except Exception:
                    self._physical_potential[env_index] = 0.0
        if 0 in env_indices:
            try:
                self._read_native_lidar()
            except (RuntimeError, ValueError):
                self._last_native_scan = None
                self.bridges[0].clear_native_sensor_scan()

    def reset(self, seed=None, seeds=None):
        """Reset all scenes.

        Single-env: ``reset(seed=…)`` → observation dict (legacy trainers).
        Multi-env / vector trainers: ``reset(seeds=[…])`` → list of observations.
        """
        vector_api = seeds is not None or self.num_envs > 1
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        if seeds is None:
            base = int(self.cfg.seed if seed is None else seed)
            if seed is not None:
                self.cfg.seed = int(seed)
            seeds = [base + i for i in range(self.num_envs)]
        if not self._physics_initialized:
            self.td = self.sim_env.reset()
            self._physics_initialized = True
        elif hasattr(self.sim_env, "progress_buf"):
            self.sim_env.progress_buf.zero_()
        self._reset_env_indices(list(range(self.num_envs)), seeds=seeds)
        self._needs_reset = False
        observations = [bridge.observe() for bridge in self.bridges]
        return observations if vector_api else observations[0]

    def observe_search(self):
        if self._needs_reset:
            self.reset()
        self._sync_bridge()
        obs = [bridge.observe() for bridge in self.bridges]
        return obs[0] if self.num_envs == 1 else obs

    def coverage(self):
        return float(getattr(self.bridges[0].env, "coverage", lambda: 0.0)())

    def _normalize_actions(self, actions):
        action = np.asarray(actions, dtype=np.float32)
        if action.ndim == 2:
            if self.num_envs != 1 or action.shape != (self.cfg.n_uavs, 4):
                raise ValueError(
                    f"single-env action must be ({self.cfg.n_uavs}, 4); got {action.shape}"
                )
            return action[None, ...], True
        if action.shape != (self.num_envs, self.cfg.n_uavs, 4):
            raise ValueError(
                f"vector action must be ({self.num_envs}, {self.cfg.n_uavs}, 4); got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("Isaac pursuit action must be finite")
        return action, False

    def _step_physics_batch(self, velocity_batch, yaw_batch):
        """Advance all envs one decision step with batched Lee velocity tracking."""
        velocity_tensor = torch.as_tensor(
            np.nan_to_num(velocity_batch, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float32,
            device=self.sim_env.device,
        )
        yaw_tensor = torch.as_tensor(
            np.nan_to_num(yaw_batch, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float32,
            device=self.sim_env.device,
        )
        physics_steps = max(1, round(self.cfg.decision_dt / self.physics_dt))
        for _ in range(physics_steps):
            state = self.sim_env.drone.get_state(check_nan=False)
            # Hold current xyz as position setpoints so Lee tracks velocity only.
            # Replace non-finite entries so controller.compute cannot explode.
            if not torch.isfinite(state).all():
                state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            target_pos = state[..., :3].clone()
            with torch.no_grad():
                motor_action = self.sim_env.controller.compute(
                    state[..., :13],
                    target_pos=target_pos,
                    target_vel=velocity_tensor,
                    target_yaw=yaw_tensor,
                ).clamp(-1.0, 1.0)
                if not torch.isfinite(motor_action).all():
                    motor_action = torch.nan_to_num(
                        motor_action, nan=0.0, posinf=0.0, neginf=0.0
                    ).clamp(-1.0, 1.0)
            self.td = self.sim_env.step(self.td.set(("agents", "action"), motor_action))

    def _gap_potential(self, gap):
        scale = max(float(self.cfg.approach_distance_scale), 1e-6)
        return float(np.exp(-max(float(gap), 0.0) / scale))

    def _apply_physical_gap_shaping(self, env_index, gap, terminal):
        """Discount-consistent potential on the physical k-th-UAV gap.

        Kinematic approach terms are removed by the caller. This term is zero
        at a true physical terminal, so it only assigns credit for closing the
        last meters of the body, not the kinematic proxy.
        """
        if not hasattr(self, "_physical_potential"):
            self._physical_potential = np.zeros(self.num_envs, dtype=np.float64)
        nxt = 0.0 if terminal else self._gap_potential(gap)
        previous = float(self._physical_potential[env_index])
        delta = float(self.cfg.capture_team_progress_weight) * (
            float(self.cfg.reward_gamma) * nxt - previous
        )
        self._physical_potential[env_index] = nxt
        return float(delta)

    def _finalize_env_result(self, env_index, result, safety_reasons, *, physics_failed=False):
        physical = self._physical_state(env_index)
        if physics_failed or not np.isfinite(physical).all():
            # Emit a safe truncated transition; caller will soft-reset the clone.
            result = dict(result)
            result["reward"] = float(result.get("reward", 0.0))
            result["reward_components"] = dict(result.get("reward_components", {}))
            result["reward_components"]["isaac_physics_failure"] = -1.0
            result["reward"] -= 1.0
            result["capture_success"] = False
            result["minimum_capture_gap"] = float(self.cfg.grid_size)
            result["safety_reasons"] = safety_reasons
            result["controller_feasible_rate"] = 0.0
            result["execution_mode"] = "isaac_lee_velocity"
            result["sensor_mode"] = (
                "isaac_lidar" if self._native_lidar_enabled else "geometric_lidar_fallback"
            )
            result["quadrotor_states"] = np.nan_to_num(physical, nan=0.0).tolist()
            result["terminated"] = False
            result["truncated"] = True
            result["physics_failed"] = True
            self._capture[env_index] = False
            self._safety_reasons[env_index] = safety_reasons
            self._episode_return[env_index] += float(result["reward"])
            return result

        actual_gap = self.minimum_capture_gap(env_index)
        capture = self._has_physical_capture(physical, self._target_position(env_index), env_index)
        # The kinematic bridge can enter the capture sphere before the Hummingbird
        # does. That proxy must not end the episode or count as a capture.
        components = dict(result.get("reward_components", {}))
        kinematic_capture_bonus = float(components.get("capture", 0.0))
        result["reward"] = float(result.get("reward", 0.0))
        if kinematic_capture_bonus and not capture:
            result["reward"] -= kinematic_capture_bonus
            components["capture"] = 0.0
            components["kinematic_capture_ignored"] = kinematic_capture_bonus
        for key in (
            "individual_approach",
            "capture_team_progress",
            "nearest_approach",
            "encirclement_progress",
        ):
            amount = float(components.get(key, 0.0))
            if amount:
                result["reward"] -= amount
                components[key] = 0.0
        already_physical = bool(self._capture[env_index])
        self._capture[env_index] = bool(capture)
        result["capture_success"] = bool(capture)
        result["reward_components"] = components
        result["minimum_capture_gap"] = actual_gap
        result["safety_reasons"] = safety_reasons
        result["controller_feasible_rate"] = self._controller_feasible_rate(physical, env_index)
        result["execution_mode"] = "isaac_lee_velocity"
        result["sensor_mode"] = (
            "isaac_lidar" if self._native_lidar_enabled else "geometric_lidar_fallback"
        )
        result["quadrotor_states"] = physical.tolist()
        task_timeout = int(self._steps[env_index]) >= self.cfg.search_steps and not self._capture[env_index]
        result["terminated"], result["truncated"] = self._episode_end_flags(
            self._capture[env_index],
            task_timeout,
            False,
            bool(result.get("truncated", False)),
        )
        result["reward"] = float(result["reward"])
        # Pay the capture bonus on the first physical entry. The kinematic
        # bonus was removed above when the body was still outside the sphere.
        pay_physical_capture = bool(capture and not already_physical and kinematic_capture_bonus <= 0.0)
        result["reward_components"]["isaac_capture_reconciliation"] = (
            float(pay_physical_capture) * self.cfg.capture_reward
        )
        if pay_physical_capture:
            result["reward"] += self.cfg.capture_reward
        terminal = bool(result["terminated"])
        shaping = self._apply_physical_gap_shaping(env_index, actual_gap, terminal)
        result["reward_components"]["physical_capture_progress"] = shaping
        result["reward"] += shaping
        result["physics_failed"] = False
        self._safety_reasons[env_index] = safety_reasons
        self._episode_return[env_index] += float(result["reward"])
        return result

    def step_joint(self, actions):
        action_batch, single = self._normalize_actions(actions)
        if self._needs_reset and self.num_envs == 1:
            raise RuntimeError("step_joint called after episode end; call observe_search to reset")

        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import yaw_from_quaternion
        from experiments.radar5_frozen_transfer_v1.sim2real_contract import policy_action_to_setpoint

        self._sync_bridge()
        velocity_batch = np.zeros((self.num_envs, self.cfg.n_uavs, 3), dtype=np.float32)
        yaw_batch = np.zeros((self.num_envs, self.cfg.n_uavs), dtype=np.float32)
        safety_all = []
        for env_index, bridge in enumerate(self.bridges):
            measured_yaw = np.asarray(
                [yaw_from_quaternion(q) for q in bridge.env.quadrotor_states[:, 6:10]],
                dtype=np.float32,
            )
            requested_velocity, target_yaw = policy_action_to_setpoint(
                action_batch[env_index],
                measured_yaw,
                decision_dt=self.cfg.decision_dt,
                max_horizontal_velocity=self.cfg.max_horizontal_velocity,
                max_vertical_velocity=self.cfg.max_vertical_velocity,
                max_yaw_rate=self.cfg.max_reference_yaw_rate,
            )
            requested_velocity, safety_reasons = bridge.filter_references(requested_velocity)
            velocity_batch[env_index] = requested_velocity.astype(np.float32)
            yaw_batch[env_index] = np.asarray(target_yaw, dtype=np.float32)
            safety_all.append(safety_reasons)

        self._step_physics_batch(velocity_batch, yaw_batch)
        batch_states = self._physical_state_batch()
        corrupt = self._corrupt_env_mask(batch_states)
        if np.any(corrupt):
            self._physics_fail_count += int(np.count_nonzero(corrupt))
            print(
                f"[isaac] physics corrupt envs={np.flatnonzero(corrupt).tolist()} "
                f"total_fails={self._physics_fail_count}",
                flush=True,
            )
        self._sync_bridge(batch=batch_states)

        results = []
        done_indices = []
        for env_index, bridge in enumerate(self.bridges):
            physics_failed = bool(corrupt[env_index])
            if physics_failed:
                # Skip Radar5 advance on corrupted physics; emit truncated failure.
                result = {
                    "reward": 0.0,
                    "reward_components": {},
                    "capture_success": False,
                    "terminated": False,
                    "truncated": True,
                    "obs": bridge.observe(),
                }
            else:
                result = bridge.advance()
                self._write_target_pose(env_index)
                # Pursuer states were synced before advance; do not rebuild the
                # expensive hetero-graph observation a second time.
                if result.get("obs") is None:
                    bridge.sync_pursuers(batch_states[env_index])
                    result["obs"] = bridge.observe()
                if env_index == 0:
                    self._read_native_lidar()
            self._steps[env_index] += 1
            result = self._finalize_env_result(
                env_index, result, safety_all[env_index], physics_failed=physics_failed
            )
            done = bool(result["terminated"] or result["truncated"])
            result["_episode_done"] = done
            if done and self._auto_reset and self.num_envs > 1:
                result["info"] = {
                    "final_obs": result["obs"],
                    "final_terminated": bool(result["terminated"]),
                    "final_truncated": bool(result["truncated"]),
                    "auto_reset": True,
                    "capture_success": bool(result.get("capture_success", False)),
                    "episode_return": float(self._episode_return[env_index]),
                    "physics_failed": bool(result.get("physics_failed", False)),
                }
                done_indices.append(env_index)
            elif done and self.num_envs == 1:
                self._needs_reset = True
                result["info"] = {
                    "auto_reset": False,
                    "episode_return": float(self._episode_return[env_index]),
                }
            else:
                result["info"] = {
                    "auto_reset": False,
                    "episode_return": float(self._episode_return[env_index]),
                }
            results.append(result)

        if done_indices:
            # Stagger seeds so auto-resets do not clone the same episode.
            next_seeds = [
                int(self.cfg.seed) * 1_000_003
                + int(self._physics_fail_count) * 997
                + int(self._steps.sum()) * 17
                + env_index
                for env_index in done_indices
            ]
            self._reset_env_indices(done_indices, seeds=next_seeds)
            for env_index in done_indices:
                results[env_index]["obs"] = self.bridges[env_index].observe()
                results[env_index]["terminated"] = False
                results[env_index]["truncated"] = False

        return results[0] if single else results

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