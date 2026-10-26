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
    # Keep Kit defaults close to OmniDrones/Isaac; only bump window for GUI demos.
    config = {"headless": headless, "anti_aliasing": 1}
    if not headless:
        config["width"] = 1440
        config["height"] = 900
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
        native_lidar: bool | None = None,
        num_envs: int = 1,
        env_spacing: float | None = None,
        visual_model_scale: float | None = None,
        physx_closed_loop: bool = False,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.cfg = replace(cfg, seed=int(seed))
        self._headless = headless
        # Pure PhysX 3 pursuers + 1 same-model evader (see SIM2REAL.md).
        self._physx_closed_loop = bool(physx_closed_loop)
        self._n_physics_drones = 4 if self._physx_closed_loop else int(cfg.n_uavs)
        self._evader_drone_index = 3 if self._physx_closed_loop else None
        # Pure closed-loop still uses a PhysX Hummingbird body for the evader
        # (collision / capture / state tensor). Lee velocity tracking on that
        # 4th body is too unstable for dual-capture (FRPN ~0–8%); exact pose
        # integration on the articulation recovers ~30%+ and is the supported
        # path. Set False only to experiment with Lee-on-evader again.
        self._kinematic_evader = bool(self._physx_closed_loop)
        # Lee altitude setpoint for the PhysX evader (updated on each reset).
        self._evader_alt_hold = float(self.cfg.initial_altitude)
        # GUI-only mesh enlargement; PhysX / capture radii stay on the real model.
        # ~4×: readable from chase cam without drowning the scene / occluding cover.
        if visual_model_scale is None:
            self._visual_model_scale = 4.0 if not headless else 1.0
        else:
            self._visual_model_scale = max(float(visual_model_scale), 1e-3)
        # PhysX RangeSensor does not register under OmniDrones' GPU pipeline
        # (USD Lidar prim exists, is_lidar_sensor=False, clouds stay empty).
        # Default both hybrid and closed to geometric LiDAR; pass
        # native_lidar=True only for experiments that can use a CPU physics scene.
        if native_lidar is None:
            self._native_lidar_enabled = False
        else:
            self._native_lidar_enabled = bool(native_lidar)
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
                    (
                        "task.formation=radar5_three_plus_evader"
                        if self._physx_closed_loop
                        else "task.formation=radar5_three"
                    ),
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
        # OmniDrones Formation requires the GPU pipeline; PhysX RangeSensor then
        # often fails to register when parented under GPU articulations. Lidars
        # are created as world-fixed Xforms and pose-synced each step instead.
        isaac_cfg.sim.use_gpu_pipeline = True
        isaac_cfg.sim.use_gpu = True
        # Keep False so per-env Radar5 buildings/targets added after GridCloner
        # remain independently editable; GPU PhysX still batches num_envs>1.
        isaac_cfg.sim.replicate_physics = False
        # GUI: disable flatcache so Viewport meshes track PhysX/Fabric poses.
        # Training keeps flatcache for throughput.
        isaac_cfg.sim.use_flatcache = bool(headless)
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
        # Index 3 = same-model PhysX evader (pure closed loop).
        # Stay clear of the pursuer triangle (avoids Lee collision blow-ups) but
        # keep spawn range inside lidar_target_detection_range so native scans
        # can hit at t=0. spacing*4≈8.8 → ~12.5 m diagonal vs range 13 m left
        # almost no margin; spacing*2.5≈5.5 → ~7.8 m is safe.
        evader_xy = max(float(self.cfg.formation_spacing) * 2.5, 5.5)
        formation_task.FORMATIONS["radar5_three_plus_evader"] = [
            [0.0, 0.0, self.cfg.initial_altitude],
            [self.cfg.formation_spacing, 0.0, self.cfg.initial_altitude],
            [0.0, self.cfg.formation_spacing, self.cfg.initial_altitude],
            [evader_xy, evader_xy, self.cfg.initial_altitude],
        ]
        self.isaac_cfg = isaac_cfg
        try:
            self.sim_env = Formation(isaac_cfg, headless=headless)
        except Exception:
            self.simulation_app.close()
            raise
        self.sim_env.controller = self.sim_env.controller.to(self.sim_env.device)
        if int(self.sim_env.drone.n) != int(self._n_physics_drones):
            raise RuntimeError(
                f"Expected {self._n_physics_drones} PhysX drones, got {self.sim_env.drone.n}"
            )

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
                external_target_physics=self._physx_closed_loop,
            )
            for index in range(self.num_envs)
        ]
        self.bridge = self.bridges[0]  # single-env / getattr compatibility
        self.lidar = self.lidars[0]
        self.envs = [bridge.env for bridge in self.bridges]
        self._create_scene_prims()
        # sensor_paths layout: env-major, pursuers only → length num_envs * n_uavs.
        # With physx_closed (4 articulations/env), skip the evader slot (index 3).
        self.sensor_paths = [None] * (self.cfg.n_uavs * self.num_envs)
        self._sensor_creation_errors = []
        self.lidar_interface = None
        if self._native_lidar_enabled:
            from omni.isaac.range_sensor import _range_sensor

            self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
            prim_paths = [str(p).rstrip("/") for p in self.sim_env.drone.prim_paths]
            n_phys = int(self._n_physics_drones)
            expected = self.num_envs * n_phys
            if len(prim_paths) != expected:
                print(
                    f"[isaac] warning: prim_paths={len(prim_paths)} expected={expected}; "
                    "native lidar indexing may be partial",
                    flush=True,
                )
            # Parent lidars under the env Xform (not the GPU articulation).
            # Parenting under Hummingbird/base_link left type=Lidar USD prims that
            # the range_sensor plugin never registered (alive=False, empty clouds).
            for env_index in range(self.num_envs):
                for drone_index in range(self.cfg.n_uavs):
                    slot = env_index * self.cfg.n_uavs + drone_index
                    parent = f"/World/envs/env_{env_index}"
                    leaf = f"radar5_lidar_uav{drone_index}"
                    try:
                        self.sensor_paths[slot] = self._create_lidar(leaf, parent)
                    except Exception as exc:  # pragma: no cover - simulator-dependent fallback
                        self._sensor_creation_errors.append((f"{parent}/{leaf}", exc))
                        self.sensor_paths[slot] = None
            created = sum(1 for p in self.sensor_paths if p is not None)
            print(
                f"[isaac] native_lidar sensors={created}/{len(self.sensor_paths)} "
                f"(world-fixed under env; physx_closed={self._physx_closed_loop})",
                flush=True,
            )
            self._lidar_pose_ops = []
            try:
                from pxr import UsdGeom
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                for path in self.sensor_paths:
                    if path is None:
                        self._lidar_pose_ops.append(None)
                        continue
                    prim = stage.GetPrimAtPath(path)
                    xform = UsdGeom.Xformable(prim)
                    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
                    if "xformOp:translate" in ops:
                        translate = ops["xformOp:translate"]
                    else:
                        translate = xform.AddTranslateOp()
                    if "xformOp:orient" in ops:
                        orient = ops["xformOp:orient"]
                    else:
                        orient = xform.AddOrientOp()
                    self._lidar_pose_ops.append((translate, orient))
            except Exception as exc:
                print(f"[isaac] lidar pose ops unavailable: {exc}", flush=True)
                self._lidar_pose_ops = [None] * len(self.sensor_paths)
        self._initialize_scene_transforms()
        self._apply_gui_visual_enlargement()
        if not headless:
            # Dome/key lights + compact pad framing (must run after scene prims).
            self._apply_viewport_presentation()
        self._last_native_scan = None
        self._physics_initialized = False
        self._lidar_warmed = False
        self._patch_multirotor_throttle_reset()
        self._patch_formation_episode_end()
        # Formation reset first so articulations exist, then warm LiDAR, then task reset.
        print(
            f"[isaac] num_envs={self.num_envs} env_spacing={self._env_spacing} "
            f"replicate_physics={bool(isaac_cfg.sim.replicate_physics)} "
            f"gpu_pipeline={bool(isaac_cfg.sim.use_gpu_pipeline)} "
            f"native_lidar={self._native_lidar_enabled} "
            f"physx_closed={self._physx_closed_loop} "
            f"kinematic_evader={self._kinematic_evader} "
            f"physics_drones={self._n_physics_drones} "
            f"compact_pad={not headless} "
            f"visual_model_scale={self._visual_model_scale}",
            flush=True,
        )
        self.td = self.sim_env.reset()
        self._warmup_native_lidars()
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
        from capture_geometry import angular_span_from_cfg, evaluate_capture

        state = self._physical_state(env_index)
        target = self._target_position(env_index)
        active = ~np.asarray(self.bridges[env_index].env.disabled_uavs, dtype=bool)
        result = evaluate_capture(
            state[:, :3],
            target,
            radius=float(self.cfg.capture_radius),
            required_uavs=int(self.cfg.capture_required_uavs),
            angular_span_deg=angular_span_from_cfg(self.cfg),
            active_mask=active,
        )
        if not np.isfinite(result.kth_gap) or result.kth_gap >= 1e8:
            return float(self.cfg.grid_size)
        return float(result.kth_gap)

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
        from capture_geometry import angular_span_from_cfg, evaluate_capture

        active = ~np.asarray(self.bridges[env_index].env.disabled_uavs, dtype=bool)
        result = evaluate_capture(
            np.asarray(state, dtype=float).reshape(-1, 13)[:, :3],
            target,
            radius=float(self.cfg.capture_radius),
            required_uavs=int(self.cfg.capture_required_uavs),
            angular_span_deg=angular_span_from_cfg(self.cfg),
            active_mask=active,
        )
        return bool(result.success)

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
        """Create lidar under ``parent``; ``path`` may be absolute or leaf name."""
        from experiments.radar5_frozen_transfer_v1.isaac_sensor_bridge import IsaacLidarAdapter

        parent = str(parent).rstrip("/")
        # Isaac RangeSensorCreateLidar expects a leaf name (or absolute). Prefer leaf.
        leaf = str(path).rstrip("/").rsplit("/", 1)[-1]
        # Non-zero rotation_rate is required for PhysX lidar to emit clouds;
        # rotation_rate=0 previously left sensors registered but unreadably empty.
        scan_hz = float(getattr(self.cfg, "lidar_scan_hz", 10.0) or 10.0)
        IsaacLidarAdapter.create_physx_lidar(
            leaf,
            parent,
            min_range=self.cfg.lidar_min_range,
            max_range=self.cfg.lidar_target_detection_range,
            horizontal_fov=self.cfg.lidar_horizontal_fov_deg,
            vertical_fov=self.cfg.lidar_vertical_max_deg - self.cfg.lidar_vertical_min_deg,
            horizontal_resolution=2.0,
            vertical_resolution=2.0,
            rotation_rate=max(scan_hz, 1.0),
        )
        return f"{parent}/{leaf}"

    def _warmup_native_lidars(self, physics_steps: int = 48):
        """Play timeline and step PhysX so range sensors register before reads."""
        if not self._native_lidar_enabled:
            return
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_stopped() or not timeline.is_playing():
                timeline.play()
        except Exception:
            pass
        # Prefer raw physics steps; fall back to zero-action env steps.
        stepped = 0
        try:
            sim = getattr(self.sim_env, "sim", None)
            if sim is not None and hasattr(sim, "step"):
                for _ in range(max(int(physics_steps), 1)):
                    sim.step(render=False)
                    stepped += 1
        except Exception as exc:
            print(f"[isaac] lidar warmup sim.step soft-fail: {exc}", flush=True)
        if stepped == 0:
            try:
                import torch

                n_phys = int(self._n_physics_drones)
                zeros = torch.zeros(
                    self.num_envs, n_phys, 4, device=self.sim_env.device, dtype=torch.float32
                )
                for _ in range(max(int(physics_steps), 1)):
                    if getattr(self, "td", None) is None:
                        self.td = self.sim_env.reset()
                    self.td = self.sim_env.step(self.td.set(("agents", "action"), zeros))
                    stepped += 1
            except Exception as exc:
                print(f"[isaac] lidar warmup env.step soft-fail: {exc}", flush=True)
        # Probe whether the interface recognizes created paths.
        alive = 0
        pts_total = 0
        try:
            iface = self.lidar_interface
            for path in self.sensor_paths:
                if path is None:
                    continue
                try:
                    registered = bool(
                        hasattr(iface, "is_lidar_sensor") and iface.is_lidar_sensor(path)
                    )
                    data = iface.get_point_cloud_data(path)
                    n_pts = int(np.asarray(data, dtype=float).size // 3)
                    pts_total += n_pts
                    # Empty buffers without registration are not "alive".
                    if registered or n_pts > 0:
                        alive += 1
                except Exception:
                    pass
        except Exception:
            pass
        print(
            f"[isaac] lidar warmup steps={stepped} sensors_alive={alive}/"
            f"{sum(1 for p in self.sensor_paths if p)} pts={pts_total} "
            f"gpu_pipeline={bool(self.isaac_cfg.sim.use_gpu_pipeline)}",
            flush=True,
        )
        self._lidar_warmed = True

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

    @staticmethod
    def _bind_building_material(stage, prim_path, color, opacity=1.0):
        """Matte building / pad material for physical-sim look (RTX)."""
        try:
            from pxr import Gf, Sdf, UsdShade
        except Exception:
            return
        mat_path = f"{prim_path}/Looks/BuildingMat"
        shader_path = f"{mat_path}/Shader"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        alpha = float(np.clip(opacity, 0.05, 1.0))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(alpha)
        # Higher roughness + slight darkening avoids top-down washout under dome light.
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(material)

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
        # Near-black slate — stays dark under dome light, clear in top-down.
        building_palette = [
            (0.08, 0.09, 0.11),
            (0.06, 0.07, 0.09),
            (0.10, 0.09, 0.08),
            (0.07, 0.08, 0.10),
        ]
        # Fully opaque physical buildings (GUI + headless).
        building_opacity = 1.0
        # Pad only slightly beyond the Radar5 grid (25 m + 1.5 m each side).
        ground_margin = 1.5
        ground_extent = float(self.cfg.grid_size) + 2.0 * ground_margin
        ground_half = ground_extent / 2.0
        ground_center = float(self.cfg.grid_size) / 2.0
        # Shrink Isaac's huge ground to our map instead of deleting it (avoids black void).
        if not self._headless:
            self._fit_default_ground_plane(stage)
        for env_index in range(self.num_envs):
            root = f"/World/envs/env_{env_index}"
            ground = UsdGeom.Cube.Define(stage, f"{root}/radar5_ground")
            # Bright pad on top of the fitted Isaac ground (map + margin only).
            ground.CreateDisplayColorAttr([(0.86, 0.89, 0.94)])
            ground_translate, ground_scale = self._translate_scale_ops(ground.GetPrim())
            # Slightly above Isaac ground so the pale pad is what you see in-map.
            ground_translate.Set(Gf.Vec3d(ground_center, ground_center, 0.02))
            ground_scale.Set(Gf.Vec3f(ground_half, ground_half, 0.02))
            # PhysX LiDAR needs collision geometry. Without this the pad is
            # visual-only and native scans return empty clouds in headless.
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
            if not self._headless:
                self._bind_building_material(
                    stage, f"{root}/radar5_ground", (0.86, 0.89, 0.94), 1.0
                )

            target_path = f"{root}/radar5_target"
            target = UsdGeom.Sphere.Define(stage, target_path)
            # Collision / semantic sphere for PhysX LiDAR. Capture geometry still
            # uses articulation poses + cfg.target_diameter, not this radius.
            # At ~8–13 m range a 0.25 m sphere is often missed by 2° beams.
            visual_r = max(float(self.cfg.target_diameter) / 2.0, 0.55)
            if not self._headless:
                visual_r = max(visual_r, 0.65)
            target.CreateRadiusAttr(float(visual_r))
            target.CreateDisplayColorAttr([(0.92, 0.12, 0.12)])
            add_update_semantics(target.GetPrim(), "radar5_target", "class")
            # Also stamp type=radar5_target — some Isaac builds expose the type
            # token in get_prim_data instead of the class label / prim path.
            try:
                add_update_semantics(target.GetPrim(), "radar5_target", "type")
            except Exception:
                pass
            UsdPhysics.CollisionAPI.Apply(target.GetPrim())
            rigid = UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
            rigid.CreateKinematicEnabledAttr(True)
            self._target_paths.append(target_path)
            self._target_translates.append(self._translate_op(target.GetPrim()))

            env_buildings = []
            for building_index in range(self.cfg.building_count):
                path = f"{root}/radar5_building_{building_index}"
                color = building_palette[building_index % len(building_palette)]
                building = UsdGeom.Cube.Define(stage, path)
                building.CreateDisplayColorAttr([color])
                building.CreateDisplayOpacityAttr([float(building_opacity)])
                if not self._headless:
                    self._bind_building_material(stage, path, color, building_opacity)
                UsdPhysics.CollisionAPI.Apply(building.GetPrim())
                env_buildings.append(self._translate_scale_ops(building.GetPrim()))
            self._building_ops.append(env_buildings)
        # Back-compat aliases for single-env overlays / smoke scripts.
        self._target_path = self._target_paths[0]
        self._target_translate = self._target_translates[0]

    def _apply_gui_visual_enlargement(self) -> None:
        """Enlarge Hummingbird body meshes for GUI only (capture radii / control unchanged).

        Scales USD Mesh geom under each drone. Lee / capture still use bridge or
        articulation root poses, not mesh extent; target keeps its original sphere model.
        """
        if self._headless or abs(self._visual_model_scale - 1.0) < 1e-6:
            return
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
        except Exception as exc:
            print(f"[isaac] visual enlargement skipped: {exc}", flush=True)
            return
        stage = omni.usd.get_context().get_stage()
        scale = float(self._visual_model_scale)
        scaled = 0

        def _set_scale(prim) -> bool:
            nonlocal scaled
            xform = UsdGeom.Xformable(prim)
            scale_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            if scale_op is None:
                scale_op = xform.AddScaleOp()
            try:
                current = scale_op.Get()
                if current is None:
                    current = Gf.Vec3f(1.0, 1.0, 1.0)
                # Idempotent-ish: if already magnified, leave it.
                if float(current[0]) >= scale * 0.9:
                    return False
                scale_op.Set(
                    Gf.Vec3f(
                        float(current[0]) * scale,
                        float(current[1]) * scale,
                        float(current[2]) * scale,
                    )
                )
                scaled += 1
                return True
            except Exception:
                return False

        for drone_path in self.sim_env.drone.prim_paths:
            root = stage.GetPrimAtPath(str(drone_path))
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    _set_scale(prim)
        print(
            f"[isaac] GUI Hummingbird mesh scale={scale} meshes={scaled} "
            f"(target model original; capture radii unchanged)",
            flush=True,
        )

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

    def _fit_default_ground_plane(self, stage) -> None:
        """Keep Isaac ground visible (for lighting) but cover it with our compact pad.

        Fully hiding ``defaultGroundPlane`` previously left a black void under RTX.
        The bright ``radar5_ground`` pad (grid + margin) sits on top; chase-cam
        framing is scaled to ``grid_size`` so the arena does not feel infinite.
        """
        try:
            from pxr import UsdGeom

            for path in (
                "/World/defaultGroundPlane",
                "/World/defaultGroundPlane/GroundPlane",
                "/World/defaultGroundPlane/CollisionPlane",
            ):
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                try:
                    UsdGeom.Imageable(prim).MakeVisible()
                except Exception:
                    pass
            print(
                f"[isaac] ground: Isaac plane + compact pad "
                f"({float(self.cfg.grid_size):.0f}m+1.5m margin)",
                flush=True,
            )
        except Exception as exc:
            print(f"[isaac] fit defaultGroundPlane skipped: {exc}", flush=True)

    def _apply_viewport_presentation(self) -> None:
        """Dome + key light and a quieter stage for GUI demos (no-op if USD fails)."""
        try:
            import omni.usd
            from pxr import Gf, UsdGeom, UsdLux
        except Exception as exc:  # pragma: no cover - depends on Kit
            print(f"[isaac] viewport presentation skipped: {exc}", flush=True)
            return
        stage = omni.usd.get_context().get_stage()
        self._fit_default_ground_plane(stage)
        mid = float(self.cfg.grid_size) * 0.5
        alt = float(self.cfg.initial_altitude)
        try:
            origin = self._world_origin(0)
        except Exception:
            origin = np.zeros(3, dtype=float)

        dome_path = "/World/radar5_dome_light"
        if not stage.GetPrimAtPath(dome_path):
            dome = UsdLux.DomeLight.Define(stage, dome_path)
            # Bright dome so pale buildings / pad are not a black void.
            dome.CreateIntensityAttr(1600.0)
            dome.CreateColorAttr(Gf.Vec3f(0.90, 0.93, 0.98))
        else:
            try:
                stage.GetPrimAtPath(dome_path).GetAttribute("inputs:intensity").Set(1600.0)
            except Exception:
                pass

        key_path = "/World/radar5_key_light"
        if not stage.GetPrimAtPath(key_path):
            key = UsdLux.DistantLight.Define(stage, key_path)
            key.CreateIntensityAttr(2400.0)
            key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
            key.CreateAngleAttr(1.2)
            xform = UsdGeom.Xformable(key.GetPrim())
            rotate = xform.AddRotateXYZOp()
            rotate.Set(Gf.Vec3f(-42.0, 28.0, 8.0))

        fill_path = "/World/radar5_fill_light"
        if not stage.GetPrimAtPath(fill_path):
            fill = UsdLux.DistantLight.Define(stage, fill_path)
            fill.CreateIntensityAttr(900.0)
            fill.CreateColorAttr(Gf.Vec3f(0.80, 0.86, 0.95))
            xform = UsdGeom.Xformable(fill.GetPrim())
            rotate = xform.AddRotateXYZOp()
            rotate.Set(Gf.Vec3f(-18.0, -55.0, 0.0))

        try:
            from omni.isaac.core.utils.viewports import set_camera_view

            # Frame the Radar5 pad (grid + margin), not the old infinite arena.
            g = float(self.cfg.grid_size)
            focus = [float(origin[0] + mid), float(origin[1] + mid), float(origin[2] + alt)]
            set_camera_view(
                eye=[focus[0] + 0.65 * g, focus[1] - 0.55 * g, focus[2] + 0.40 * g],
                target=focus,
            )
        except Exception as exc:
            print(f"[isaac] initial camera skipped: {exc}", flush=True)

        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set("/app/viewport/grid/enabled", False)
            # Mild translucency for GUI peek-through; still looks like solid concrete.
            try:
                settings.set("/rtx/translucency/enabled", True)
            except Exception:
                pass
        except Exception:
            pass
        print(
            f"[isaac] viewport presentation applied "
            f"(pad={float(self.cfg.grid_size):.0f}m+margin, solid buildings)",
            flush=True,
        )

    def _world_origin(self, env_index: int = 0) -> np.ndarray:
        """GridCloner cell origin in world XYZ for ``env_index``."""
        offsets = self._env_offsets()
        row = offsets[env_index]
        if torch.is_tensor(row):
            return row.detach().float().cpu().numpy().reshape(3)
        return np.asarray(row, dtype=np.float64).reshape(3)

    # Named chase-camera presets as fractions of grid_size (fits 25 m pad + margin).
    CAMERA_PRESET_FRAC = {
        # Elevated diagonal side — stays around the task map, not a huge arena.
        "side": (0.65, -0.55, 0.40),
        "side_left": (-0.65, -0.55, 0.40),
        "oblique": (0.55, -0.70, 0.48),
        "high": (0.20, -0.28, 0.85),
        # Overhead with slight tilt so roof edges / walls stay readable (not washed flat).
        "top": (0.18, -0.18, 1.10),
    }

    def camera_eye_target(
        self,
        *,
        env_index: int = 0,
        eye_offset: tuple[float, float, float] | None = None,
        view: str = "side",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(eye, focus)`` for the named chase-camera preset, or None if invalid."""
        physical = np.asarray(self._physical_state(env_index), dtype=float)
        target = np.asarray(self._target_position(env_index), dtype=float)
        # PhysX blow-ups leave NaNs; fall back to Radar5 bridge kinematics so the
        # camera keeps tracking through recovery instead of freezing on buildings.
        if not np.isfinite(physical).all():
            physical = np.asarray(
                self.bridges[env_index].env.quadrotor_states, dtype=float
            )
        if not np.isfinite(physical).all() or not np.isfinite(target).all():
            return None
        origin = self._world_origin(env_index)
        # Bias focus slightly toward the target so the chase stays centered while
        # pursuers remain in frame; keeps camera out of building corners.
        pursuer_mean = physical[:, :3].mean(axis=0)
        focus = 0.40 * pursuer_mean + 0.60 * target + origin
        if eye_offset is None:
            g = float(self.cfg.grid_size)
            frac = self.CAMERA_PRESET_FRAC.get(str(view), self.CAMERA_PRESET_FRAC["side"])
            eye_offset = (frac[0] * g, frac[1] * g, frac[2] * g)
        eye = focus + np.asarray(eye_offset, dtype=float)
        return eye, focus

    def follow_camera(
        self,
        *,
        env_index: int = 0,
        eye_offset: tuple[float, float, float] | None = None,
        view: str = "side",
    ) -> None:
        """Chase cam. Default ``side`` keeps height readable and cuts occlusion."""
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
        except Exception:
            return
        pair = self.camera_eye_target(
            env_index=env_index, eye_offset=eye_offset, view=view
        )
        if pair is None:
            return
        eye, focus = pair
        try:
            set_camera_view(eye=eye.tolist(), target=focus.tolist())
        except Exception:
            pass

    def apply_replay_frame(self, pursuer_states, target_xyz, *, env_index: int = 0) -> None:
        """Pose-only frame for trajectory replay (no Lee / PhysX control step)."""
        bridge = self.bridges[env_index].env
        states = np.asarray(pursuer_states, dtype=np.float64).reshape(self.cfg.n_uavs, 13)
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        bridge.quadrotor_states[:] = states
        bridge.positions[:] = states[:, :2]
        bridge.altitudes[:] = states[:, 2]
        bridge.dynamic_targets[0] = target[:2]
        bridge.target_altitudes[0] = float(target[2])
        self._teleport_pursuers_from_bridge([env_index])
        self._write_target_pose(env_index)
        try:
            if hasattr(self.sim_env, "sim") and hasattr(self.sim_env.sim, "forward"):
                self.sim_env.sim.forward()
        except Exception:
            pass
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_stopped() or not timeline.is_playing():
                timeline.play()
        except Exception:
            pass

    def _bridge_capture_gap(self, env_index: int = 0) -> float:
        """Kinematic 2nd-nearest gap from Radar5 bridge (PhysX-independent HUD)."""
        env = self.bridges[env_index].env
        state = np.asarray(env.quadrotor_states, dtype=float)
        target = self._target_position(env_index)
        active = ~np.asarray(env.disabled_uavs, dtype=bool)
        required = int(self.cfg.capture_required_uavs)
        if np.count_nonzero(active) < required or not np.isfinite(state).all():
            return float(self.cfg.grid_size)
        distances = np.linalg.norm(state[:, :3] - target[None, :], axis=1)
        required_distance = float(np.partition(distances[active], required - 1)[required - 1])
        return float(max(required_distance - self.cfg.capture_radius, 0.0))

    def _physical_state_batch(self, *, include_evader: bool = False):
        """Radar5 pursuer states ``(N, n_uavs, 13)``; optional full PhysX fleet.

        Uses ``check_nan=False`` so one bad clone cannot abort the whole batch;
        callers recover via :meth:`_corrupt_env_mask`. Non-finite OmniDrones
        rows are mapped to NaN Radar5 rows instead of raising.
        """
        from experiments.radar5_frozen_transfer_v1.state_adapter import omnidrones_state_to_radar5

        n_phys = int(self._n_physics_drones)
        state = self.sim_env.drone.get_state(check_nan=False)
        if isinstance(state, torch.Tensor):
            raw = state.detach()
            if torch.isfinite(raw).all():
                converted = omnidrones_state_to_radar5(raw)
                full = np.asarray(converted, dtype=np.float64).reshape(
                    self.num_envs, n_phys, 13
                )
                self._last_full_physics = full
                return full if include_evader else full[:, : int(self.cfg.n_uavs)]
            raw_np = raw.detach().cpu().numpy()
        else:
            raw_np = np.asarray(state)
        raw_np = raw_np.reshape(self.num_envs, n_phys, -1)
        full = np.full((self.num_envs, n_phys, 13), np.nan, dtype=np.float64)
        for env_index in range(self.num_envs):
            row = raw_np[env_index]
            if not np.isfinite(row).all():
                continue
            try:
                full[env_index] = omnidrones_state_to_radar5(row)
            except ValueError:
                continue
        self._last_full_physics = full
        return full if include_evader else full[:, : int(self.cfg.n_uavs)]

    def _evader_state_batch(self):
        if not self._physx_closed_loop:
            return np.full((self.num_envs, 13), np.nan, dtype=np.float64)
        full = self._physical_state_batch(include_evader=True)
        return full[:, int(self._evader_drone_index)]

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
        if self._physx_closed_loop:
            evader = self._evader_state_batch()[env_index]
            if np.isfinite(evader).all():
                return np.asarray(evader[:3], dtype=np.float64)
        env = self.bridges[env_index].env
        return np.asarray([*env.dynamic_targets[0], env.target_altitudes[0]], dtype=np.float64)

    def _write_target_pose(self, env_index: int | None = None):
        """HUD sphere only. In physx_closed_loop this mirrors PhysX — never drives it."""
        from pxr import Gf

        if not getattr(self, "_target_translates", None):
            return
        indices = range(self.num_envs) if env_index is None else (env_index,)
        for index in indices:
            self._target_translates[index].Set(Gf.Vec3d(*self._target_position(index).tolist()))

    def _sync_targets_from_physics(self, full_batch=None):
        if not self._physx_closed_loop:
            return
        if full_batch is None:
            full_batch = self._physical_state_batch(include_evader=True)
        idx = int(self._evader_drone_index)
        for env_index, bridge in enumerate(self.bridges):
            row = full_batch[env_index, idx]
            if not np.isfinite(row).all():
                continue
            bridge.sync_target_from_physics(row[:3], row[3:6])

    def _sync_world_lidar_poses(self, batch=None) -> None:
        """Move world-fixed LiDAR Xforms onto current pursuer poses (env frame)."""
        ops = getattr(self, "_lidar_pose_ops", None)
        if not ops:
            return
        from pxr import Gf

        if batch is None:
            batch = self._physical_state_batch()
        for env_index in range(self.num_envs):
            physical = batch[env_index]
            if not np.isfinite(physical).all():
                continue
            for drone_index in range(self.cfg.n_uavs):
                slot = env_index * self.cfg.n_uavs + drone_index
                if slot >= len(ops) or ops[slot] is None:
                    continue
                translate, orient = ops[slot]
                pos = physical[drone_index, :3]
                quat = physical[drone_index, 6:10]  # wxyz
                try:
                    translate.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                    orient.Set(
                        Gf.Quatf(
                            float(quat[0]),
                            float(quat[1]),
                            float(quat[2]),
                            float(quat[3]),
                        )
                    )
                except Exception:
                    pass

    def _read_native_lidar(self, env_index: int | None = None):
        """Read PhysX LiDAR into Radar5 bridges.

        Call **before** ``bridge.advance()`` / ``observe`` so the decision sees
        the current scan. ``env_index=None`` updates every parallel env.
        """
        if not self._native_lidar_enabled:
            self._last_native_scan = None
            for bridge in self.bridges:
                bridge.clear_native_sensor_scan()
            return
        if not getattr(self, "_lidar_warmed", False):
            for bridge in self.bridges:
                bridge.clear_native_sensor_scan()
            return
        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import quaternion_to_rotation

        indices = range(self.num_envs) if env_index is None else (int(env_index),)
        batch = self._physical_state_batch()
        self._sync_world_lidar_poses(batch)
        last = None
        for index in indices:
            physical = batch[index]
            if not np.isfinite(physical).all():
                self.bridges[index].clear_native_sensor_scan()
                continue
            positions = physical[:, :3]
            rotations = np.stack([quaternion_to_rotation(row[6:10]) for row in physical])
            start = index * self.cfg.n_uavs
            paths = self.sensor_paths[start : start + self.cfg.n_uavs]
            # Prefer semantic sphere (HUD mirrors PhysX target); also accept
            # hummi/evader prim names if the sphere is absent.
            tokens = ("radar5_target", "Hummingbird", "hummingbird", "evader")
            try:
                scan = self.lidars[index].read_native_scan(
                    sim_time=float(self._steps[index]) * self.cfg.decision_dt,
                    lidar_interface=self.lidar_interface,
                    sensor_paths=paths,
                    sensor_origins=positions,
                    sensor_rotations=rotations,
                    target_prim_tokens=tokens,
                )
            except (RuntimeError, ValueError) as exc:
                self.bridges[index].clear_native_sensor_scan()
                if index == 0:
                    print(f"[isaac] native lidar env={index}: {exc}", flush=True)
                continue
            last = scan
            # Prefer geometric fallback when reads fail, or when the PhysX
            # lidar returns no semantic target hits. Empty-but-"available"
            # native masks previously blinded FRPN/RL (0 detections forever).
            usable = bool(scan.get("sensor_available", False))
            any_hit = bool(np.any(scan.get("visible"))) if scan.get("visible") is not None else False
            if usable and not any_hit:
                usable = False
            if usable:
                self.bridges[index].set_native_sensor_scan(scan)
            else:
                self.bridges[index].clear_native_sensor_scan()
        self._last_native_scan = last

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
        # Articulation root writes require a live physics view; if the timeline
        # is stopped Isaac falls back to USD setGlobalPose (illegal under Direct GPU).
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_stopped() or not timeline.is_playing():
                timeline.play()
        except Exception:
            pass
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        env_ids = torch.as_tensor(env_indices, device=self.sim_env.device, dtype=torch.long)
        offsets = self._env_offsets()[env_ids].unsqueeze(1)  # (K, 1, 3)
        n_phys = int(self._n_physics_drones)
        locals_pos = []
        locals_quat = []
        for index in env_indices:
            physical = np.asarray(self.bridges[index].env.quadrotor_states, dtype=np.float64)
            pos = physical[:, :3].copy()
            quat = physical[:, 6:10].copy()
            if n_phys > self.cfg.n_uavs:
                # Pad evader from Radar5 target truth (recovery / replay only).
                env = self.bridges[index].env
                evader_pos = np.array(
                    [*env.dynamic_targets[0], env.target_altitudes[0]], dtype=np.float64
                )
                evader_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                pos = np.vstack([pos, evader_pos[None, :]])
                quat = np.vstack([quat, evader_quat[None, :]])
            locals_pos.append(pos)
            locals_quat.append(quat)
        positions = (
            torch.as_tensor(np.stack(locals_pos), dtype=torch.float32, device=self.sim_env.device)
            + offsets
        )
        orientations = torch.as_tensor(
            np.stack(locals_quat), dtype=torch.float32, device=self.sim_env.device
        )
        self.sim_env.drone.set_world_poses(positions, orientations, env_ids)
        self.sim_env.drone.set_velocities(
            torch.zeros((len(env_indices), n_phys, 6), device=self.sim_env.device),
            env_ids,
        )
        self._clear_drone_dynamics(env_ids)

    def _resync_pursuer_visuals(self, batch_states=None) -> None:
        """Push current PhysX pursuer poses so the Kit Viewport shows Hummingbirds."""
        if batch_states is None:
            batch_states = self._physical_state_batch()
        if not np.isfinite(batch_states).all():
            return
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_stopped() or not timeline.is_playing():
                timeline.play()
        except Exception:
            pass
        env_ids = torch.arange(self.num_envs, device=self.sim_env.device, dtype=torch.long)
        offsets = self._env_offsets()[env_ids].unsqueeze(1)
        positions = (
            torch.as_tensor(batch_states[:, :, :3], dtype=torch.float32, device=self.sim_env.device)
            + offsets
        )
        orientations = torch.as_tensor(
            batch_states[:, :, 6:10], dtype=torch.float32, device=self.sim_env.device
        )
        try:
            self.sim_env.drone.set_world_poses(positions, orientations, env_ids)
        except Exception as exc:
            print(f"[isaac] visual resync skipped: {exc}", flush=True)

    def _recover_corrupt_envs(self, env_indices) -> None:
        """Re-seat Hummingbirds from Radar5 bridge state after a PhysX NaN blow-up."""
        if not env_indices:
            return
        print(f"[isaac] recovering corrupt envs={list(env_indices)}", flush=True)
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_stopped() or not timeline.is_playing():
                timeline.play()
        except Exception:
            pass
        # Prefer bridge kinematic poses (finite) over NaN PhysX rows.
        self._teleport_pursuers_from_bridge(list(env_indices))
        try:
            if hasattr(self.sim_env, "sim") and hasattr(self.sim_env.sim, "forward"):
                self.sim_env.sim.forward()
        except Exception:
            pass
        for env_index in env_indices:
            self._write_target_pose(env_index)

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
        try:
            self._read_native_lidar()
        except (RuntimeError, ValueError):
            self._last_native_scan = None
            for bridge in self.bridges:
                bridge.clear_native_sensor_scan()
        # Warm KF tracks from the post-reset geometric/native scan. Without this,
        # FRPN/MPC station-keep on step 0 (track.initialized=False → act_norm=0).
        # Use belief update only — bridge.advance() would kinematically nudge
        # Radar5 pursuer state and desync it from PhysX.
        for env_index in env_indices:
            try:
                if not self._native_lidar_enabled or self.bridges[env_index].env._external_sensor_scan is None:
                    self.bridges[env_index].clear_native_sensor_scan()
                self.bridges[env_index].env._update_target_beliefs(force=True)
            except Exception as exc:
                print(f"[isaac] track warmup env={env_index} soft-fail: {exc}", flush=True)
        if hasattr(self, "_physical_potential"):
            for env_index in env_indices:
                try:
                    self._physical_potential[env_index] = self._gap_potential(
                        self.minimum_capture_gap(env_index)
                    )
                except Exception:
                    self._physical_potential[env_index] = 0.0

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
        self._evader_alt_hold = float(self.cfg.initial_altitude)
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

    def _clamp_fleet_speeds(self, velocity_cmds=None) -> None:
        """Clamp PhysX lin/ang velocities to configured flight envelopes.

        For pure closed-loop the 4th Lee body previously ran away at
        ``target_speed * 2.5`` with unbounded climb. When ``velocity_cmds`` is
        provided we also inject the commanded evader linvel (governor) so the
        articulation stays a PhysX body without relying on Lee vertical track.
        """
        try:
            vel = self.sim_env.drone.get_velocities(True)
        except Exception:
            return
        if not torch.is_tensor(vel):
            return
        n_phys = int(self._n_physics_drones)
        if vel.ndim == 2:
            vel = vel.reshape(self.num_envs, n_phys, -1)
        if vel.shape[-1] < 3:
            return
        max_p = float(self.cfg.max_horizontal_velocity) * 2.5
        # Evader command is derated to 0.75 * target_speed in step_joint; allow
        # only a small overshoot for Lee/injection noise.
        max_e = float(self.cfg.target_speed) * 0.75 * 1.15
        max_ang = 8.0
        max_ev_z = float(getattr(self.cfg, "target_vertical_speed", 0.8) or 0.8)
        lin = vel[..., :3]
        speed = torch.linalg.norm(lin, dim=-1, keepdim=True).clamp_min(1e-6)
        limits = torch.full(
            (self.num_envs, n_phys, 1),
            max_p,
            device=vel.device,
            dtype=vel.dtype,
        )
        eidx = int(self._evader_drone_index) if self._evader_drone_index is not None else None
        if self._physx_closed_loop and eidx is not None:
            limits[:, eidx, :] = max_e
        scale = torch.clamp(limits / speed, max=1.0)
        vel = vel.clone()
        vel[..., :3] = lin * scale
        if self._physx_closed_loop and eidx is not None:
            # Hard-limit vertical channel independently of horizontal speed budget.
            vel[:, eidx, 2] = torch.clamp(vel[:, eidx, 2], -max_ev_z, max_ev_z)
            if velocity_cmds is not None:
                cmd = torch.as_tensor(
                    velocity_cmds, dtype=vel.dtype, device=vel.device
                )
                if cmd.ndim == 2:
                    cmd = cmd.reshape(self.num_envs, n_phys, 3)
                # Inject commanded linvel on the evader — Lee motors still run,
                # but tracked speed matches the Radar5 occlusion policy.
                vel[:, eidx, :3] = cmd[:, eidx, :3]
                vel[:, eidx, 2] = torch.clamp(vel[:, eidx, 2], -max_ev_z, max_ev_z)
                h = torch.linalg.norm(vel[:, eidx, :2], dim=-1, keepdim=True).clamp_min(1e-6)
                vel[:, eidx, :2] = vel[:, eidx, :2] * torch.clamp(
                    (max_e * 0.999) / h, max=1.0
                )
        if vel.shape[-1] >= 6:
            ang = vel[..., 3:6]
            ang_spd = torch.linalg.norm(ang, dim=-1, keepdim=True).clamp_min(1e-6)
            vel[..., 3:6] = ang * torch.clamp(max_ang / ang_spd, max=1.0)
            if self._physx_closed_loop and eidx is not None:
                vel[:, eidx, 3:6] = 0.0
        env_ids = torch.arange(self.num_envs, device=self.sim_env.device, dtype=torch.long)
        try:
            self.sim_env.drone.set_velocities(vel, env_ids)
        except Exception:
            try:
                self.sim_env.drone.set_velocities(vel.reshape(self.num_envs * n_phys, -1))
            except Exception:
                pass
        # Contain altitude: Lee vertical channel still drifts the evader to z≫max.
        if not self._physx_closed_loop:
            return
        try:
            pos_w, ori_w = self.sim_env.drone.get_world_poses(True)
        except Exception:
            return
        if pos_w.ndim == 2:
            pos_w = pos_w.reshape(self.num_envs, n_phys, 3)
            ori_w = ori_w.reshape(self.num_envs, n_phys, 4)
        offsets = self._env_offsets().reshape(self.num_envs, 1, 3)
        local = pos_w - offsets
        z_min = float(self.cfg.min_altitude)
        z_max = float(self.cfg.max_altitude)
        z = local[..., 2:3]
        out = (z < z_min) | (z > z_max)
        if eidx is not None:
            # Always snap the closed-loop evader into the band (even if pursuers
            # are fine) — soft any() over the whole fleet previously let a
            # single in-band pursuer hide an out-of-band evader when shapes
            # mismatched, and GPU pose writes are cheap for N=1.
            e_z = local[:, eidx, 2]
            e_out = (e_z < z_min) | (e_z > z_max)
            if bool(e_out.any().item()) or bool(out.any().item()):
                local = local.clone()
                local[..., 2:3] = torch.clamp(z, z_min, z_max)
                # Kill residual climb/descent into the wall.
                try:
                    vel2 = self.sim_env.drone.get_velocities(True)
                    if vel2.ndim == 2:
                        vel2 = vel2.reshape(self.num_envs, n_phys, -1)
                    vel2 = vel2.clone()
                    hit_low = local[:, eidx, 2] <= z_min + 1e-3
                    hit_high = local[:, eidx, 2] >= z_max - 1e-3
                    if bool(hit_low.any().item()):
                        vel2[hit_low, eidx, 2] = torch.clamp(vel2[hit_low, eidx, 2], min=0.0)
                    if bool(hit_high.any().item()):
                        vel2[hit_high, eidx, 2] = torch.clamp(vel2[hit_high, eidx, 2], max=0.0)
                    self.sim_env.drone.set_velocities(vel2, env_ids)
                except Exception:
                    pass
                try:
                    self.sim_env.drone.set_world_poses(local + offsets, ori_w, env_ids)
                except Exception:
                    pass
        elif bool(out.any().item()):
            local = local.clone()
            local[..., 2:3] = torch.clamp(z, z_min, z_max)
            try:
                self.sim_env.drone.set_world_poses(local + offsets, ori_w, env_ids)
            except Exception:
                pass

    def _integrate_kinematic_evader(self, velocity_cmds: np.ndarray) -> None:
        """Advance only the PhysX evader prim by exact velocity integration.

        Pursuer articulations are left untouched (Lee/PhysX authority). Rewriting
        the whole fleet each step previously corrupted pursuer tracking.
        """
        if not self._physx_closed_loop or not self._kinematic_evader:
            return
        cmds = np.asarray(velocity_cmds, dtype=np.float64).reshape(self.num_envs, 3)
        dt = float(self.cfg.decision_dt)
        idx = int(self._evader_drone_index)
        n_phys = int(self._n_physics_drones)
        env_ids = torch.arange(self.num_envs, device=self.sim_env.device, dtype=torch.long)
        offsets = self._env_offsets()[env_ids]  # (N, 3)
        # Read the live fleet pose, then overwrite only the evader slot.
        # OmniDrones get_world_poses(clone: bool) — never pass env_ids here
        # (that was interpreted as clone=Tensor and crashed when num_envs>1).
        pos_w, ori_w = self.sim_env.drone.get_world_poses(True)
        if pos_w.ndim == 2:
            pos_w = pos_w.reshape(self.num_envs, n_phys, 3)
            ori_w = ori_w.reshape(self.num_envs, n_phys, 4)
        try:
            vel_w = self.sim_env.drone.get_velocities(True)
            if vel_w.ndim == 2:
                vel_w = vel_w.reshape(self.num_envs, n_phys, -1)
        except Exception:
            vel_w = torch.zeros(
                (self.num_envs, n_phys, 6), device=self.sim_env.device, dtype=torch.float32
            )

        full = self._physical_state_batch(include_evader=True)
        for env_index in range(self.num_envs):
            cur = full[env_index, idx, :3]
            if not np.isfinite(cur).all():
                bridge = self.bridges[env_index].env
                cur = np.array(
                    [*bridge.dynamic_targets[0], bridge.target_altitudes[0]],
                    dtype=np.float64,
                )
            nxt = cur + cmds[env_index] * dt
            nxt[0] = float(np.clip(nxt[0], 0.5, float(self.cfg.grid_size) - 0.5))
            nxt[1] = float(np.clip(nxt[1], 0.5, float(self.cfg.grid_size) - 0.5))
            nxt[2] = float(
                np.clip(nxt[2], float(self.cfg.min_altitude), float(self.cfg.max_altitude))
            )
            pos_w[env_index, idx] = (
                torch.as_tensor(nxt, dtype=torch.float32, device=self.sim_env.device)
                + offsets[env_index]
            )
            ori_w[env_index, idx] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.sim_env.device
            )
            if vel_w.shape[-1] >= 3:
                vel_w[env_index, idx, :3] = torch.as_tensor(
                    cmds[env_index], dtype=torch.float32, device=self.sim_env.device
                )
                if vel_w.shape[-1] >= 6:
                    vel_w[env_index, idx, 3:6] = 0.0

        self.sim_env.drone.set_world_poses(pos_w, ori_w, env_ids)
        self.sim_env.drone.set_velocities(vel_w, env_ids)
        try:
            if hasattr(self.sim_env, "sim") and hasattr(self.sim_env.sim, "forward"):
                self.sim_env.sim.forward()
        except Exception:
            pass

    def _step_physics_batch(self, velocity_batch, yaw_batch):
        """Advance all envs one decision step with batched Lee velocity tracking.

        ``velocity_batch`` / ``yaw_batch`` are sized to ``_n_physics_drones``
        (3 hybrid, 4 closed-loop with last slot = evader).
        """
        n_phys = int(self._n_physics_drones)
        velocity_batch = np.asarray(velocity_batch, dtype=np.float32)
        yaw_batch = np.asarray(yaw_batch, dtype=np.float32)
        if velocity_batch.shape != (self.num_envs, n_phys, 3):
            raise ValueError(
                f"velocity_batch must be {(self.num_envs, n_phys, 3)}; got {velocity_batch.shape}"
            )
        if yaw_batch.shape != (self.num_envs, n_phys):
            raise ValueError(
                f"yaw_batch must be {(self.num_envs, n_phys)}; got {yaw_batch.shape}"
            )
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
        # Altitude hold: current-z position setpoints give Lee zero vertical
        # position error, so a bad vz channel drifts the fleet to the floor
        # (pursuers) or ceiling (evader). Pin every body to the hold altitude.
        hold_z = float(
            getattr(self, "_evader_alt_hold", None) or self.cfg.initial_altitude
        )
        eidx = (
            int(self._evader_drone_index)
            if self._physx_closed_loop and self._evader_drone_index is not None
            else None
        )
        for _ in range(physics_steps):
            state = self.sim_env.drone.get_state(check_nan=False)
            # Hold current xyz as position setpoints so Lee tracks velocity only.
            # Replace non-finite entries so controller.compute cannot explode.
            if not torch.isfinite(state).all():
                state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            target_pos = state[..., :3].clone()
            target_pos[..., 2] = hold_z
            # Soft-dampen commanded vertical rates so FRPN/MPC los terms cannot
            # dive the fleet into the ground plane.
            vel_cmd = velocity_tensor
            if self._physx_closed_loop:
                max_vz_p = float(getattr(self.cfg, "max_vertical_velocity", None) or 0.8)
                max_vz_e = float(getattr(self.cfg, "target_vertical_speed", 0.8) or 0.8)
                vel_cmd = velocity_tensor.clone()
                vel_cmd[..., 2] = torch.clamp(vel_cmd[..., 2], -max_vz_p, max_vz_p)
                if eidx is not None:
                    vel_cmd[:, eidx, 2] = torch.clamp(
                        vel_cmd[:, eidx, 2], -max_vz_e, max_vz_e
                    )
            with torch.no_grad():
                motor_action = self.sim_env.controller.compute(
                    state[..., :13],
                    target_pos=target_pos,
                    target_vel=vel_cmd,
                    target_yaw=yaw_tensor,
                ).clamp(-1.0, 1.0)
                if not torch.isfinite(motor_action).all():
                    motor_action = torch.nan_to_num(
                        motor_action, nan=0.0, posinf=0.0, neginf=0.0
                    ).clamp(-1.0, 1.0)
            self.td = self.sim_env.step(self.td.set(("agents", "action"), motor_action))
            # Hard speed governor: Lee/PhysX occasionally runaway on the 4th
            # body; clamp + inject commanded evader linvel every physics substep.
            self._clamp_fleet_speeds(velocity_cmds=vel_cmd if self._physx_closed_loop else velocity_tensor)

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
        physx_broken = bool(physics_failed or not np.isfinite(physical).all())
        # Viewport play: keep the episode alive on PhysX blow-ups. Report the
        # Radar5 kinematic gap (not the grid_size=25 sentinel) so HUD/logs stay
        # meaningful while articulations are reseated.
        if physx_broken and not self._headless:
            result = dict(result)
            bridge_gap = self._bridge_capture_gap(env_index)
            result["reward"] = float(result.get("reward", 0.0))
            result["reward_components"] = dict(result.get("reward_components", {}))
            result["reward_components"]["isaac_physics_recovered"] = 1.0
            result["capture_success"] = False
            result["minimum_capture_gap"] = bridge_gap
            result["safety_reasons"] = safety_reasons
            result["controller_feasible_rate"] = 0.0
            result["execution_mode"] = "isaac_lee_velocity"
            result["sensor_mode"] = (
                "isaac_lidar" if self._native_lidar_enabled else "geometric_lidar_fallback"
            )
            bridge_state = np.asarray(
                self.bridges[env_index].env.quadrotor_states, dtype=float
            )
            result["quadrotor_states"] = np.nan_to_num(bridge_state, nan=0.0).tolist()
            result["terminated"] = False
            result["truncated"] = False
            result["physics_failed"] = False
            self._safety_reasons[env_index] = safety_reasons
            self._episode_return[env_index] += float(result["reward"])
            return result
        if physx_broken:
            # Headless: emit a safe truncated transition; caller soft-resets.
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
        if self._physx_closed_loop:
            self._sync_targets_from_physics()
        n_phys = int(self._n_physics_drones)
        velocity_batch = np.zeros((self.num_envs, n_phys, 3), dtype=np.float32)
        yaw_batch = np.zeros((self.num_envs, n_phys), dtype=np.float32)
        evader_cmd = np.zeros((self.num_envs, 3), dtype=np.float32)
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
            velocity_batch[env_index, : self.cfg.n_uavs] = requested_velocity.astype(np.float32)
            yaw_batch[env_index, : self.cfg.n_uavs] = np.asarray(target_yaw, dtype=np.float32)
            if self._physx_closed_loop:
                evader_vel = bridge.env.desired_evader_world_velocity(0).astype(np.float32)
                # Clamp to configured target speeds, then derate horizontal speed so
                # pursuer Lee lag can still close for dual-capture. Without this,
                # FRPN/MPC on PhysX closed stayed ~0–8% while hybrid (kinematic
                # target) reached ~75% at the same cfg.target_speed.
                horiz = float(np.linalg.norm(evader_vel[:2]))
                max_h = float(self.cfg.target_speed)
                if horiz > max_h > 0:
                    evader_vel[:2] *= max_h / horiz
                # 0.75 leaves pursuers (~1.48 m/s) a clear closing margin after
                # Lee tracking loss on the 4-body Formation controller.
                evader_vel[:2] *= 0.75
                max_vz = float(getattr(self.cfg, "target_vertical_speed", 0.8))
                evader_vel[2] = float(np.clip(evader_vel[2], -max_vz, max_vz))
                if self._kinematic_evader:
                    # Hover under Lee; pose is advanced after the physics step.
                    velocity_batch[env_index, self._evader_drone_index] = 0.0
                    evader_cmd[env_index] = evader_vel
                else:
                    velocity_batch[env_index, self._evader_drone_index] = evader_vel
                # Hold current yaw for the evader (policy does not command yaw).
                evader_state = self._evader_state_batch()[env_index]
                if np.isfinite(evader_state).all():
                    yaw_batch[env_index, self._evader_drone_index] = float(
                        yaw_from_quaternion(evader_state[6:10])
                    )
            safety_all.append(safety_reasons)

        self._step_physics_batch(velocity_batch, yaw_batch)
        if self._physx_closed_loop and self._kinematic_evader:
            self._integrate_kinematic_evader(evader_cmd)
        full_batch = self._physical_state_batch(include_evader=self._physx_closed_loop)
        batch_states = full_batch[:, : self.cfg.n_uavs] if self._physx_closed_loop else full_batch
        corrupt = self._corrupt_env_mask(batch_states)
        if self._physx_closed_loop:
            # Evader blow-ups also count as corrupt for formal headless runs.
            evader = full_batch[:, self._evader_drone_index]
            corrupt = corrupt | ~np.isfinite(evader).reshape(self.num_envs, -1).all(axis=1)
            # Governor keeps |v| ≤ 1.15 * target_speed; anything above 1.5× is
            # a real articulation blow-up worth recovering.
            max_e = float(self.cfg.target_speed) * 1.5
            evader_spd = np.linalg.norm(evader[:, 3:6], axis=-1)
            corrupt = corrupt | (evader_spd > max_e)
        if np.any(corrupt):
            self._physics_fail_count += int(np.count_nonzero(corrupt))
            print(
                f"[isaac] physics corrupt envs={np.flatnonzero(corrupt).tolist()} "
                f"total_fails={self._physics_fail_count}",
                flush=True,
            )
            # Recover articulations from Radar5 bridge state and continue when
            # possible. Truncating every Lee glitch made closed-loop FRPN/MPC
            # collect collapse to ~0% even after kinematic-evader stabilization.
            recovered = [i for i in range(self.num_envs) if corrupt[i]]
            self._recover_corrupt_envs(recovered)
            if self._physx_closed_loop and self._kinematic_evader:
                # Re-seat the evader after pursuer recovery using the last command.
                self._integrate_kinematic_evader(evader_cmd)
            full_batch = self._physical_state_batch(include_evader=self._physx_closed_loop)
            batch_states = (
                full_batch[:, : self.cfg.n_uavs] if self._physx_closed_loop else full_batch
            )
            corrupt = self._corrupt_env_mask(batch_states)
            if self._physx_closed_loop:
                evader = full_batch[:, self._evader_drone_index]
                corrupt = corrupt | ~np.isfinite(evader).reshape(self.num_envs, -1).all(axis=1)
        # Do not re-push world poses every step: under Direct-GPU PhysX that
        # path can re-corrupt articulations and spam recovery. Recover-only.
        self._sync_bridge(batch=batch_states)
        if self._physx_closed_loop:
            self._sync_targets_from_physics(full_batch)
        # HUD sphere tracks PhysX target so semantic LiDAR can hit radar5_target.
        self._write_target_pose()
        # Native LiDAR must be installed before advance/observe.
        try:
            self._read_native_lidar()
        except Exception as exc:
            print(f"[isaac] native lidar batch failed: {exc}", flush=True)
            for bridge in self.bridges:
                bridge.clear_native_sensor_scan()

        results = []
        done_indices = []
        for env_index, bridge in enumerate(self.bridges):
            physics_failed = bool(corrupt[env_index])
            # Headless: truncate on corrupt PhysX. Hybrid GUI may recover;
            # physx_closed never treats kinematic fallback as formal success.
            if physics_failed and (self._headless or self._physx_closed_loop):
                result = {
                    "reward": 0.0,
                    "reward_components": {},
                    "capture_success": False,
                    "terminated": False,
                    "truncated": True,
                    "physics_recovered": False,
                    "obs": bridge.observe(),
                }
            else:
                if physics_failed and not self._headless:
                    # Reseat again from last finite bridge poses before advancing.
                    self._recover_corrupt_envs([env_index])
                    result_flag_recovered = True
                else:
                    result_flag_recovered = False
                result = bridge.advance()
                result["physics_recovered"] = bool(result_flag_recovered)
                # Pursuer states were synced before advance; do not rebuild the
                # expensive hetero-graph observation a second time.
                if result.get("obs") is None:
                    if not physics_failed:
                        bridge.sync_pursuers(batch_states[env_index])
                    result["obs"] = bridge.observe()
            self._steps[env_index] += 1
            fatal_physics = physics_failed and (self._headless or self._physx_closed_loop)
            result = self._finalize_env_result(
                env_index, result, safety_all[env_index], physics_failed=fatal_physics
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