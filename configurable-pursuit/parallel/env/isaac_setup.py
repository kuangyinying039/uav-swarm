"""Isaac 启动与并行场景搭建；不采样任务场景，不执行每步控制。

调用前由运行入口启动 Isaac app。返回的物理对象由 od_env 持有，
全局 app/SimulationContext 的最终关闭仍由运行入口负责。
"""
import math
import torch
from parallel.env.visualization import ObstacleVisualization
from parallel.env.config_loader import resolved_env_spacing


class IsaacSetup:
    """一次性初始化结果；构造结束后把资源交给环境。"""
    def __init__(self, cfg, scene, background, *, simulation, visualization):
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.utils import prims as prim_utils
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.robots.robot import RobotBase
        from omni_drones.utils.scene import design_scene
        import omni.usd
        import omni.physics.tensors as tensors
        from pxr import UsdPhysics, PhysxSchema, Gf
        device = str(cfg.sim.device)
        n = scene.initial_pos.shape[0]
        self.sim = SimulationContext(stage_units_in_meters=1., physics_dt=simulation['dt'], rendering_dt=simulation['dt'],
                                     sim_params=cfg.sim, backend='torch', device=device)
        if not math.isclose(float(self.sim.get_physics_dt()),simulation['dt'],rel_tol=1e-7,abs_tol=1e-10):
            raise RuntimeError(f'Isaac physics dt={self.sim.get_physics_dt()} differs from configured simulation.dt={simulation["dt"]}')
        self.stage = stage = omni.usd.get_context().get_stage()
        cloner = GridCloner(spacing=resolved_env_spacing(simulation,scene.config['bounds']))
        cloner.define_base_env('/World/envs')
        root = '/World/envs/env_0'
        prim_utils.define_prim(root)
        self.drone, self.controller = MultirotorBase.make(simulation['drone_model'], simulation['controller'], device)
        from parallel.env.downwash import finite_downwash
        self.drone.downwash = finite_downwash
        self.drone.spawn(translations=scene.initial_pos[0],prim_paths=[root+'/'+scene.layout.aircraft_prim_name(j) for j in range(scene.layout.num_aircraft)])
        design_scene()
        # Filter every cross-aircraft link pair, preserving joints and ground.
        bodies = [p for p in stage.Traverse() if 'Hummingbird_' in str(p.GetPath()) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        # Preserve the previous view preparation sleep threshold, but author it
        # before physics exists. Cloning propagates this to every environment.
        for body in bodies:
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateSleepThresholdAttr().Set(0.)
            if simulation['aircraft_collisions']:
                continue
            owner = str(body.GetPath()).split('/Hummingbird_')[1].split('/')[0]
            others = [p.GetPath() for p in bodies if str(p.GetPath()).split('/Hummingbird_')[1].split('/')[0] != owner]
            UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(others)
        paths = cloner.generate_paths('/World/envs/env', n)
        offsets = cloner.clone(source_prim_path=root, prim_paths=paths, replicate_physics=False, copy_from_source=True)
        self.offsets = torch.as_tensor(offsets, device=device, dtype=torch.float32)
        RobotBase._envs_positions = self.offsets[:, None]
        for i, path in enumerate(paths):
            for j in range(scene.layout.num_aircraft):
                prim = stage.GetPrimAtPath(path+'/'+scene.layout.aircraft_prim_name(j))
                prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*scene.initial_pos[i, j].cpu().tolist()))
                yaw = float(scene.initial_yaw[i, j])
                attr = prim.GetAttribute('xformOp:orient')
                q = Gf.Quatd(math.cos(yaw/2), Gf.Vec3d(0., 0., math.sin(yaw/2))) if str(attr.GetTypeName()) == 'quatd' else Gf.Quatf(math.cos(yaw/2), Gf.Vec3f(0., 0., math.sin(yaw/2)))
                attr.Set(q)
        cloner.filter_collisions(self.sim.get_physics_context().prim_path, '/World/collisions', paths,
                                 global_paths=['/World/defaultGroundPlane'])
        self.visualizer = ObstacleVisualization(stage, paths,
            scene.static.shape[1]+background.shape[1], static_count=scene.static.shape[1], config=visualization)
        self.visualizer.update(scene.static, background, scene.radii[:,:background.shape[1]])
        self.sim.reset()
        self.gpu_physics = bool(PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(self.sim.get_physics_context().prim_path)).GetEnableGPUDynamicsAttr().Get())
        assert self.gpu_physics
        # Isaac 4.1 RigidPrimView defaults to preparing contact reporters,
        # which authors PhysxRigidBodyAPI/sleepThreshold after physics starts.
        # The USD change is consumed on the first rendered physics step and
        # attempts forbidden CPU setGlobalPose under direct GPU transport.
        # Circle tasks use no contact reports. Scope this compatibility adapter
        # to initialization, preserving all tensor pose/joint resets and forces.
        import omni_drones.robots.drone.multirotor as multirotor_module
        rigid_view = multirotor_module.RigidPrimView
        def contact_free_view(*args, **kwargs):
            kwargs['prepare_contact_sensors'] = False
            return rigid_view(*args, **kwargs)
        multirotor_module.RigidPrimView = contact_free_view
        try:
            self.drone.initialize(track_contact_forces=False)
        finally:
            multirotor_module.RigidPrimView = rigid_view
        observed = [int(path.split('/env_')[1].split('/')[0]) for path in self.drone.prim_paths]
        assert observed == [i for i in range(n) for _ in range(scene.layout.num_aircraft)]
        expected=[path+'/'+scene.layout.aircraft_prim_name(j) for path in paths for j in range(scene.layout.num_aircraft)]
        if list(self.drone.prim_paths)!=expected: raise RuntimeError('Physics aircraft identity order differs from EntityLayout')
        self.tensor_sim = tensors.create_simulation_view('torch')
        self.tensor_sim.set_subspace_roots('/')
        self.joints = self.drone._view.get_joint_positions().clone()
