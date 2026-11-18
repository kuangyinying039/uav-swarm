"""组织场景重置、障碍运动、观测和任务/奖励结算；不包含采样实现。"""
import copy
import time
import numpy as np
import torch

from parallel.env.scene import Scene
from parallel.env.task import evaluate_task
from parallel.env.dynamic_obstacles import DynamicObstacleMotion
from parallel.env.config_loader import load_settings
from parallel.env.reward import EvaderReward
from parallel.env.observation import build_observation


def norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


class BatchRules:
    def __init__(self, seeds, device='cuda:0', dtype=torch.float32, sampler_backend='cpu', flight_mode=None, reset_motion=None, environment_config=None, pursuer_config=None, formation=None):
        self.settings = load_settings(environment_config)
        if flight_mode is not None:
            self.settings['simulation']['flight_mode']=flight_mode
            self.settings=load_settings(self.settings)
        flight_mode=self.settings['simulation']['flight_mode']
        self.flight_mode = flight_mode
        self.reset_motion = reset_motion or ('hover' if flight_mode == 'planar' else 'reference')
        if self.reset_motion not in ('hover','reference'): raise ValueError('unknown reset motion')
        seeds = list(seeds)
        if sampler_backend != 'cpu': raise ValueError('Native scene currently supports sampler_backend=cpu only')
        self.sampler_backend = sampler_backend
        self.profile_reset = False
        self.last_reset_profile = {}
        self.device, self.dtype = device, dtype
        from parallel.strategies.config import load_config
        self.pursuer_config=load_config(pursuer_config)[0]
        if formation=='one_side_triangle':
            scene_config=copy.deepcopy(self.settings['scene'])
            scene_config['dynamic_obstacles']['count']=0
            self.settings['scene']=scene_config
        self.scene=Scene(seeds,self.settings['scene'],self.pursuer_config['spawn'],
            device=device,dtype=dtype,reset_motion=self.reset_motion,
            target_height=self.settings['simulation']['target_height'],formation=formation)
        self.n=len(seeds)
        self.layout=self.scene.layout
        c=self.settings['scene'];d=c['dynamic_obstacles']
        # Compatibility view for existing task/control callers; no legacy files.
        cfg=self.cfg=dict(dt=self.settings['simulation']['dt'],robot_radius=c['evader_radius'],
            boundary_half_extent_x=c['bounds']['half_extent_x'],boundary_half_extent_y=c['bounds']['half_extent_y'],
            dynamic_goal_switch_threshold=d['goal_switch_threshold'],dynamic_local_goal_half_range=d['local_goal_half_range'],
            obstacle_region_half_extent=d['region_half_extent'])
        self._bind_scene_tensors()
        self.dynamic_obstacles = DynamicObstacleMotion(self.n,self.layout.num_dynamic,device,dtype,dt=cfg['dt'],
            goal_threshold=cfg['dynamic_goal_switch_threshold'],
            local_half_range=cfg['dynamic_local_goal_half_range'],
            region_extent=cfg['obstacle_region_half_extent'])
        self.steps = torch.zeros(self.n,device=device,dtype=torch.long)
        self.reward = EvaderReward(self.n,device,dtype,config=self.settings["reward"])
        self.reset(torch.arange(self.n,device=device), resample=False)

    @property
    def initial(self): return self.scene.initial

    @initial.setter
    def initial(self,value): self.scene.initial=value

    def _bind_scene_tensors(self):
        for name in ('static','buildings','goal','frame','origin','radii','speed','initial_pos','initial_yaw','initial_velocity'):
            setattr(self,name,getattr(self.scene,name))

    def _load_initial(self,ids=None):
        self.scene.load_initial(ids)
        self._bind_scene_tensors()

    def tensor(self, data):
        return torch.as_tensor(np.asarray(data), device=self.device, dtype=self.dtype)

    def reset(self, ids, resample=True, seeds=None):
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("Reset ids must be a one-dimensional int64 tensor")
        if ((ids < 0) | (ids >= self.n)).any() or ids.unique().numel() != ids.numel():
            raise ValueError("Reset ids must be unique and within [0,N)")
        selected = ids.cpu().tolist()
        if seeds is not None and not resample:
            raise ValueError('Explicit seeds require resample=True')
        if seeds is not None and len(seeds) != len(selected):
            raise ValueError('One seed is required per selected environment')
        if not selected:
            return
        profile = {}
        def mark(key, started):
            if self.profile_reset and str(self.device).startswith('cuda'): torch.cuda.synchronize()
            profile[key] = time.perf_counter()-started
        started=time.perf_counter()
        self.scene.reset(selected,resample=resample,seeds=seeds)
        self._bind_scene_tensors()
        mark('scene_reset_seconds', started); started=time.perf_counter()
        self.dynamic_obstacles.reset(ids,[self.initial[i] for i in selected])
        mark('motion_buffers_seconds', started); started=time.perf_counter()
        self.steps[ids]=0
        self.reward.reset(ids,norm(self.goal[ids]-self.initial_pos[ids,0,:2]))
        mark('rule_buffers_seconds', started)
        self.last_reset_profile=profile

    @property
    def bg(self):
        return self.dynamic_obstacles.position

    @property
    def bgvel(self):
        return self.dynamic_obstacles.velocity

    @property
    def bg_targets(self):
        return self.dynamic_obstacles.targets

    def _sample_dynamic_obstacle_goal(self,i,j):
        return self.scene.sample_dynamic_goal(i,j)

    def prepare_dynamic_obstacles(self):
        """wrapper 在 P 采样前调用，保留背景先于 P 的 RNG 次序。"""
        self.dynamic_obstacles.prepare(self._sample_dynamic_obstacle_goal)

    def advance_dynamic_obstacles(self):
        self.dynamic_obstacles.advance(self.origin[:,self.layout.dynamic_objects],self.speed[:,self.layout.dynamic_objects],self._sample_dynamic_obstacle_goal)

    @property
    def history(self):
        """旧诊断读取入口；历史数据实际由 reward 模块维护。"""
        return self.reward.history

    def observation(self,state,motor_state=None):
        return build_observation(state,goal=self.goal,frame=self.frame,static=self.static,
            background=self.bg,bg_velocity=self.bgvel,radii=self.radii,dtype=self.dtype,
            config=self.settings["observation"],motor_state=motor_state,
            boundary_half_extent=(self.cfg['boundary_half_extent_x'],self.cfg['boundary_half_extent_y']))

    def reward_done(self,state,previous_distance,max_steps=None,previous_state=None,previous_bg=None,diagnostic_state=None,active=None):
        """组织一次逻辑结算：事件/终局先计算，奖励随后计算。"""
        goal_height=self.settings['simulation']['target_height'] if self.flight_mode=='free_3d' else None
        task=evaluate_task(state,goal_position=self.goal,static=self.static,
            background=self.bg,radii=self.radii,steps=self.steps+1,max_steps=self.settings["task"]["max_episode_steps"] if max_steps is None else max_steps,
            previous_state=state if previous_state is None else previous_state,
            previous_bg=self.bg if previous_bg is None else previous_bg,
            diagnostic_state=state if diagnostic_state is None else diagnostic_state,
            flight_mode=self.flight_mode,config=self.settings["task"],
            evader_radius=self.cfg["robot_radius"],
            boundary_half_extent=(self.cfg["boundary_half_extent_x"],self.cfg["boundary_half_extent_y"]),
            active=active,goal_height=goal_height)
        self.steps+=1
        reward,terms=self.reward.compute(state,goal_position=self.goal,
            previous_distance=previous_distance,steps=self.steps,task=task)
        info=dict(task.info,reward_terms=terms,reward_raw_terms=self.reward.last_raw.clone(),
                  reward_term_names=self.reward.term_names,
                  outcome_names=("ongoing","captured","static_collision","dynamic_collision","out_of_bounds","physical_failure","success","timeout"))
        return reward,task.terminated,task.truncated,info
