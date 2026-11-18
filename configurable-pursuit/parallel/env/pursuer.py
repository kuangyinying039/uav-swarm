"""人工追捕状态机与每局策略选择；不推进物理、不修改背景、不执行安全修正。"""
import hashlib
import json
import numpy as np
from parallel.strategies.config import load_config
from parallel.strategies.base import Context
import torch


def norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def unit(x):
    return torch.where(norm(x)>1e-9, x/norm(x).clamp_min(1e-9), torch.zeros_like(x))


class ScriptedPursuer:
    def __init__(self, n, count, device, dtype=torch.float32, config=None, dt=None):
        self.count, self.device, self.dtype = count, device, dtype
        self.config,self.strategy_names,self.strategies,self.probabilities=load_config(config)
        if dt is None:
            from parallel.env.config_loader import load_settings
            dt=load_settings()['simulation']['dt']
        self.common=self.config['common'];self.dt=dt
        self.strategy_ids=torch.zeros(n,count,device=device,dtype=torch.long)
        self.starts=torch.zeros(n,2,device=device,dtype=dtype)
        self.targets = torch.zeros(n,count,2,device=device,dtype=dtype)
        self.mode = torch.zeros(n,count,device=device,dtype=torch.long)
        self.ever = torch.zeros_like(self.mode,dtype=torch.bool)
        self.detected = torch.zeros_like(self.ever)
        self.search = torch.zeros_like(self.mode)
        self.last_seen = torch.zeros_like(self.targets)
        self.nominal = torch.zeros_like(self.targets)

    def reset(self, ids, initial):
        """仅重置指定局；initial 为这些局的采样快照。"""
        def tensor(data, dtype=None):
            return torch.as_tensor(np.asarray(data),device=self.device,dtype=dtype or self.dtype)
        objects = [[o for o in s['dynamic_obstacles'] if o['is_pursuer']] for s in initial]
        if not objects: return
        self.targets[ids] = tensor([[[o['goal_x'],o['goal_y']] for o in row] for row in objects]).reshape(len(initial),self.count,2)
        for name,field in [('mode','pursuer_mode'),('ever','pursuer_ever_detected'),('detected','pursuer_detected'),('search','pursuer_search_steps_remaining')]:
            getattr(self,name)[ids] = tensor([[o[field] for o in row] for row in objects],getattr(self,name).dtype)
        self.last_seen[ids] = tensor([[[o['last_seen_x'],o['last_seen_y']] for o in row] for row in objects]).reshape(len(initial),self.count,2)
        self.nominal[ids] = 0
        self.starts[ids] = tensor([s['start'] for s in initial])
        # 独立随机流：从局初始 RNG 状态与配置 seed 派生，不消耗地图/巡逻 RNG。
        # 同 seed 或 resample=False 得到相同策略；部分 reset 只写对应局。
        choices=[]
        for snapshot in initial:
            payload=json.dumps(snapshot['rng_state'],sort_keys=True,default=lambda x:x.tolist())
            digest=hashlib.sha256((str(self.config['sampling_seed'])+':'+payload).encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:16],'little'))
            choices.append(rng.choice(len(self.strategies),size=self.count,p=self.probabilities))
        self.strategy_ids[ids]=tensor(choices,torch.long)

    def act(self, state, origin, speed, sample_goal, *, goal=None, static=None,
            radii=None, bounds=None, activity_bounds=None):
        """返回全组 P 的世界系期望速度。策略计算与公共状态切换解耦。

        goal/static 等是显式场景输入；sample_goal 仅供巡逻时采样。
        """
        e,p=state[:,0,:2],state[:,1:,:2]
        # 仅 direct_chase 的旧调用可省略地图；其他策略要求完整场景。
        if goal is None:
            if any(name!='direct_chase' and prob>0 for name,prob in zip(self.strategy_names,self.probabilities)):
                raise ValueError('Non-direct policies require scene context')
            goal=e;static=torch.empty(len(e),0,3,device=self.device,dtype=self.dtype)
            radii=torch.full_like(speed,.3);bounds=e.new_tensor([10.5,13.])
        if activity_bounds is None:
            activity_bounds=e.new_full((2,),self.config['spawn']['region_half_extent'])
        c=Context(e,state[:,0,7:9],p,origin,self.starts,goal,static,radii,bounds,
                  torch.where((torch.arange(self.count,device=self.device)+3)%2==0,1.,-1.).to(self.dtype))
        common=self.common
        radius=torch.full_like(speed,common['responsibility_radius'])
        deployment=torch.zeros_like(self.detected)
        deploy_radius=radius.clone();deploy_speed=torch.zeros_like(speed)
        for index,strategy in enumerate(self.strategies):
            settings=strategy.deployment()
            if settings is not None:
                mask=self.strategy_ids==index
                deployment |= mask
                deploy_radius=torch.where(mask,torch.full_like(radius,settings[0]),deploy_radius)
                deploy_speed=torch.where(mask,torch.full_like(speed,settings[1]),deploy_speed)
                radius=torch.where(mask,torch.maximum(radius,deploy_radius),radius)
        inside=(norm(e[:,None]-origin).squeeze(-1)<=radius)|(radius<=0)
        visible=norm(e[:,None]-p).squeeze(-1)<=common['detection_range']
        self.detected=inside & (visible if common['use_detection_range'] else torch.ones_like(visible))
        searching=~self.detected & inside & self.ever & (self.search>0)
        deploying=~self.detected & ~searching & deployment
        returning=~self.detected & ~searching & ~deploying & (~inside | self.ever)
        arrived=returning & (norm(origin-p).squeeze(-1)<=common['goal_switch_threshold'])
        patrol=(~self.detected & ~searching & ~returning & ~deploying)|arrived
        self.mode=torch.where(self.detected,1,torch.where(searching,2,torch.where(deploying,4,torch.where(returning & ~arrived,3,0))))
        self.ever=((self.ever|self.detected)&~arrived)&~deploying
        self.search=torch.where(self.detected,max(1,round(common['search_duration']/self.dt)),torch.where(searching,self.search-1,0))
        self.last_seen=torch.where(self.detected[...,None],e[:,None],self.last_seen)
        targets=self.targets.clone()
        for index,strategy in enumerate(self.strategies):
            # 没选中该策略时跳过，尤其避免无意义堵点搜索。
            selected=self.strategy_ids==index
            if not selected.any():continue
            live=selected & self.detected
            if live.any():
                candidate=strategy.target(c,live=True)
                candidate=torch.maximum(torch.minimum(candidate,activity_bounds),-activity_bounds)
                delta=candidate-origin
                r=common['responsibility_radius']
                if r>0:candidate=origin+delta*(r/norm(delta).clamp_min(r))
                targets=torch.where(live[...,None],candidate,targets)
            idle=selected & deploying
            if idle.any():
                candidate=strategy.target(c,live=False)
                candidate=torch.maximum(torch.minimum(candidate,activity_bounds),-activity_bounds)
                delta=candidate-origin;r=strategy.deployment()[0]
                if r>0:candidate=origin+delta*(r/norm(delta).clamp_min(r))
                targets=torch.where(idle[...,None],candidate,targets)
        self.targets=torch.where(searching[...,None],self.last_seen,targets)
        self.targets=torch.where((returning & ~arrived)[...,None],origin,self.targets)
        switches=arrived | (patrol & (norm(self.targets-p).squeeze(-1)<common['goal_switch_threshold']))
        if common['idle_mode']=='hold':switches=arrived
        for i,j in switches.nonzero().cpu().tolist():
            target=sample_goal(i,j)
            if arrived[i,j] and np.linalg.norm(np.asarray(target)-p[i,j].cpu().numpy())<common['goal_switch_threshold']:
                target=sample_goal(i,j)
            self.targets[i,j]=torch.as_tensor(target,device=self.device,dtype=self.dtype)
        commanded_speed=torch.where(self.detected|searching,speed,torch.minimum(speed,torch.full_like(speed,common['patrol_speed'])))
        commanded_speed=torch.where(deploying,torch.minimum(speed,deploy_speed),commanded_speed)
        stopped=deploying & (norm(self.targets-p).squeeze(-1)<=common['goal_switch_threshold'])
        if common['idle_mode']=='hold':stopped |= patrol & ~arrived
        commanded_speed=torch.where(stopped,torch.zeros_like(speed),commanded_speed)
        self.nominal=unit(self.targets-p)*commanded_speed[...,None]
        return self.nominal
