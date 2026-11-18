"""具名 E 奖励：原始信号×scale；事件互斥优先，过程奖励仅在无事件时生效。"""
import torch
from parallel.env.config_loader import load_settings,TERM_NAMES,EVENT_NAMES


def norm(x):return torch.linalg.vector_norm(x,dim=-1,keepdim=True)
def unit(x):return torch.where(norm(x)>1e-9,x/norm(x).clamp_min(1e-9),torch.zeros_like(x))


class EvaderReward:
    term_names=TERM_NAMES

    def __init__(self,n,device,dtype,config=None):
        self.config=load_settings()['reward'] if config is None else config
        self.history=torch.zeros(n,self.config['stall']['window_steps']+1,device=device,dtype=dtype)
        self.last_raw=None

    def reset(self,ids,initial_distance):
        self.history[ids]=initial_distance
        if self.last_raw is not None:self.last_raw[ids]=0

    def progress_signal(self,previous_distance,distance,terminated,truncated):
        p=self.config['progress']
        if p['mode']=='distance':return previous_distance-distance
        # Phi(s) = -distance / distance_scale，单位由配置显式定义。
        previous_phi=-previous_distance/p['potential_distance_scale']
        next_phi=-distance/p['potential_distance_scale']
        zero=torch.zeros_like(terminated)
        if p['zero_potential_on_termination']:zero |= terminated
        if p['zero_potential_on_truncation']:zero |= truncated
        next_phi=torch.where(zero,torch.zeros_like(next_phi),next_phi)
        return p['gamma']*next_phi-previous_phi

    def stall_signal(self,state,steps,task):
        p=self.config['stall'];progress=self.history[:,0]-task.distance
        active=(steps>=p['window_steps'])&(task.distance<=p['near_goal_distance'])
        num_pursuers=state.shape[1]-1
        num_dynamic=task.dynamic_clearance.shape[1]-num_pursuers
        nearest=task.dynamic_clearance[:,num_dynamic:].amin(-1) if num_pursuers else torch.full_like(task.distance,float('inf'))
        active &= (task.static_clearance>=p['min_static_clearance'])&(nearest>=p['min_pursuer_clearance'])
        active &= (progress<p['min_progress'])&(norm(state[:,0,7:9]).squeeze(-1)<p['max_speed'])&~task.info['success']
        return torch.where(active,(1-progress.clamp_min(0)/p['min_progress']).clamp(0,1),0.)

    def compute(self,state,*,goal_position,previous_distance,steps,task):
        self.history=torch.roll(self.history,-1,1);self.history[:,-1]=task.distance
        event_flags=(task.info['captured'],task.info['static_collision'],task.info['dynamic_collision'],
                     task.info['boundary_collision'],task.info['physical_failure'],task.info['success'],task.truncated)
        signals={name:flag.to(state.dtype) for name,flag in zip(EVENT_NAMES,event_flags)}
        signals['progress']=self.progress_signal(previous_distance,task.distance,task.terminated,task.truncated)
        signals['goal_velocity']=(state[:,0,7:9]*unit(goal_position-state[:,0,:2])).sum(-1)
        signals['step']=torch.ones_like(task.distance)
        signals['stall']=self.stall_signal(state,steps,task)
        self.last_raw=torch.stack([signals[k] for k in self.term_names],-1)
        # 不根据 enabled/scale 选择事件：关闭被捕惩罚不会回退到成功或前进奖励。
        outcome=task.info['outcome'];dense=outcome==0
        terms=[]
        for index,name in enumerate(self.term_names):
            cfg=self.config['terms'][name]
            active=(outcome==index+1) if name in EVENT_NAMES else dense
            scale=cfg['scale'] if cfg['enabled'] else 0.
            terms.append(torch.where(active,signals[name]*scale,torch.zeros_like(task.distance)))
        terms=torch.stack(terms,-1)
        return terms.sum(-1),terms
