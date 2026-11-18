"""E 外部控制、P 人工策略的薄适配层；底层环境可绕过本层直接接收全体动作。"""
import torch
from parallel.env.pursuer import ScriptedPursuer
from parallel.env.pursuer_safety import SafetyController, cbf
from parallel.env.interface import TensorSpec


class ScriptedPursuerWrapper:
    # 防止拼错属性或给内部设置赋值时，悄悄在 wrapper 上创建影子属性。
    __slots__ = ('_env', 'pursuer', 'safety')

    def __init__(self, env, pursuer_config=None):
        env._ensure_open()
        self._env = env
        rules = env.rules
        from parallel.strategies.config import load_config
        selected=rules.pursuer_config if pursuer_config is None else load_config(pursuer_config)[0]
        if selected['spawn']!=rules.pursuer_config['spawn']:
            raise ValueError('Pass pursuer spawn config to ParallelEnv before wrapping; scene is already initialized')
        self.pursuer = ScriptedPursuer(env.n, rules.initial_pos.shape[1]-1, env.device, rules.dtype, config=selected, dt=rules.cfg["dt"])
        self.safety = SafetyController(env.n, env.device, rules.dtype, count=rules.layout.num_pursuers, static_count=rules.static.shape[1],
            radius=selected['spawn']['radius'],north_boundary=selected['spawn']['region_half_extent'],dt=rules.cfg['dt'],speed_limit=selected['spawn']['speed_max'])
        ids = torch.arange(env.n,device=env.device)
        self.pursuer.reset(ids, rules.initial)

    @property
    def env(self):
        """直接包裹的环境，只读；正常交互请使用 wrapper 的公开方法。"""
        return self._env

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def n(self): return self._env.n

    @property
    def num_envs(self): return self._env.num_envs

    @property
    def device(self): return self._env.device

    @property
    def closed(self): return self._env.closed

    @property
    def action_spec(self):
        return TensorSpec((self.n,2), 'float32', 'dimensionless', 'task_local',
            f"E 动作；归一化后转成最大{self._env.rules.settings['simulation']['evader_speed_limit']}m/s世界系速度", (-1.,1.))

    @property
    def observation_spec(self): return self._env.observation_spec

    @property
    def observation_layout(self): return self._env.observation_layout

    @property
    def observation_scale(self): return self._env.observation_scale

    def motor_state(self): return self._env.motor_state()

    @property
    def state_spec(self): return self._env.state_spec

    @property
    def profile_reset(self): return self._env.profile_reset

    @profile_reset.setter
    def profile_reset(self, enabled): self.set_reset_profiling(enabled)

    def set_reset_profiling(self, enabled):
        self._env.set_reset_profiling(enabled)

    def state(self):
        self._env._ensure_open()
        return self._env.state()

    def observe(self):
        self._env._ensure_open()
        return self._env.observe()

    def close(self):
        self._env.close()

    def reset(self, ids=None, resample=True, seeds=None):
        self._env._ensure_open()
        if ids is None:
            ids = torch.arange(self.n,device=self.device)
        if not isinstance(ids,torch.Tensor) or ids.dtype != torch.long:
            raise TypeError("ids must be an int64 tensor or None")
        obs = self.env.reset(ids, resample=resample, seeds=seeds)
        ids = ids.to(device=self.env.device,dtype=torch.long)
        self.pursuer.reset(ids, [self._env.rules.initial[i] for i in ids.cpu().tolist()])
        self.safety.reset(ids)
        return obs

    def _sample_goal(self, i, j):
        return self._env.rules.scene.sample_pursuer_goal(i,j)

    @torch.no_grad()
    def step(self, action):
        state = self.state()
        action = action.to(device=self.env.device,dtype=state.dtype)
        if action.shape != (self.env.n,2) or not torch.isfinite(action).all():
            raise ValueError('Expected finite normalized E actions [N,2]')
        local = action.clamp(-1,1)
        local = local/torch.linalg.vector_norm(local,dim=-1,keepdim=True).clamp_min(1.)
        commands = torch.zeros_like(state[:,:,:2])
        commands[:,0] = self._env.rules.settings['simulation']['evader_speed_limit']*torch.einsum('nk,nkd->nd',local,self._env.rules.frame)
        # 每步固定先采样动态障碍目标，再采样 P 目标。
        self._env.rules.prepare_dynamic_obstacles()
        nominal = self.pursuer.act(state,self._env.rules.origin[:,self._env.rules.layout.pursuer_objects],self._env.rules.speed[:,self._env.rules.layout.pursuer_objects],self._sample_goal,
            goal=self._env.rules.goal,static=self._env.rules.static,radii=self._env.rules.radii[:,self._env.rules.layout.pursuer_objects],
            bounds=state.new_tensor([self._env.rules.cfg['boundary_half_extent_x'],self._env.rules.cfg['boundary_half_extent_y']]),
            activity_bounds=state.new_full((2,),self.pursuer.config['spawn']['region_half_extent']))
        commands[:,1:] = self.safety.commands(state[:,1:,:2],state[:,1:,7:9],
            cbf(state[:,1:,:2],nominal,self._env.rules.static,self.pursuer.config['spawn']['radius'],self.pursuer.config['spawn']['speed_max']),self._env.rules.static)
        obs,reward,term,trunc,info = self.env.step(commands)
        info.update(pursuer_strategy_ids=self.pursuer.strategy_ids.clone(),pursuer_strategy_names=self.pursuer.strategy_names,
                    pursuer_mode=self.pursuer.mode.clone())
        info.update(self.safety.observe_transition(state,info['state'],self._env.rules.static))
        return obs,reward,term,trunc,info
