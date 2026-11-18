"""公开演示基线：仅依赖环境状态/几何，无研究策略、权重或训练数据依赖。"""
import torch


def unit(x):
    return x/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-6)


def action(env, policy='goal_avoid'):
    """输出wrapper需要的局部归一化动作；基础势场可能卡住，不是研究算法。"""
    state=env.state();r=env.unwrapped.rules;e=state[:,0,:2]
    if policy=='hold':return torch.zeros_like(e)
    desired=unit(r.goal-e)
    if policy=='goal_avoid':
        positions=torch.cat((r.static[:,:,:2],r.bg,state[:,1:,:2]),1)
        radii=torch.cat((r.static[:,:,2],r.radii),1)
        delta=e[:,None]-positions
        clearance=torch.linalg.vector_norm(delta,dim=-1)-radii-r.cfg['robot_radius']
        strength=(1-clearance/2.).clamp(0,2)
        desired=desired+(unit(delta)*strength[...,None]).sum(1)
        bounds=e.new_tensor([r.cfg['boundary_half_extent_x'],r.cfg['boundary_half_extent_y']])
        wall_gap=bounds-r.cfg['robot_radius']-e.abs()
        desired-=e.sign()*(1-wall_gap/1.).clamp(0,2)
    elif policy!='goal':raise ValueError('Unknown public evader baseline')
    # frame rows are the local axes in world coordinates.
    local=torch.einsum('nd,nkd->nk',unit(desired),r.frame)
    return local.clamp(-1,1)
