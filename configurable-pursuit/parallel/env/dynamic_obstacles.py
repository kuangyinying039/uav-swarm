"""非追捕者动态圆的运动；运动学更新，不调用无人机物理或追捕策略。"""
import torch


def norm(x):return torch.linalg.vector_norm(x,dim=-1,keepdim=True)
def unit(x):return torch.where(norm(x)>1e-9,x/norm(x).clamp_min(1e-9),torch.zeros_like(x))


class DynamicObstacleMotion:
    def __init__(self,n,count,device,dtype,*,dt,goal_threshold,local_half_range,region_extent):
        self.dt=dt
        self.goal_threshold=goal_threshold
        self.local_half_range=local_half_range
        self.region_extent=region_extent
        self.position=torch.zeros(n,count,2,device=device,dtype=dtype)
        self.velocity=torch.zeros_like(self.position)
        self.targets=torch.zeros_like(self.position)
        # 按局标记：部分reset不能让其他局已经准备的目标被再次采样。
        self.prepared=torch.zeros(n,device=device,dtype=torch.bool)

    def reset(self,ids,initial):
        """initial 只包含 ids 对应局的采样快照；不改其他局。"""
        if not ids.numel():return
        count=self.position.shape[1]
        if count==0:
            self.prepared[ids]=False
            return
        objects=[s['dynamic_obstacles'][:count] for s in initial]
        def tensor(data):return torch.as_tensor(data,device=self.position.device,dtype=self.position.dtype).reshape(len(initial),count,2)
        self.targets[ids]=tensor([[[o['goal_x'],o['goal_y']] for o in row] for row in objects])
        self.position[ids]=tensor([[[o['x'],o['y']] for o in row] for row in objects])
        self.velocity[ids]=tensor([[[o['vx'],o['vy']] for o in row] for row in objects])
        self.prepared[ids]=False

    def prepare(self,sample_goal):
        """每局每步只准备一次目标，回调使用 scene 的逐局随机流。"""
        switches=(norm(self.targets-self.position).squeeze(-1)<self.goal_threshold)&~self.prepared[:,None]
        for i,j in switches.nonzero().cpu().tolist():
            self.targets[i,j]=torch.as_tensor(sample_goal(i,j),device=self.position.device,dtype=self.position.dtype)
        self.prepared[:]=True

    def advance(self,origin,speed,sample_goal):
        """朝目标直线运动，在出生点局部范围与全局活动区交集中裁剪。"""
        self.prepare(sample_goal)
        old=self.position.clone()
        self.position=torch.maximum(torch.minimum(self.position+unit(self.targets-self.position)*speed[...,None]*self.dt,
            torch.minimum(origin+self.local_half_range,torch.full_like(self.position,self.region_extent))),
            torch.maximum(origin-self.local_half_range,torch.full_like(self.position,-self.region_extent)))
        self.velocity=(self.position-old)/self.dt
        self.prepared[:]=False
