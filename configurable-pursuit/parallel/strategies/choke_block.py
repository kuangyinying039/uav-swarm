import torch
from .base import Strategy,unit,norm,side

class ChokeBlock(Strategy):
    defaults={'min_gap':1.2,'max_gap':3.5,'route_half_width':2.5,
        'min_forward_distance':1.,'max_forward_distance':20.,'formation_offset':.25,
        'fallback_distance':3.,'deployment_radius':10.,'deployment_speed':1.2,
        'route_progress':.55,'static_margin':.7}
    def __init__(self,params=None):
        super().__init__(params)
        if self.params['route_progress']>1 or self.params['max_forward_distance']<self.params['min_forward_distance']:
            raise ValueError('Invalid choke route progress or forward interval')
    def deployment(self):
        return self.params['deployment_radius'],self.params['deployment_speed']
    def target(self,c,live=True):
        p=self.params
        anchor=c.e if live else c.start
        heading=unit(c.ev) if live else unit(c.goal-c.start)
        if live: heading=torch.where(norm(heading)>1e-9,heading,unit(c.goal-anchor))
        desired=torch.full_like(anchor[:,0],p['fallback_distance']) if live else norm(c.goal-c.start).squeeze(-1)*p['route_progress']
        fallback=anchor+heading*desired[:,None]
        result=fallback[:,None].expand_as(c.p).clone()
        best=torch.full_like(c.p[...,0],float('inf'))
        safe=c.radii+p['static_margin']
        minimum=torch.maximum(torch.full_like(safe,p['min_gap']),2*safe)
        maximum=torch.maximum(torch.full_like(safe,p['max_gap']),minimum)
        # 按障碍对次序更新，只在严格更优时替换，保留同分规则。
        # 每对批处理全部环境和 P，避免 N×P×障碍对×障碍数的大临时张量。
        for i in range(c.static.shape[1]-1):
            for j in range(i+1,c.static.shape[1]):
                a,b=c.static[:,i],c.static[:,j]
                delta=b[:,:2]-a[:,:2];length=norm(delta).squeeze(-1)
                direction=unit(delta);gap=length-a[:,2]-b[:,2]
                center=a[:,:2]+direction*(a[:,2]+.5*gap)[:,None]
                rel=center-anchor
                forward=(rel*heading).sum(-1);lateral=(rel*side(heading)).sum(-1).abs()
                offset=torch.minimum(torch.full_like(safe,p['formation_offset']),(.5*gap[:,None]-safe).clamp_min(0))
                target=center[:,None]+direction[:,None]*c.side_sign[None,:,None]*offset[...,None]
                clearance=norm(target[:,:,None]-c.static[:,None,:,:2]).squeeze(-1)-c.static[:,None,:,2]-safe[:,:,None]
                valid=(length[:,None]>1e-9)&(gap[:,None]>=minimum)&(gap[:,None]<=maximum)
                valid &= ((forward>=p['min_forward_distance'])&(forward<=p['max_forward_distance'])&(lateral<=p['route_half_width']))[:,None]
                valid &= (clearance>=-1e-9).all(-1)
                score=2*lateral[:,None]+.25*(forward-desired).abs()[:,None]+.1*gap[:,None]+.02*norm(target-c.p).squeeze(-1)
                better=valid&(score<best)
                result=torch.where(better[...,None],target,result)
                best=torch.where(better,score,best)
        return result
