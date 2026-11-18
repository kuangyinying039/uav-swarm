import torch
from .base import Strategy,unit,norm,side

class FollowPressure(Strategy):
    defaults={'rear_distance':.8,'lateral_distance':1.6}
    def target(self,c,live=True):
        heading=unit(c.ev)
        heading=torch.where(norm(heading)>1e-9,heading,unit(c.goal-c.e))
        heading=heading[:,None].expand_as(c.p)
        heading=torch.where(norm(heading)>1e-9,heading,unit(c.e[:,None]-c.p))
        delta=c.static[:,:,:2]-c.e[:,None];dist=norm(delta).squeeze(-1)
        clear=dist-c.static[:,:,2]
        clear=torch.where(dist>1e-9,clear,torch.full_like(clear,float('inf')))
        x,y=c.e.unbind(-1);bx,by=c.bounds
        edges=torch.stack((x+bx,bx-x,y+by,by-y),-1)
        dirs=torch.tensor([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]],device=c.e.device,dtype=c.e.dtype)
        directions=torch.cat((unit(delta),dirs[None].expand(len(x),-1,-1)),1)
        index=torch.cat((clear,edges),-1).argmin(-1)
        hazard=directions[torch.arange(len(x),device=x.device),index][:,None]
        lateral=hazard-heading*(hazard*heading).sum(-1,keepdim=True)
        lateral=torch.where(norm(lateral)>1e-9,lateral,side(heading)*c.side_sign[None,:,None])
        return c.e[:,None]-heading*self.params['rear_distance']-unit(lateral)*self.params['lateral_distance']
