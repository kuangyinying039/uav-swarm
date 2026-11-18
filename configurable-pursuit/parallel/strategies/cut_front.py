import torch
from .base import Strategy,unit,norm

class CutFront(Strategy):
    defaults={'cut_in_distance':2.0}
    def target(self,c,live=True):
        heading=unit(c.ev)[:,None].expand_as(c.p)
        heading=torch.where(norm(heading)>1e-9,heading,unit(c.e[:,None]-c.p))
        return c.e[:,None]+heading*self.params['cut_in_distance']
