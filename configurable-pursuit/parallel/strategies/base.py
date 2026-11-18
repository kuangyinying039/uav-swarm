"""策略统一合同：输入整组可用状态，输出期望目标；不包含动力学或安全投影。"""
from dataclasses import dataclass
import math
import torch


def norm(x): return torch.linalg.vector_norm(x,dim=-1,keepdim=True)
def unit(x): return torch.where(norm(x)>1e-9,x/norm(x).clamp_min(1e-9),torch.zeros_like(x))
def side(x): return torch.stack((-x[...,1],x[...,0]),-1)


@dataclass
class Context:
    e: torch.Tensor
    ev: torch.Tensor
    p: torch.Tensor
    origin: torch.Tensor
    start: torch.Tensor
    goal: torch.Tensor
    static: torch.Tensor
    radii: torch.Tensor
    bounds: torch.Tensor
    side_sign: torch.Tensor


class Strategy:
    defaults = {}
    def __init__(self, params=None):
        params = params or {}
        unknown = set(params)-set(self.defaults)
        if unknown: raise ValueError(f'Unknown strategy parameters: {unknown}')
        self.params = dict(self.defaults,**params)
        for k,v in self.params.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                raise ValueError(f'{k} must be finite and nonnegative')

    def target(self, c, live=True): raise NotImplementedError
    def deployment(self):
        """无目标时特殊部署的 (责任半径,速度)；普通策略返回 None。"""
        return None
