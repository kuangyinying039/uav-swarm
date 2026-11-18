"""Private zero-distance regularization of upstream downwash, no model removal."""
import torch


def finite_downwash(p0, p1, p1_t, kr=2, kz=1):
    # Same upstream normalization and positive downstream axial separation.
    direction = p1_t / (torch.norm(p1_t,dim=-1,keepdim=True)+1e-6)
    relative = p1.unsqueeze(0)-p0.unsqueeze(1)
    z = (relative*direction).sum(-1,keepdim=True)
    radial = torch.norm(relative-z*direction,dim=-1,keepdim=True)
    z = z.clamp_min(0)
    # Define the coplanar limit as zero induced downwash. For z>0 retain
    # the upstream function; 1um denominator prevents only 0/0 at coincidence.
    weight = torch.exp(-.5*(kr*radial/z.clamp_min(1e-6)).square())/(1+kz*z).square()
    weight = torch.where(z>0,weight,torch.zeros_like(weight))
    force=weight * -p1_t
    n=force.shape[0]
    if n==1: return force.new_empty(1,0,3)
    return force.flatten(0,1)[1:].unflatten(0,(n-1,n+1))[:,:-1].reshape(n,n-1,3)
