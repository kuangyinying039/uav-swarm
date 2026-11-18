"""通用批量几何计算；不依赖角色、策略、安全控制或任务规则。"""
import torch


def segment_clearance(start, end, radius):
    """返回相对运动线段到圆边界的最小有符号净空。

    start/end: [..., 2]，两个圆心在步首/步末的相对位置。
    radius: 可广播至 [...] 的半径和；静态圆也可使用同一公式。
    返回 [...]: >0 分离，=0 相切，<0 重叠。是否将相切判为碰撞由调用方决定。
    只进行张量计算，不修改输入；覆盖零长度线段。
    """
    delta = end - start
    t = (-(start * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return torch.linalg.vector_norm(start + t[..., None] * delta, dim=-1) - radius


