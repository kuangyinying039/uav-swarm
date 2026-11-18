"""公开张量规格；shape 包含并行维，不依赖 Gym 或 Isaac。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    units: str
    frame: str
    description: str
    bounds: tuple[float, float] | None = None
