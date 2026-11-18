"""离散难度调度。按滑动捕获率在中等、困难、严峻之间选择下一局。

规则与原先的 DiscreteAEG 相同：样本不足的档优先，捕获率落在
[r_min, r_max] 的档次之，整体太高就偏向更难，太低就偏向更容易。
"""

from __future__ import annotations

from collections import defaultdict, deque
import random
from dataclasses import dataclass, field

import numpy as np


LEVEL_SPEED = {
    "medium": (1.46, 0.84),
    "hard": (1.52, 0.88),
    "severe": (1.72, 1.00),
}
DEFAULT_LEVELS = ("medium", "hard", "severe")


@dataclass
class AEGConfig:
    levels: tuple[str, ...] = DEFAULT_LEVELS
    r_min: float = 0.5
    r_max: float = 0.9
    window: int = 50
    cold_start_level: str = "hard"
    min_samples_per_level: int = 5


@dataclass
class DiscreteAEG:
    cfg: AEGConfig = field(default_factory=AEGConfig)
    _history: dict[str, deque] = field(default_factory=dict)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    current_level: str = "hard"
    selections: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        levels = tuple(str(level) for level in self.cfg.levels)
        if not levels:
            raise ValueError("AEG requires at least one difficulty level")
        unknown = [level for level in levels if level not in LEVEL_SPEED]
        if unknown:
            raise ValueError(f"Unknown AEG levels: {unknown}")
        cold = str(self.cfg.cold_start_level)
        if cold not in levels:
            cold = levels[min(1, len(levels) - 1)]
        object.__setattr__(self, "cfg", AEGConfig(
            levels=levels,
            r_min=float(self.cfg.r_min),
            r_max=float(self.cfg.r_max),
            window=max(int(self.cfg.window), 1),
            cold_start_level=cold,
            min_samples_per_level=max(int(self.cfg.min_samples_per_level), 1),
        ))
        self._history = {level: deque(maxlen=self.cfg.window) for level in self.cfg.levels}
        self.current_level = self.cfg.cold_start_level
        self.selections = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def observe(self, level: str, captured: bool) -> None:
        level = str(level)
        if level not in self._history:
            return
        self._history[level].append(1.0 if captured else 0.0)

    def capture_rate(self, level: str) -> float:
        hist = self._history.get(str(level))
        if not hist:
            return float("nan")
        return float(np.mean(hist))

    def global_capture_rate(self) -> float:
        values = [value for hist in self._history.values() for value in hist]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _mid_band_levels(self) -> list[str]:
        chosen = []
        for level in self.cfg.levels:
            hist = self._history[level]
            if len(hist) < self.cfg.min_samples_per_level:
                continue
            rate = float(np.mean(hist))
            if self.cfg.r_min <= rate <= self.cfg.r_max:
                chosen.append(level)
        return chosen

    def _under_explored(self) -> list[str]:
        return [level for level in self.cfg.levels if len(self._history[level]) < self.cfg.min_samples_per_level]

    def next_level(self) -> str:
        under = self._under_explored()
        if under:
            choice = str(under[int(self._rng.integers(len(under)))])
        else:
            mid = self._mid_band_levels()
            if mid:
                choice = str(mid[int(self._rng.integers(len(mid)))])
            else:
                order = list(self.cfg.levels)
                rate = self.global_capture_rate()
                if np.isfinite(rate) and rate > self.cfg.r_max:
                    weights = np.linspace(0.5, 1.5, num=len(order))
                elif np.isfinite(rate) and rate < self.cfg.r_min:
                    weights = np.linspace(1.5, 0.5, num=len(order))
                else:
                    weights = np.ones(len(order))
                weights = weights / weights.sum()
                choice = str(self._rng.choice(order, p=weights))
        self.current_level = choice
        self.selections[choice] += 1
        return choice

    def stats(self) -> dict:
        return {
            "current_level": self.current_level,
            "capture_rates": {level: self.capture_rate(level) for level in self.cfg.levels},
            "samples": {level: len(self._history[level]) for level in self.cfg.levels},
            "selections": {level: int(self.selections.get(level, 0)) for level in self.cfg.levels},
            "global_capture_rate": self.global_capture_rate(),
            "r_min": self.cfg.r_min,
            "r_max": self.cfg.r_max,
            "window": self.cfg.window,
            "levels": list(self.cfg.levels),
        }


class LayoutCurriculum:
    """70% 重放未撞柱的旧局并扰动出生点，30% 抽新的障碍布局。"""

    def __init__(self, local_probability: float = 0.7, noise: float = 1.5, limit: int = 512):
        self.local_probability = float(local_probability)
        self.noise = float(noise)
        self.pool: deque[int] = deque(maxlen=int(limit))
        self._rng = random.Random(0)

    def set_seed(self, seed: int) -> None:
        self._rng.seed(int(seed))

    def note(self, seed: int, captured: bool, collision: bool) -> None:
        if not collision:
            self.pool.append(int(seed))

    def choose(self, fresh: list[int]) -> tuple[list[int], list[bool]]:
        seeds, nudge = [], []
        for seed in fresh:
            if self.pool and self._rng.random() < self.local_probability:
                seeds.append(self._rng.choice(self.pool))
                nudge.append(True)
            else:
                seeds.append(int(seed))
                nudge.append(False)
        return seeds, nudge


def reset_with_curriculum(env, finished, episode_seed, next_seed, captured, outcome, curriculum: LayoutCurriculum) -> int:
    import torch

    count = int(finished.numel())
    fresh = list(range(next_seed, next_seed + count))
    for local_index, index in enumerate(finished.tolist()):
        code = int(outcome[index])
        curriculum.note(episode_seed[index], bool(captured[index].item()), code in (2, 3))
    seeds, nudge = curriculum.choose(fresh)
    env.reset(finished, seeds=seeds)
    mask = torch.tensor(nudge, device=finished.device, dtype=torch.bool)
    if bool(mask.any()):
        env.nudge_starts(finished[mask], curriculum.noise)
    for local_index, index in enumerate(finished.tolist()):
        episode_seed[index] = seeds[local_index]
    return next_seed + count
