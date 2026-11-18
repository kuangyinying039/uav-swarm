"""Discrete Adaptive Environment Generator for Radar5 difficulty ladders.

Unlike the THU continuous GenBuffer (cylinder task vectors + farthest-point
sampling), this scheduler only selects among existing ``medium/hard/severe``
domain configs using a rolling capture-rate band, matching OPEN's
``[R_min, R_max]`` curriculum idea without breaking the 5-building contract.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_AEG_LEVELS = ("medium", "hard", "severe")


@dataclass
class AEGConfig:
    levels: tuple[str, ...] = DEFAULT_AEG_LEVELS
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
        object.__setattr__(self, "cfg", AEGConfig(
            levels=levels,
            r_min=float(self.cfg.r_min),
            r_max=float(self.cfg.r_max),
            window=max(int(self.cfg.window), 1),
            cold_start_level=str(self.cfg.cold_start_level),
            min_samples_per_level=max(int(self.cfg.min_samples_per_level), 1),
        ))
        if self.cfg.cold_start_level not in self.cfg.levels:
            object.__setattr__(
                self,
                "cfg",
                AEGConfig(
                    **{
                        **self.cfg.__dict__,
                        "cold_start_level": self.cfg.levels[min(1, len(self.cfg.levels) - 1)],
                    }
                ),
            )
        self._history = {
            level: deque(maxlen=self.cfg.window) for level in self.cfg.levels
        }
        self.current_level = self.cfg.cold_start_level
        self.selections = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def observe(self, level: str, captured: bool) -> None:
        level = str(level)
        if level not in self._history:
            # Ignore unknown ladder tags rather than crashing a long Isaac run.
            return
        self._history[level].append(1.0 if captured else 0.0)

    def capture_rate(self, level: str) -> float:
        hist = self._history.get(str(level))
        if not hist:
            return float("nan")
        return float(np.mean(hist))

    def global_capture_rate(self) -> float:
        values = [v for hist in self._history.values() for v in hist]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _mid_band_levels(self) -> list[str]:
        mid = []
        for level in self.cfg.levels:
            hist = self._history[level]
            if len(hist) < self.cfg.min_samples_per_level:
                continue
            rate = float(np.mean(hist))
            if self.cfg.r_min <= rate <= self.cfg.r_max:
                mid.append(level)
        return mid

    def _under_explored(self) -> list[str]:
        return [
            level
            for level in self.cfg.levels
            if len(self._history[level]) < self.cfg.min_samples_per_level
        ]

    def next_level(self) -> str:
        """Pick the next training domain.

        Priority:
        1. Levels still below ``min_samples_per_level`` (cold start / balance)
        2. Levels whose rolling capture rate sits in ``[r_min, r_max]``
        3. If global rate is high, bias harder; if low, bias easier
        4. Fallback: cold_start_level
        """
        under = self._under_explored()
        if under:
            choice = str(under[int(self._rng.integers(len(under)))])
            self.current_level = choice
            self.selections[choice] += 1
            return choice

        mid = self._mid_band_levels()
        if mid:
            choice = str(mid[int(self._rng.integers(len(mid)))])
            self.current_level = choice
            self.selections[choice] += 1
            return choice

        global_rate = self.global_capture_rate()
        order = list(self.cfg.levels)
        if np.isfinite(global_rate):
            if global_rate > self.cfg.r_max:
                # Prefer harder domains (later in the nominal ladder order).
                weights = np.linspace(0.5, 1.5, num=len(order))
            elif global_rate < self.cfg.r_min:
                weights = np.linspace(1.5, 0.5, num=len(order))
            else:
                weights = np.ones(len(order))
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


def resolve_aeg_levels(levels: Sequence[str] | None = None) -> tuple[str, ...]:
    if not levels:
        return DEFAULT_AEG_LEVELS
    return tuple(str(level) for level in levels)
