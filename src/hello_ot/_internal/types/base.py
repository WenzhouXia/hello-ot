from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


ArrayType = Any


@dataclass
class HierarchyLevel:
    """CN: restricted OT 的单层数据。 EN: One restricted-OT hierarchy level."""

    level_idx: int
    points: ArrayType
    masses: ArrayType
    cost_vec: Optional[ArrayType] = None
    child_labels: Optional[ArrayType] = None


class BaseHierarchy(ABC):
    """CN: HELLO 所需的最小 hierarchy 接口。 EN: Minimal hierarchy interface required by HELLO."""

    def __init__(self, levels: list[HierarchyLevel]):
        self.levels = list(levels)

    @property
    def finest_level(self) -> HierarchyLevel:
        return self.levels[0]

    @property
    def coarsest_level(self) -> HierarchyLevel:
        return self.levels[-1]

    @abstractmethod
    def prolongate(self, coarse_potential: ArrayType, coarse_level_idx: int, fine_level_idx: int) -> ArrayType:
        raise NotImplementedError


class BaseStrategy(ABC):
    """CN: candidate-edge strategy 的最小接口。 EN: Minimal candidate-edge strategy interface."""

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        return None
