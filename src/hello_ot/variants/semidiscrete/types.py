from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SemiDiscreteResult:
    """
    CN: 由重复经验 OT 对偶平均得到的 semi-discrete OT 结果。
    EN: Semi-discrete OT result obtained by averaging repeated empirical-OT duals.
    """

    target_dual: np.ndarray
    sample_duals: Optional[np.ndarray]
    source_seeds: Tuple[int, ...]
    num_repeats: int
    source_sample_count: int
    target_shape: Tuple[int, int]
    source_sampling_backend: str
    records: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["SemiDiscreteResult"]
