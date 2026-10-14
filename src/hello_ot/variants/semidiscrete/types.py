from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Tuple

import numpy as np

from hello_ot.variants._records import RecordMapping


@dataclass(frozen=True, eq=False)
class SemiDiscreteIterationRecord(RecordMapping):
    """
    CN: 一次经验 OT 重复求解及 target dual 平均更新的记录。
    EN: Record of one empirical-OT repeat and target-dual averaging update.
    """

    repeat_index: int
    source_seed: int
    mode: Literal["cold", "target_dual_warm_start"]
    wall_time: float
    objective: float
    peak_active_support_size: int


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
    records: Tuple[SemiDiscreteIterationRecord, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["SemiDiscreteIterationRecord", "SemiDiscreteResult"]
