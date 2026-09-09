from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from hello_ot.types import SparseOTSolution


@dataclass(frozen=True)
class UnbalancedResult:
    """
    CN: KL 非平衡最优传输（UOT）求解结果。
    EN: Solution result for KL-unbalanced optimal transport (UOT).
    """

    objective: float
    dual_objective: float
    relative_primal_dual_gap: float
    solution: SparseOTSolution
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    transported_mass: float
    total_wall_time: float
    iterations: int
    converged: bool
    records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["UnbalancedResult"]
