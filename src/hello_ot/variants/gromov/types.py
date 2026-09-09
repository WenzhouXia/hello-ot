from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from hello_ot.types import SparseOTSolution


@dataclass(frozen=True)
class GromovResult:
    """
    CN: 平方欧氏低秩 Gromov-Wasserstein 求解结果。
    EN: Solution result for squared-Euclidean low-rank Gromov-Wasserstein.
    """

    objective: float
    solution: SparseOTSolution
    total_wall_time: float
    iterations: int
    converged: bool
    records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["GromovResult"]
