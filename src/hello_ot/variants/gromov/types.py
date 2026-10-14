from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from hello_ot.types import SparseOTSolution
from hello_ot.variants._records import RecordMapping


@dataclass(frozen=True, eq=False)
class GromovIterationRecord(RecordMapping):
    """
    CN: 一次 GW full-step 外层迭代的记录。
    EN: Record of one GW full-step outer iteration.
    """

    iteration: int
    wall_time: float
    objective: float
    relative_objective_change: float
    sparse_nnz: int


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
    records: tuple[GromovIterationRecord, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["GromovIterationRecord", "GromovResult"]
