from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LinearOTResult:
    """
    CN: 统一保存 solver 输出，同时允许稠密、稀疏和 permutation transport。
    EN: Store solver output uniformly while allowing dense, sparse, and permutation transports.
    """

    method: str
    solver_objective: float
    runtime_sec: float
    transport_kind: str
    transport: Any
    converged: bool
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportEvaluation:
    """
    CN: 在原始 OT 问题上统一重算的 objective、边际残差与 representation 元数据。
    EN: Canonically recomputed objective, marginal residual, and representation metadata.
    """

    objective: float
    primal_feasibility: float
    primal_l2_abs_error: float
    row_marginal_l2_error: float
    col_marginal_l2_error: float
    transport_mass_error: float
    transport_kind: str
    transport_shape: tuple[int, int]
    transport_nnz: int
