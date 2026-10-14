from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from hello_ot.types import SparseOTSolution
from hello_ot.variants._records import RecordMapping


@dataclass(frozen=True, eq=False)
class UOTIterationRecord(RecordMapping):
    """
    CN: 一次 UOT FCFW 外层迭代的当前状态与历史界记录。
    EN: Current-state and historical-bound record for one UOT FCFW outer iteration.
    """

    iteration: int
    wall_time: float
    primal_objective: float
    dual_objective: float
    relative_primal_dual_gap: float
    post_correction_dual_objective: float | None
    best_primal_objective: float
    best_primal_iteration: int
    best_dual_objective: float
    best_dual_iteration: int
    best_relative_primal_dual_gap: float
    fw_directional_gap: float
    atom_count: int
    nonzero_atom_count: int


@dataclass(frozen=True)
class UOTConvergenceBounds:
    """
    CN: 跨迭代累计的最佳 UOT primal upper bound 与 dual lower bound。
    EN: Best UOT primal upper bound and dual lower bound accumulated across iterations.
    """

    primal_objective: float
    primal_iteration: int
    dual_objective: float
    dual_iteration: int
    relative_primal_dual_gap: float


@dataclass(frozen=True)
class _UOTIterationSnapshot:
    """
    CN: 同一 FCFW 迭代产生的原子化 primal/dual 状态。
    EN: Atomic primal/dual state produced by one FCFW iteration.
    """

    iteration: int
    coupling: Any
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    transported_mass: float
    source_potential: np.ndarray
    target_potential: np.ndarray
    primal_objective: float
    dual_objective: float

    @classmethod
    def from_state_and_certificate(
        cls,
        *,
        iteration: int,
        state: Mapping[str, Any],
        certificate: Mapping[str, Any],
    ) -> _UOTIterationSnapshot:
        """
        CN: 复制同一轮的状态与 certificate，避免后续迭代改变已保存 snapshot。
        EN: Copy one iteration's state and certificate so later iterations cannot alter the saved snapshot.
        """
        return cls(
            iteration=int(iteration),
            coupling=certificate["coupling"].tocoo(copy=True),
            source_marginal=np.asarray(certificate["source_marginal"], dtype=np.float64).copy(),
            target_marginal=np.asarray(certificate["target_marginal"], dtype=np.float64).copy(),
            transported_mass=float(certificate["transported_mass"]),
            source_potential=np.asarray(state["source_potential"], dtype=np.float64).copy(),
            target_potential=np.asarray(state["target_potential"], dtype=np.float64).copy(),
            primal_objective=float(certificate["primal_objective"]),
            dual_objective=float(state["dual_objective"]),
        )


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
    best_bounds: UOTConvergenceBounds
    snapshot_iteration: int
    convergence_reason: Literal["historical_bounds", "fw_directional_gap", "max_iterations"]
    total_wall_time: float
    iterations: int
    converged: bool
    records: tuple[UOTIterationRecord, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["UOTConvergenceBounds", "UOTIterationRecord", "UnbalancedResult"]
