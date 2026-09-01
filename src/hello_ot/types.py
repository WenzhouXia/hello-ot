from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple, Union

import numpy as np

from .state import WarmStartState


ArrayLike = Any
DualPreparation = Literal["preserve", "source_from_target", "target_from_source"]
PointCloudCostType = Literal["l2^2", "l1", "l2", "linf"]


class HelloSolveError(RuntimeError):
    """
    CN: HELLO 求解失败；正常返回的 Result 始终表示完整求解结果。
    EN: HELLO solve failure; a returned Result always denotes a complete solve result.
    """


def _float32_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape={array.shape}.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty; got shape={array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _float64_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(size):
        raise ValueError(f"{name} must have length {size}; got {array.size}.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _mass_vector(value: Optional[Any], *, size: int, name: str) -> np.ndarray:
    if value is None:
        return np.full(int(size), 1.0 / float(size), dtype=np.float64)
    array = _float64_vector(value, size=size, name=name)
    if np.any(array < 0.0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and nonnegative.")
    total = float(array.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return array


@dataclass(frozen=True)
class Problem:
    """
    CN: HELLO 的正式平衡 OT 输入；四类代价统一接收两侧点云。
    EN: Formal balanced-OT input for HELLO; all four costs accept two point clouds.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    cost_type: PointCloudCostType
    source_mass: Optional[np.ndarray] = None
    target_mass: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        source = _float32_matrix(self.source_points, name="source_points")
        target = _float32_matrix(self.target_points, name="target_points")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must have the same feature dimension.")
        cost_type = str(self.cost_type).strip().lower()
        if cost_type not in {"l2^2", "l1", "l2", "linf"}:
            raise ValueError("cost_type must be one of: l2^2, l1, l2, linf")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "cost_type", cost_type)
        object.__setattr__(self, "source_mass", _mass_vector(self.source_mass, size=source.shape[0], name="source_mass"))
        object.__setattr__(self, "target_mass", _mass_vector(self.target_mass, size=target.shape[0], name="target_mass"))

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.source_points.shape[0]), int(self.target_points.shape[0])


@dataclass(frozen=True)
class DualPotentials:
    """
    CN: 可只提供一侧的对偶势；缺失侧由显式 preparation 策略处理。
    EN: Dual potentials with either side optional; explicit preparation handles a missing side.
    """

    source: Optional[ArrayLike] = None
    target: Optional[ArrayLike] = None


@dataclass(frozen=True)
class PreparedDual:
    """
    CN: 已具有双侧势、可直接用于 dual assignment 的对偶对；不隐含可行性。
    EN: Two-sided potentials ready for dual assignment; feasibility is not implied.
    """

    source: ArrayLike
    target: ArrayLike


@dataclass(frozen=True)
class SolveLPStats:
    wall_time: float
    backend_iterations: Optional[int]
    primal_feasibility: Optional[float]
    primal_dual_gap: Optional[float]
    termination_norm: Optional[Literal["l2", "linf"]] = None
    termination_reason: Optional[str] = None
    termination_primal_feasibility: Optional[float] = None
    termination_dual_feasibility: Optional[float] = None
    algorithm_dual_feasibility: Optional[float] = None


@dataclass(frozen=True)
class DualViolationStats:
    wall_time: float
    violations_found: int
    edges_added: int


@dataclass(frozen=True)
class BudgetedPruningStats:
    wall_time: float
    support_before: int
    support_after: int
    budget: int
    budget_exceeded: bool


@dataclass(frozen=True)
class ActiveSupportStats:
    before_lp: int
    after_dual_violation: int
    after_budgeted_pruning: int
    peak: int


@dataclass(frozen=True)
class ConvergenceStats:
    dual_feasibility: Optional[float]
    tolerance: float
    passed: bool
    relative_full_dual_feasibility: Optional[float] = None
    relative_linf_dual_feasibility: Optional[float] = None
    max_dual_violation: Optional[float] = None
    norm: Optional[Literal["l2", "linf"]] = None


@dataclass(frozen=True)
class IterationStats:
    index: int
    objective: float
    wall_time: float
    bookkeeping_time: float
    solve_lp: SolveLPStats
    dual_violation: DualViolationStats
    budgeted_pruning: BudgetedPruningStats
    support: ActiveSupportStats
    convergence: ConvergenceStats


@dataclass(frozen=True)
class InitializationStats:
    wall_time: float
    dual_propagation_time: float
    dual_assignment_time: float
    northwest_augmentation_time: float
    inherited_dual_side: Optional[Literal["source", "target"]]
    assignment_topk: int
    support_after_forward: int
    support_after_reverse: Optional[int]
    northwest_edges_added: int
    initial_active_support_size: int


@dataclass(frozen=True)
class InitializationResult:
    """
    CN: 单层 initialization 的运行时输出；三项均为已有对象的引用，不复制 support 或 dual 数组。
    EN: Runtime output of one level's initialization; all fields reference existing objects without copying support or dual arrays.
    """

    active_support: Any
    dual_warm_start: Optional[Any]
    statistics: Mapping[str, Any]


@dataclass(frozen=True)
class RefinementSummary:
    wall_time: float
    iterations: int
    final_objective: float
    final_active_support_size: int
    peak_active_support_size: int
    stop_reason: str
    converged: bool


@dataclass(frozen=True)
class RefinementResult:
    iterations: Tuple[IterationStats, ...]
    summary: RefinementSummary
    solution: Optional["SparseOTSolution"] = None


@dataclass(frozen=True)
class CoarsestSolveStats:
    wall_time: float
    objective: float
    support_size: int


LevelSolveResult = Union[CoarsestSolveStats, RefinementResult]


@dataclass(frozen=True)
class LevelResult:
    level_index: int
    kind: Literal["coarsest", "refined"]
    n_source: int
    n_target: int
    initialization: Optional[InitializationStats]
    solve: LevelSolveResult
    wall_time: float


@dataclass(frozen=True)
class SolveStageResult:
    levels: Tuple[LevelResult, ...]
    wall_time: float


@dataclass(frozen=True)
class PrewarmStats:
    """
    CN: 正式 HELLO 预热的紧凑结果；operations 只记录已触发的 kernel 路径。
    EN: Compact formal HELLO prewarm result; operations only names the exercised kernel paths.
    """

    wall_time: float
    operations: Tuple[str, ...]


@dataclass(frozen=True)
class SparseOTSolution:
    shape: Tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray

    def __post_init__(self) -> None:
        rows = np.ascontiguousarray(np.asarray(self.rows, dtype=np.int64).reshape(-1))
        cols = np.ascontiguousarray(np.asarray(self.cols, dtype=np.int64).reshape(-1))
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64).reshape(-1))
        source_dual = np.ascontiguousarray(np.asarray(self.source_dual, dtype=np.float64).reshape(-1))
        target_dual = np.ascontiguousarray(np.asarray(self.target_dual, dtype=np.float64).reshape(-1))
        if not (rows.size == cols.size == values.size):
            raise ValueError("rows, cols, and values must have equal lengths")
        positive = values > np.float64(1e-12)
        rows = np.ascontiguousarray(rows[positive])
        cols = np.ascontiguousarray(cols[positive])
        values = np.ascontiguousarray(values[positive])
        if source_dual.size != int(self.shape[0]) or target_dual.size != int(self.shape[1]):
            raise ValueError("dual lengths must match solution shape")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_dual", source_dual)
        object.__setattr__(self, "target_dual", target_dual)

    def to_sparse_matrix(self) -> Any:
        from scipy import sparse

        return sparse.coo_matrix((self.values, (self.rows, self.cols)), shape=self.shape)

    def to_warm_start_state(self) -> WarmStartState:
        dual_uv = np.concatenate([self.source_dual, self.target_dual]).astype(np.float64, copy=False)
        return WarmStartState(
            rows=self.rows.astype(np.int32, copy=True),
            cols=self.cols.astype(np.int32, copy=True),
            x_prev=self.values.astype(np.float64, copy=True),
            dual_uv=np.ascontiguousarray(dual_uv, dtype=np.float64),
            n_source=int(self.shape[0]),
            n_target=int(self.shape[1]),
            northwest_positions=None,
        )


@dataclass(frozen=True)
class TraceResult:
    payload: Mapping[str, Any]

    def export(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dict(self.payload)), encoding="utf-8")


@dataclass(frozen=True)
class Result:
    objective: float
    solution: SparseOTSolution
    solve_stage: SolveStageResult
    total_wall_time: float
    peak_gpu_memory_mib: Optional[float] = None
    trace: Optional[TraceResult] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActiveSupportStats",
    "BudgetedPruningStats",
    "CoarsestSolveStats",
    "ConvergenceStats",
    "DualPotentials",
    "DualPreparation",
    "DualViolationStats",
    "Result",
    "HelloSolveError",
    "InitializationStats",
    "InitializationResult",
    "IterationStats",
    "LevelResult",
    "PreparedDual",
    "PrewarmStats",
    "Problem",
    "RefinementResult",
    "RefinementSummary",
    "SolveLPStats",
    "SolveStageResult",
    "SparseOTSolution",
    "TraceResult",
]
