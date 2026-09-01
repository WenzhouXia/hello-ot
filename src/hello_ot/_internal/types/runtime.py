from __future__ import annotations

from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.hierarchical_solver import HierarchicalOTSolver


@dataclass
class SolveSpec:
    max_inner_iter: int
    convergence_criterion: str
    objective_plateau_iters: int
    tolerance: Dict[str, float]
    cost_type: str
    use_bfs_skeleton: bool


@dataclass
class ProblemDef:
    mode: str
    solver: HierarchicalOTSolver
    solve_spec: SolveSpec
    backend: str = "hierarchical"
    trace_collector: Any = None
    trace_prefix: str = "solve_ot"
    profiling_options: Any = None
    printing_options: Any = None
    profiler: Any = None
    reporter: Any = None
    inner_iteration_callback: Any = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_inner_iter(self) -> int:
        return self.solve_spec.max_inner_iter

    @property
    def convergence_criterion(self) -> str:
        return self.solve_spec.convergence_criterion

    @property
    def objective_plateau_iters(self) -> int:
        return self.solve_spec.objective_plateau_iters

    @property
    def tolerance(self) -> Dict[str, float]:
        return self.solve_spec.tolerance

    @property
    def cost_type(self) -> str:
        return self.solve_spec.cost_type

    @property
    def use_bfs_skeleton(self) -> bool:
        return self.solve_spec.use_bfs_skeleton


@dataclass
class AlgorithmState:
    run_state: Optional[Dict[str, Any]] = None
    level_indices: List[int] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None


@dataclass
class LevelData(MutableMapping[str, Any]):
    level_idx: int
    is_coarsest: bool
    t_level_start: float
    current_iter: int = 0
    completed_iters: int = 0
    level_lp_time: float = 0.0
    level_pricing_time: float = 0.0
    curr_active_size: Optional[int] = None
    level_s: Any = None
    level_t: Any = None
    cost_type: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    _FIELD_NAMES = {
        "level_idx",
        "is_coarsest",
        "t_level_start",
        "current_iter",
        "completed_iters",
        "level_lp_time",
        "level_pricing_time",
        "curr_active_size",
        "level_s",
        "level_t",
        "cost_type",
    }

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "LevelData":
        copied = dict(data)
        kwargs = {
            "level_idx": int(copied.pop("level_idx")),
            "is_coarsest": bool(copied.pop("is_coarsest")),
            "t_level_start": float(copied.pop("t_level_start")),
            "current_iter": int(copied.pop("current_iter", copied.pop("inner_iter", 0))),
            "completed_iters": int(copied.pop("completed_iters", copied.pop("iters_done", 0))),
            "level_lp_time": float(copied.pop("level_lp_time", 0.0)),
            "level_pricing_time": float(copied.pop("level_pricing_time", 0.0)),
            "curr_active_size": copied.pop("curr_active_size", None),
            "level_s": copied.pop("level_s", None),
            "level_t": copied.pop("level_t", None),
            "cost_type": copied.pop("cost_type", None),
            "extras": copied,
        }
        return cls(**kwargs)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "level_idx": self.level_idx,
            "is_coarsest": self.is_coarsest,
            "t_level_start": self.t_level_start,
            "current_iter": self.current_iter,
            "completed_iters": self.completed_iters,
            "level_lp_time": self.level_lp_time,
            "level_pricing_time": self.level_pricing_time,
            "curr_active_size": self.curr_active_size,
            "level_s": self.level_s,
            "level_t": self.level_t,
            "cost_type": self.cost_type,
        }
        out.update(self.extras)
        return out

    def __getitem__(self, key: str) -> Any:
        if key in self._FIELD_NAMES:
            return getattr(self, key)
        return self.extras[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._FIELD_NAMES:
            setattr(self, key, value)
            return
        self.extras[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._FIELD_NAMES:
            raise KeyError(f"Cannot delete required LevelData field: {key}")
        del self.extras[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


@dataclass
class LevelState(MutableMapping[str, Any]):
    level_index: int
    max_inner_iter: int
    is_coarsest: bool
    t_level_start: float
    current_iter: int = 0
    completed_iters: int = 0
    level_lp_time: float = 0.0
    level_pricing_time: float = 0.0
    curr_active_size: Optional[int] = None
    level_s: Any = None
    level_t: Any = None
    cost_type: Optional[str] = None
    mode_state: Dict[str, Any] = field(default_factory=dict)

    _FIELD_NAMES = {
        "level_idx",
        "level_index",
        "is_coarsest",
        "t_level_start",
        "current_iter",
        "completed_iters",
        "level_lp_time",
        "level_pricing_time",
        "curr_active_size",
        "level_s",
        "level_t",
        "cost_type",
    }

    @classmethod
    def from_legacy_data(
        cls,
        *,
        level_index: int,
        max_inner_iter: int,
        data: Dict[str, Any] | LevelData,
    ) -> "LevelState":
        raw = data.as_dict() if isinstance(data, LevelData) else dict(data)
        level_data = LevelData.from_mapping(raw)
        return cls(
            level_index=level_index,
            max_inner_iter=max_inner_iter,
            is_coarsest=level_data.is_coarsest,
            t_level_start=level_data.t_level_start,
            current_iter=level_data.current_iter,
            completed_iters=level_data.completed_iters,
            level_lp_time=level_data.level_lp_time,
            level_pricing_time=level_data.level_pricing_time,
            curr_active_size=level_data.curr_active_size,
            level_s=level_data.level_s,
            level_t=level_data.level_t,
            cost_type=level_data.cost_type,
            mode_state=dict(level_data.extras),
        )

    @property
    def level_idx(self) -> int:
        return self.level_index

    @level_idx.setter
    def level_idx(self, value: int) -> None:
        self.level_index = int(value)

    @property
    def data(self) -> "LevelState":
        return self

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "level_idx": self.level_index,
            "is_coarsest": self.is_coarsest,
            "t_level_start": self.t_level_start,
            "current_iter": self.current_iter,
            "completed_iters": self.completed_iters,
            "level_lp_time": self.level_lp_time,
            "level_pricing_time": self.level_pricing_time,
            "curr_active_size": self.curr_active_size,
            "level_s": self.level_s,
            "level_t": self.level_t,
            "cost_type": self.cost_type,
        }
        out.update(self.mode_state)
        return out

    def __getitem__(self, key: str) -> Any:
        if key == "level_idx":
            return self.level_index
        if key in self._FIELD_NAMES:
            return getattr(self, key)
        return self.mode_state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "level_idx":
            self.level_index = int(value)
            return
        if key in self._FIELD_NAMES:
            setattr(self, key, value)
            return
        self.mode_state[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._FIELD_NAMES:
            raise KeyError(f"Cannot delete required LevelState field: {key}")
        del self.mode_state[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


@dataclass
class StepResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    objective: Optional[float] = None


@dataclass
class LPSummary:
    level_index: int
    inner_iter: int
    time_seconds: Optional[float]
    iterations: Any
    num_variables: Optional[int]
    num_constraints: Optional[int]
    primal_objective: Optional[float]
    dual_objective: Optional[float]
    primal_residual: Optional[float]
    dual_residual: Optional[float]
    gap: Optional[float]


@dataclass
class PricingSummary:
    level_index: int
    inner_iter: int
    time_seconds: Optional[float]
    active_before: int
    found: int
    added: int


@dataclass
class ConvergenceSummary:
    level_index: int
    inner_iter: int
    signed_rel_obj_change: Optional[float]
    converged: bool
    criterion: str
    plateau_counter: Any
    required_plateau: Any
    tolerance: Optional[float]
    dual_feasibility: Optional[float] = None
    dual_feasibility_tol: Optional[float] = None
    dual_feasibility_passed: Any = None


@dataclass
class FinalizeSummary:
    level_index: int
    inner_iter: int
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class IterationSummary:
    level_index: int
    inner_iter: int
    total_seconds: float
    solve_seconds: float
    finalize_seconds: float
    other_seconds: float


@dataclass
class LevelSummary:
    level_index: int
    total_seconds: float
    iterations: int
    lp_seconds: float
    pricing_seconds: float
    other_seconds: float
    active_size: Optional[int]
