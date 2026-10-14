from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional, Protocol

import numpy as np

from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.config import SolverRuntimeConfig, StoppingNormPolicy, level_cost_precision, level_stopping_norm
from hello_ot.cost import HelloCostContext
from hello_ot.hierarchy.construction import HierarchyNodeRange
from hello_ot.hierarchy.records import HierarchyLevelRecord
from hello_ot.state import GPUWarmStartState, WarmStartState


HierarchyState = WarmStartState | GPUWarmStartState


@dataclass
class DualAssignmentResult:
    state: HierarchyState
    topk_stats: Dict[str, Any]
    augment_profile: Dict[str, Any]
    dual_completion_profile: Dict[str, Any]
    forward_topk_stats: Dict[str, Any]
    support_after_forward: int
    reverse_topk_stats: Dict[str, Any]
    support_after_reverse: Optional[int]


@dataclass
class HierarchyExecution:
    """
    CN: hierarchy 算法与 cost-stage 转换共享的数据及运行设施。
    EN: Data and runtime facilities shared by the hierarchy algorithm and cost-stage transitions.
    """

    cost_context: HelloCostContext
    source_F: np.ndarray
    target_G: np.ndarray
    source_cost: np.ndarray
    target_cost: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray
    config: SolverRuntimeConfig
    assignment_topk: int
    split_count: int
    dual_feasibility_tol: float
    stopping_norm: StoppingNormPolicy
    tracer: Optional[_ChromeTraceCollector]
    pricing_index_pool: Optional[Any]
    memory_recorder: Optional[Any]
    log: bool
    perturbation: Any = None

    def context_for(self, node: HierarchyNodeRange) -> HelloCostContext:
        """
        CN: 获取当前层成本上下文，保留扰动的全局边标识。
        EN: Get this level's cost context while preserving global perturbation identities.
        """
        context = self.cost_context.sliced(
            slice(node.source_start, node.source_stop),
            slice(node.target_start, node.target_stop),
        )
        return replace(
            context,
            compute_precision=level_cost_precision(
                self.config.cost_precision, level_index=int(node.depth)
            ),
        )

    def new_reentry_detector(self) -> Any:
        """
        CN: 为当前原成本层建立独立检测器。
        EN: Create an independent detector for the current original-cost level.
        """
        return None if self.perturbation is None else self.perturbation.new_detector()

    def stopping_norm_for(self, node: HierarchyNodeRange) -> Literal["l2", "linf"]:
        """
        CN: 粗层固定使用 L2；finest_linf 只将最精细层切换到 L-infinity。
        EN: Keep coarse levels on L2; finest_linf switches only the finest level to L-infinity.
        """
        return level_stopping_norm(self.stopping_norm, level_index=int(node.depth))


@dataclass
class InitializedHierarchyLevel:
    node: HierarchyNodeRange
    arrays: Dict[str, np.ndarray]
    state: HierarchyState
    split_axis: str
    trace_args: Dict[str, Any]
    started_at: float
    initialization_elapsed: float
    support_before_solve: int
    stitch_profile: Dict[str, Any]
    assignment: DualAssignmentResult
    refinement: Any
    refine_span: Any


class HierarchyLevelFinalizer(Protocol):
    """
    CN: cost-stage 转换完成 hierarchy 层时调用的中立协议。
    EN: Neutral protocol used by cost-stage transitions to finalize a hierarchy level.
    """

    def __call__(
        self,
        initialized: InitializedHierarchyLevel,
        *,
        levels: list[HierarchyLevelRecord],
        execution: HierarchyExecution,
    ) -> tuple[HierarchyState, Dict[str, Any]]: ...


__all__ = [
    "DualAssignmentResult",
    "HierarchyExecution",
    "HierarchyLevelFinalizer",
    "HierarchyState",
    "InitializedHierarchyLevel",
]
