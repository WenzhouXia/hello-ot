from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from hello_ot._internal.runtime_context import record_solve_event

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from hello_ot.refinement.dual_violation import fmt_size_ratio
from hello_ot.refinement.support_budget import (
    POSITIVE_SUPPORT_TOLERANCE,
    active_support_budget,
)
from hello_ot.utilities import trace_span

logger = logging.getLogger(__name__)


def _select_zero_flow_removals(
    *,
    removable_pool: Any,
    reduced_costs: Any,
    num_to_remove: int,
) -> np.ndarray:
    """
    CN: 用同一规则按 reduced cost 为 NumPy 或 Torch active support 选择旧零流边。
    EN: Select old zero-flow edges by reduced cost with one rule for NumPy and Torch active supports.
    """
    torch_backend = bool(TORCH_AVAILABLE and torch.is_tensor(removable_pool))
    if torch_backend:
        indices = torch.nonzero(removable_pool, as_tuple=False).view(-1)
        if int(indices.numel()) < int(num_to_remove):
            raise RuntimeError("insufficient removable old zero-flow edges")
        order = torch.argsort(
            reduced_costs.index_select(0, indices),
            descending=True,
            stable=True,
        )
        selected = indices.index_select(0, order)[:int(num_to_remove)]
        return selected.detach().cpu().numpy().astype(np.int32, copy=False)

    indices = np.flatnonzero(removable_pool)
    if int(indices.size) < int(num_to_remove):
        raise RuntimeError("insufficient removable old zero-flow edges")
    order = np.argsort(
        -np.asarray(reduced_costs)[indices],
        kind="stable",
    )
    return np.asarray(indices[order[:int(num_to_remove)]], dtype=np.int32)


class CleaningStrategy(ABC):
    """
    CN: active support 剪枝策略的抽象基类。
    EN: Abstract base class for active-support pruning strategies.
    """

    def __init__(self, threshold_factor: float = 0.0):
        """
        CN: 初始化 active support 剪枝策略。threshold_factor 表示触发剪枝的支撑规模系数。
        EN: Initialize an active-support pruning strategy. threshold_factor controls the support-size trigger.
        """
        self.threshold = max(0.0, threshold_factor)

    @abstractmethod
    def select_indices_to_remove(self, current_inc: Dict[str, Any], num_to_remove: int, **kwargs) -> np.ndarray:
        """
        CN: 选择要从 active support 中删除的变量下标。
        EN: Select variable indices to remove from the active support.
        """
        pass

    def needs_creation_iteration(self) -> bool:
        """
        CN: 返回该策略是否需要跟踪变量的创建轮次。
        EN: Return whether this strategy needs variable-creation iteration tracking.
        """
        return False


class NoCleaning(CleaningStrategy):
    """
    CN: 不执行 budgeted pruning 的策略。
    EN: Strategy that performs no budgeted pruning.
    """

    def __init__(self):
        # CN: NoCleaning 使用零阈值，并通过返回空下标集禁用剪枝。
        # EN: NoCleaning uses a zero threshold and disables pruning by returning no indices.
        super().__init__(threshold_factor=0.0)

    def select_indices_to_remove(self, current_inc: Dict[str, Any], num_to_remove: int, **kwargs) -> np.ndarray:
        return np.empty(0, dtype=np.int32)


class DualGapCleaning(CleaningStrategy):
    """
    CN: 删除 reduced cost 最大且 primal 流量足够小的变量。
    EN: Remove variables with the largest reduced costs whose primal flow is small enough.
    """

    def __init__(
        self,
        threshold_factor: float = 10.0,
    ):
        super().__init__(threshold_factor=threshold_factor)

    def needs_creation_iteration(self) -> bool:
        return True

    def _select_protect_new_indices(
        self,
        current_active_support: Dict[str, Any],
        num_to_remove: int,
        duals: Any,
        n_source: int,
        current_inner_iter: Optional[int],
        *,
        log_prefix: str = "DualGapCleaning",
    ) -> np.ndarray:
        """
        CN: 在 protect_new 模式下选择待删除下标，只从旧变量中删，保护当前轮新加入变量。
        EN: Select removal indices for protect_new mode, pruning only old variables and preserving current-iteration additions.
        """
        # CN: protect_new 依赖 creation_iteration 与当前 inner_iter 对齐，否则无法区分本轮新边。
        # EN: protect_new needs current_inner_iter to distinguish current-iteration edges from older edges.
        if current_inner_iter is None:
            logger.info(
                "[%s] protect_new requires current_inner_iter. Skipping cleaning.",
                log_prefix,
            )
            return np.empty(0, dtype=np.int32)

        values = current_active_support.pruning_values(duals, n_source)
        if values is None:
            logger.info("[%s Error] Row or column index out of bounds for dual variables.", log_prefix)
            return np.empty(0, dtype=np.int32)
        abs_x, creation_iters, reduced_costs = values
        is_new = creation_iters == int(current_inner_iter)
        is_structural = creation_iters == -1
        old_pool = (~is_structural) & (~is_new)
        is_positive = abs_x > POSITIVE_SUPPORT_TOLERANCE
        removable_pool = old_pool & (~is_positive)

        if TORCH_AVAILABLE and torch.is_tensor(removable_pool):
            count = lambda mask: int(torch.count_nonzero(mask).item())
            support_size = int(removable_pool.numel())
        else:
            count = lambda mask: int(np.count_nonzero(mask))
            support_size = int(np.asarray(removable_pool).size)
        removable_count = count(removable_pool)
        if removable_count < int(num_to_remove):
            raise RuntimeError(
                "HELLO budgeted pruning invariant violated: "
                f"required_removals={int(num_to_remove)}, "
                f"removable_old_zero={removable_count}, "
                f"protected_positive={count(is_positive)}, "
                f"protected_structural={count(is_structural)}, "
                f"protected_new={count(is_new)}, "
                f"support_size={support_size}, "
                f"budget={support_size - int(num_to_remove)}"
            )
        return _select_zero_flow_removals(
            removable_pool=removable_pool,
            reduced_costs=reduced_costs,
            num_to_remove=num_to_remove,
        )

    def select_indices_to_remove(self, current_active_support: Dict[str, Any], num_to_remove: int, **kwargs) -> np.ndarray:
        """
        CN: 根据当前 active support 的 primal/dual 信息选择 cleaning 要删除的变量下标。
        EN: Select active-support variable indices to prune using the current primal and dual information.
        """
        # CN: cleaning 需要 dual 势和 source 数量来计算 c_ij - u_i - v_j。
        # EN: Cleaning needs dual potentials and n_source to compute c_ij - u_i - v_j.
        duals = kwargs.get('duals')
        n_source = kwargs.get('n_source')

        if duals is None or n_source is None:
            logger.info(
                "[DualGapCleaning Warning] 'duals' or 'n_source' not provided in kwargs. Skipping cleaning.")
            return np.empty(0, dtype=np.int32)
        if num_to_remove <= 0:
            return np.empty(0, dtype=np.int32)

        return self._select_protect_new_indices(
            current_active_support,
            num_to_remove,
            duals,
            n_source,
            kwargs.get("current_inner_iter"),
        )



def apply_budgeted_pruning(
    solver,
    duals: np.ndarray,
    n_s: int,
    n_t: int,
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.finalize_iteration.cleaning",
    current_inner_iter: Optional[int] = None,
    removed_keys_out: Optional[list] = None,
) -> np.ndarray:
    cleaning_strategy = solver.cleaning_strategy
    if not isinstance(cleaning_strategy, DualGapCleaning):
        raise TypeError("HELLO budgeted pruning requires DualGapCleaning")

    threshold = cleaning_strategy.threshold
    if threshold <= 0:
        return np.empty(0, dtype=np.int32)

    max_size = active_support_budget(threshold, n_s, n_t)
    curr_size = solver.active_support.size
    if curr_size <= max_size:
        return np.empty(0, dtype=np.int32)

    with trace_span(
        trace_collector,
        trace_prefix,
        "select_indices_to_remove",
        args={"curr_size": int(curr_size), "max_size": int(max_size)},
    ):
        to_remove = DualGapCleaning.select_indices_to_remove(
            cleaning_strategy,
            solver.active_support,
            curr_size - max_size,
            duals=duals,
            n_source=n_s,
            n_target=n_t,
            current_inner_iter=current_inner_iter,
        )
    if len(to_remove) <= 0:
        return np.empty(0, dtype=np.int32)

    if removed_keys_out is not None:
        removed_keys_out.append(solver.active_support.keys_at_positions(to_remove))

    if logger.isEnabledFor(logging.INFO):
        size_after = curr_size - len(to_remove)
        logger.info(
            "  [Clean] removed=%s, active_size=%s -> %s",
            fmt_size_ratio(len(to_remove), n_s, n_t),
            fmt_size_ratio(curr_size, n_s, n_t),
            fmt_size_ratio(size_after, n_s, n_t),
        )
    with trace_span(
        trace_collector,
        trace_prefix,
        "prune",
        args={"removed": int(len(to_remove))},
    ):
        solver.active_support.remove_positions(to_remove)
    record_solve_event(
        "budgeted_pruning",
        iteration=None if current_inner_iter is None else int(current_inner_iter),
        removed=np.asarray(to_remove, dtype=np.int32),
        rows=solver.active_support.rows,
        cols=solver.active_support.cols,
    )
    return np.asarray(to_remove, dtype=np.int32)


__all__ = ["CleaningStrategy", "DualGapCleaning", "NoCleaning", "apply_budgeted_pruning"]
