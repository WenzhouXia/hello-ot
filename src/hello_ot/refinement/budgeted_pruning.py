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
from hello_ot.utilities import trace_span

logger = logging.getLogger(__name__)


class CleaningStrategy(ABC):
    """Abstract base class for active set cleaning strategies."""

    # Add a small tolerance for floating point comparisons
    _TOL = 1e-10

    def __init__(self, threshold_factor: float = 0.0):
        """
        Args:
            threshold_factor: 清理阈值系数 (如 5.0 表示 active set > 5 * (N+M) 时触发清理)
        """
        self.threshold = max(0.0, threshold_factor)

    @abstractmethod
    def select_indices_to_remove(self, current_inc: Dict[str, Any], num_to_remove: int, **kwargs) -> np.ndarray:
        """
        Selects indices of variables to remove from the active set.

        Args:
            current_inc: The current active set dictionary containing at least:
                'rows', 'cols', 'c_vec', 'x_prev', 'pos_map'.
                May also contain 'creation_iteration' if needed by the strategy.
            num_to_remove: The target number of variables to remove.
            **kwargs: Additional context potentially needed by specific strategies
                      (e.g., 'duals', 'n_source').

        Returns:
            A NumPy array of absolute indices within the current_inc arrays
            corresponding to the variables selected for removal. The strategy
            might return fewer indices than num_to_remove if insufficient
            candidates are found.
        """
        pass

    def needs_creation_iteration(self) -> bool:
        """Indicates if this strategy requires tracking variable creation time."""
        return False


class NoCleaning(CleaningStrategy):
    """A strategy that performs no cleaning."""

    def __init__(self):
        # NoCleaning 的阈值通常设为 0 或无穷大，这里设为 0 且 select 返回空即可
        super().__init__(threshold_factor=0.0)

    def select_indices_to_remove(self, current_inc: Dict[str, Any], num_to_remove: int, **kwargs) -> np.ndarray:
        return np.empty(0, dtype=np.int32)


class DualGapCleaning(CleaningStrategy):
    """
    CN: 删除 dual gap 最大且 primal 流量足够小的变量。
    EN: Remove variables with the largest dual gap whose primal flow is small enough.
    """

    def __init__(
        self,
        threshold_factor: float = 10.0,
        primal_tol: float = CleaningStrategy._TOL,
    ):
        super().__init__(threshold_factor=threshold_factor)
        self.primal_tol = max(0.0, float(primal_tol))

    def needs_creation_iteration(self) -> bool:
        return True

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return bool(TORCH_AVAILABLE and torch.is_tensor(value))

    @staticmethod
    def _to_numpy_1d(value: Any, dtype: Any) -> np.ndarray:
        if TORCH_AVAILABLE and torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
        return np.asarray(value, dtype=dtype).reshape(-1)

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

        if getattr(current_active_support, "is_torch_backend", False):
            # CN: CUDA active support 路径保持张量在原 device 上，避免 cleaning 时来回拷贝。
            # EN: Keep tensors on the active-support device in the CUDA path to avoid cleaning-time transfers.
            costs = current_active_support.c_vec
            x_prev = current_active_support.x_prev
            creation_iters = current_active_support.creation_iteration
            duals_t = duals if self._is_torch_tensor(duals) else torch.as_tensor(
                duals,
                dtype=costs.dtype,
                device=costs.device,
            )
            if duals_t.device != costs.device:
                duals_t = duals_t.to(device=costs.device)
            if duals_t.dtype != costs.dtype:
                duals_t = duals_t.to(dtype=costs.dtype)

            # CN: dual 向量前 n_source 项是 source 势 u，后半部分是 target 势 v。
            # EN: The first n_source dual entries are source potentials u; the remainder are target potentials v.
            n_source_int = int(n_source)
            u = duals_t[:n_source_int]
            v = duals_t[n_source_int:]
            # CN: 当前轮新加入的边不进入删除池；creation_iteration == -1 的结构性边也不删。
            # EN: Current-iteration edges are protected; structural edges marked -1 are also excluded.
            abs_x = torch.abs(x_prev)
            is_new = creation_iters == int(current_inner_iter)
            old_pool = (creation_iters != -1) & (~is_new)
            old_zero_indices = torch.nonzero(old_pool & (abs_x < self.primal_tol), as_tuple=False).view(-1)
            need = int(num_to_remove)
            selected_parts = []

            def _dual_gaps_for(indices: Any) -> Any:
                # CN: 对候选边计算 reduced cost / dual gap: c_ij - u_i - v_j。
                # EN: Compute reduced cost / dual gap for candidates: c_ij - u_i - v_j.
                row_idx = current_active_support.rows.index_select(0, indices).to(dtype=torch.long)
                col_idx = current_active_support.cols.index_select(0, indices).to(dtype=torch.long)
                if bool(torch.any(row_idx >= n_source_int)) or bool(torch.any(col_idx >= int(v.numel()))):
                    logger.info(
                        "[%s Error] Row or column index out of bounds for dual variables.",
                        log_prefix,
                    )
                    return None
                gap = costs.index_select(0, indices) - (u.index_select(0, row_idx) + v.index_select(0, col_idx))
                return torch.where(
                    torch.isfinite(gap),
                    gap,
                    torch.full_like(gap, -torch.inf),
                )

            # CN: 第一阶段优先删除旧的零流边，并在其中优先删 dual gap 最大的边。
            # EN: First remove old zero-flow edges, prioritizing the largest dual gaps.
            if int(old_zero_indices.numel()) > 0 and need > 0:
                take = min(need, int(old_zero_indices.numel()))
                old_zero_gaps = _dual_gaps_for(old_zero_indices)
                if old_zero_gaps is None:
                    return np.empty(0, dtype=np.int32)
                if take < int(old_zero_indices.numel()):
                    top_pos = torch.topk(
                        old_zero_gaps,
                        k=int(take),
                        largest=True,
                        sorted=True,
                    ).indices
                else:
                    top_pos = torch.argsort(
                        old_zero_gaps,
                        descending=True,
                        stable=True,
                    )
                selected_parts.append(old_zero_indices.index_select(0, top_pos[:take]))
                need -= take

            # CN: 如果零流旧边不够，再扩展到旧的非零流边，按流量小优先、gap 大打破并列。
            # EN: If old zero-flow edges are insufficient, expand to old nonzero edges by small flow, then larger gap.
            if need > 0:
                selected_mask = torch.zeros(int(costs.numel()), dtype=torch.bool, device=costs.device)
                if selected_parts:
                    selected_mask[torch.cat(selected_parts)] = True
                old_indices = torch.nonzero(old_pool & (~selected_mask), as_tuple=False).view(-1)
                if int(old_indices.numel()) > 0:
                    take = min(need, int(old_indices.numel()))
                    old_gaps = _dual_gaps_for(old_indices)
                    if old_gaps is None:
                        return np.empty(0, dtype=np.int32)
                    old_abs = abs_x.index_select(0, old_indices)
                    # CN: 先按 gap 降序稳定排序，再按 primal 升序稳定排序，等价于 numpy lexsort((-gap, abs_x))。
                    # EN: Stable sort by gap descending, then primal ascending, matching numpy lexsort((-gap, abs_x)).
                    gap_order = torch.argsort(-old_gaps, stable=True)
                    old_by_gap = old_indices.index_select(0, gap_order)
                    primal_order = torch.argsort(old_abs.index_select(0, gap_order), stable=True)
                    selected_parts.append(old_by_gap.index_select(0, primal_order[:take]))
                    need -= take

            if not selected_parts:
                logger.info("[%s] No old variables found to remove.", log_prefix)
                return np.empty(0, dtype=np.int32)

            # CN: 返回的是 prune 前 active support 内的绝对位置，而不是 row/col key。
            # EN: Return absolute positions in the pre-prune active support, not row/col keys.
            selected_t = torch.cat(selected_parts).to(dtype=torch.int32)
            if need > 0:
                logger.info(
                    "[%s] Selected %d old variables; old pool is short by %d, so current new variables are preserved.",
                    log_prefix,
                    int(selected_t.numel()),
                    int(need),
                )
            else:
                logger.info(
                    "[%s] Selected %d old variables with protect_new cleaning.",
                    log_prefix,
                    int(selected_t.numel()),
                )
            return selected_t.detach().cpu().numpy().astype(np.int32, copy=False)

        # CN: NumPy 路径使用同一套选择规则，只是先把 active support 视图规整为 1D 数组。
        # EN: The NumPy path applies the same selection rule after normalizing active-support views to 1D arrays.
        rows = self._to_numpy_1d(current_active_support.rows, np.int64)
        cols = self._to_numpy_1d(current_active_support.cols, np.int64)
        costs = self._to_numpy_1d(current_active_support.c_vec, np.float64)
        x_prev = self._to_numpy_1d(current_active_support.x_prev, np.float64)
        creation_iters = self._to_numpy_1d(current_active_support.creation_iteration, np.int64)
        duals_np = self._to_numpy_1d(duals, np.float64)
        n_source_int = int(n_source)
        u = duals_np[:n_source_int]
        v = duals_np[n_source_int:]

        if np.any(rows >= n_source_int) or np.any(cols >= len(v)):
            logger.info(
                "[%s Error] Row or column index out of bounds for dual variables.",
                log_prefix,
            )
            return np.empty(0, dtype=np.int32)

        # CN: 对全部候选预先计算 dual gap，非有限值降为 -inf，避免被优先删除。
        # EN: Precompute dual gaps for all candidates and demote non-finite values to -inf.
        dual_gaps = costs - (u[rows] + v[cols])
        dual_gaps = np.where(np.isfinite(dual_gaps), dual_gaps, -np.inf)
        abs_x = np.abs(x_prev)
        is_new = creation_iters == int(current_inner_iter)
        old_pool = (creation_iters != -1) & (~is_new)
        old_zero_indices = np.flatnonzero(old_pool & (abs_x < self.primal_tol))
        need = int(num_to_remove)
        selected_parts = []

        # CN: 第一阶段：旧零流边按 dual gap 降序删除。
        # EN: Stage 1: remove old zero-flow edges in descending dual-gap order.
        if old_zero_indices.size > 0 and need > 0:
            zero_order = np.argsort(-dual_gaps[old_zero_indices], kind="stable")
            take = min(need, int(old_zero_indices.size))
            selected_parts.append(old_zero_indices[zero_order[:take]])
            need -= take

        # CN: 第二阶段：如果还需删除，则在旧边中按 primal 小优先、gap 大打破并列。
        # EN: Stage 2: if more removals are needed, prefer smaller primal values and break ties by larger gaps.
        if need > 0:
            selected_mask = np.zeros(rows.size, dtype=bool)
            if selected_parts:
                selected_mask[np.concatenate(selected_parts)] = True
            old_indices = np.flatnonzero(old_pool & (~selected_mask))
            if old_indices.size > 0:
                # CN: 旧零流候选不够时，按 primal 流量从小到大扩展旧边；同流量时删 dual gap 更大的边。
                # EN: If old zero-flow candidates are insufficient, expand old edges by increasing primal flow; ties prefer larger dual gaps.
                old_order = np.lexsort((-dual_gaps[old_indices], abs_x[old_indices]))
                take = min(need, int(old_indices.size))
                selected_parts.append(old_indices[old_order[:take]])
                need -= take

        if not selected_parts:
            logger.info("[%s] No old variables found to remove.", log_prefix)
            return np.empty(0, dtype=np.int32)

        selected = np.concatenate(selected_parts).astype(np.int32, copy=False)
        if need > 0:
            logger.info(
                "[%s] Selected %d old variables; old pool is short by %d, so current new variables are preserved.",
                log_prefix,
                int(selected.size),
                int(need),
            )
        else:
            logger.info(
                "[%s] Selected %d old variables with protect_new cleaning.",
                log_prefix,
                int(selected.size),
            )
        return selected

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

    min_supp = n_s + n_t
    max_size = int(threshold * min_supp)
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
        from .reentry import edge_keys

        support = solver.active_support
        positions = to_remove
        if torch.is_tensor(support.rows):
            positions = torch.as_tensor(to_remove, dtype=torch.long, device=support.rows.device)
        removed_keys_out.append(edge_keys(support.rows[positions], support.cols[positions], n_t))

    if logger.isEnabledFor(logging.INFO):
        size_after = curr_size - len(to_remove)
        logger.info(
            "  [Clean] removed=%s, active_size=%s -> %s",
            fmt_size_ratio(len(to_remove), n_s, n_t),
            fmt_size_ratio(curr_size, n_s, n_t),
            fmt_size_ratio(size_after, n_s, n_t),
        )
    if getattr(solver.active_support, "is_torch_backend", False):
        mask_backend = torch.ones(
            int(curr_size),
            dtype=torch.bool,
            device=solver.active_support.device,
        )
        remove_t = torch.as_tensor(
            np.asarray(to_remove, dtype=np.int64),
            dtype=torch.long,
            device=solver.active_support.device,
        )
        mask_backend[remove_t] = False
        with trace_span(
            trace_collector,
            trace_prefix,
            "prune",
            args={"removed": int(len(to_remove))},
        ):
            solver.active_support.prune(mask_backend)
    else:
        mask = np.ones(curr_size, dtype=bool)
        mask[to_remove] = False
        with trace_span(
            trace_collector,
            trace_prefix,
            "prune",
            args={"removed": int(len(to_remove))},
        ):
            solver.active_support.prune(mask)
    record_solve_event(
        "budgeted_pruning",
        iteration=None if current_inner_iter is None else int(current_inner_iter),
        removed=np.asarray(to_remove, dtype=np.int32),
        rows=solver.active_support.rows,
        cols=solver.active_support.cols,
    )
    return np.asarray(to_remove, dtype=np.int32)


__all__ = ["CleaningStrategy", "DualGapCleaning", "NoCleaning", "apply_budgeted_pruning"]
