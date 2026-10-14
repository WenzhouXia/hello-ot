from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional
from scipy.sparse import coo_matrix
import numba
from numba import types
import torch
import time
import numpy as np
import logging
from ..types.base import BaseHierarchy, BaseStrategy
from ..cost_types import normalize_cost_type_name
from ..lp.wrapper import LPSolver
from ..trace import _ChromeTraceCollector
from hello_ot.restricted_ot.active_support import ActiveSupport


from hello_ot.costs.pairs import (
    _to_pinned_numpy,
    _prepare_level_cost_cache_sqeuclidean,
    _prepare_level_cost_cache_euclidean,
    _prepare_level_cost_cache_lowrank,
    _prepare_level_cost_cache_l1,
    _prepare_level_cost_cache_linf,
    _cost_pairs_l1_batched,
    _cost_pairs_linf_batched,
    _cost_pairs_lowrank_batched,
    _cost_pairs_sqeuclidean_batched,
    _cost_pairs_euclidean_batched,
)

logger = logging.getLogger(__name__)


from hello_ot._internal.support_kernels import _northwest_corner_numba

fill_x_prev_sig = types.float32[:](
    # CN: primal row 下标。
    # EN: Primal row indices.
    types.int32[:],
    # CN: primal column 下标。
    # EN: Primal column indices.
    types.int32[:],
    # CN: primal 变量。
    # EN: Primal values.
    types.float32[:],
    # CN: 64位键避免线性化坐标溢出。
    # EN: 64-bit keys avoid overflow in linearized coordinates.
    types.int64[:],
    # CN: 排序后的 active-support 位置。
    # EN: Sorted active-support positions.
    types.int64[:],
    # CN: 待填充的 warm-start 向量。
    # EN: Warm-start vector to fill.
    types.float32[:],
    # CN: column 数。
    # EN: Number of columns.
    types.int64,
)


@numba.jit(fill_x_prev_sig, nopython=True, fastmath=True, parallel=True, cache=True)
def _fill_x_prev_numba(primal_rows, primal_cols, primal_data,
                       sorted_inc_keys, sorted_inc_pos, x_prev_filled, num_cols):
    """
    CN: 用 JIT 并行循环和 active-support 查找表填充 warm-start 向量 x_prev_filled。
    EN: Fill x_prev_filled with a JIT-compiled parallel loop and the active-support lookup table.
    """
    num_primal = len(primal_rows)
    # CN: numba.prange 在 CPU core 之间并行执行。
    # EN: numba.prange executes across CPU cores in parallel.
    for i in numba.prange(num_primal):
        # CN: 1. 将 (row, col) 坐标线性化为单个键。
        # EN: 1. Linearize the (row, col) coordinate into one key.
        key_to_find = np.int64(primal_rows[i]) * num_cols + primal_cols[i]

        # CN: 2. 在排序查找表上二分搜索。
        # EN: 2. Binary-search the sorted lookup table.
        index = np.searchsorted(sorted_inc_keys, key_to_find)

        # CN: 3. 确认查找到完整匹配的键。
        # EN: 3. Verify that an exact key match was found.
        if index < len(sorted_inc_keys) and sorted_inc_keys[index] == key_to_find:
            # CN: 4. 取原始位置并写入 primal 变量。
            # EN: 4. Recover the original position and assign the primal value.
            pos_to_update = sorted_inc_pos[index]
            value_to_assign = primal_data[i]
            x_prev_filled[pos_to_update] = value_to_assign

    return x_prev_filled


_expand_kernel_sig = types.void(
    # CN: coarse row 下标。
    # EN: Coarse row indices.
    types.int32[:],
    # CN: coarse column 下标。
    # EN: Coarse column indices.
    types.int32[:],
    # CN: coarse primal 变量。
    # EN: Coarse primal values.
    types.float64[:],
    # CN: source CSR 指针。
    # EN: Source CSR pointers.
    types.int32[:],
    # CN: source child 下标。
    # EN: Source child indices.
    types.int32[:],
    # CN: source child 权重。
    # EN: Source child weights.
    types.float64[:],
    # CN: target CSR 指针。
    # EN: Target CSR pointers.
    types.int32[:],
    # CN: target child 下标。
    # EN: Target child indices.
    types.int32[:],
    # CN: target child 权重。
    # EN: Target child weights.
    types.float64[:],
    # CN: 64位 edge offset 避免溢出。
    # EN: 64-bit edge offsets avoid overflow.
    types.int64[:],
    # CN: 输出 row 下标。
    # EN: Output row indices.
    types.int32[:],
    # CN: 输出 column 下标。
    # EN: Output column indices.
    types.int32[:],
    # CN: 输出 primal 变量。
    # EN: Output primal values.
    types.float32[:],
)


@numba.njit(_expand_kernel_sig, parallel=True, cache=True)
def _numba_expand_kernel(
    coarse_rows, coarse_cols, coarse_data,
    s_ptr, s_indices, s_weights,
    t_ptr, t_indices, t_weights,
    edge_offsets,
    out_rows, out_cols, out_data
):
    n_edges = len(coarse_data)
    for k in numba.prange(n_edges):
        c_r = coarse_rows[k]
        c_c = coarse_cols[k]
        val = coarse_data[k]

        s_start, s_end = s_ptr[c_r], s_ptr[c_r+1]
        t_start, t_end = t_ptr[c_c], t_ptr[c_c+1]

        write_idx = edge_offsets[k]

        for i in range(s_start, s_end):
            f_r = s_indices[i]
            w_r = s_weights[i]
            for j in range(t_start, t_end):
                f_c = t_indices[j]
                w_c = t_weights[j]

                out_rows[write_idx] = f_r
                out_cols[write_idx] = f_c
                out_data[write_idx] = val * w_r * w_c
                write_idx += 1


@numba.njit(types.int64[:](types.int32[:], types.int32[:], types.int64[:], types.int64[:]), cache=True)
def _precalc_offsets(coarse_rows, coarse_cols, s_counts, t_counts):
    n = len(coarse_rows)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    for i in range(n):
        # CN: 直接使用传入的 child counts。
        # EN: Use the supplied child counts directly.
        size = s_counts[coarse_rows[i]] * t_counts[coarse_cols[i]]
        offsets[i] = total
        total += size
    offsets[n] = total
    return offsets


class HierarchicalOTSolver:
    def __init__(
        self,
        hierarchy_s: BaseHierarchy,
        hierarchy_t: BaseHierarchy,
        strategy: Optional[BaseStrategy],
        solver: LPSolver,
        cleaning_strategy: Optional[CleaningStrategy] = None,
        lp_solver_verbose: int = 0,
    ):
        self.hierarchy_s = hierarchy_s
        self.hierarchy_t = hierarchy_t
        self.lp_solver = solver
        self.strategy = strategy
        self.cleaning_strategy = cleaning_strategy
        self.lp_solver_verbose = lp_solver_verbose

        self.solutions: Dict[int, Dict[str, Any]] = {}
        self.active_support: Optional[ActiveSupport] = None

        self.build_time = hierarchy_s.build_time + hierarchy_t.build_time
        self._profiling_enabled = False
        self._lp_solver_kwargs: Dict[str, Any] = {}
        self._use_gpu_active_support = False
        self._active_support_device = "cpu"
        self._report_added_violation_stats = False
        self._added_violation_rel_threshold = 1e-6
        self._runtime_logging: Dict[str, Any] = {
            "enabled": True,
            "progress": True,
            "profile_iter": True,
            "profile_level": True,
            "profile_run": True,
            "warm_start": True,
            "iter_interval": 10,
        }

    def _runtime_log_enabled(self, category: str) -> bool:
        cfg = getattr(self, "_runtime_logging", None)
        if not isinstance(cfg, dict):
            return True
        if not bool(cfg.get("enabled", True)):
            return False
        return bool(cfg.get(category, True))

    def _runtime_log(self, category: str, message: str, *, flush: bool = False) -> None:
        if self._runtime_log_enabled(category):
            print(message, flush=flush)

    @staticmethod
    def _prepare_level_cost_cache_sqeuclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_sqeuclidean(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_euclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_euclidean(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_lowrank(
        F: np.ndarray,
        G: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        *,
        dot_scale: float = 1.0,
        compute_precision: str = "fp32",
    ) -> Dict[str, Any]:
        return _prepare_level_cost_cache_lowrank(
            F, G, a, b, dot_scale=dot_scale, compute_precision=compute_precision
        )

    @staticmethod
    def _prepare_level_cost_cache_l1(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_l1(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_linf(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_linf(points_s, points_t)

    @staticmethod
    def _northwest_corner(p_in: np.ndarray, q_in: np.ndarray):
        return _northwest_corner_numba(p_in, q_in)

    @staticmethod
    def _fill_x_prev(
        primal_rows,
        primal_cols,
        primal_data,
        sorted_inc_keys,
        sorted_inc_pos,
        x_prev_filled,
        num_cols,
    ):
        return _fill_x_prev_numba(
            primal_rows,
            primal_cols,
            primal_data,
            sorted_inc_keys,
            sorted_inc_pos,
            x_prev_filled,
            num_cols,
        )

    def _create_active_support(self, level_cache: Dict[str, Any], *, track_creation: bool):
        return ActiveSupport(
            level_cache,
            track_creation=track_creation,
            device=getattr(self, "_active_support_device", "cpu"),
        )

    @staticmethod
    def _logger_is_enabled_info() -> bool:
        return logger.isEnabledFor(logging.INFO)

    @staticmethod
    def _logger_info(message: str, *args: Any) -> None:
        logger.info(message, *args)

    @staticmethod
    def _time_now() -> float:
        return time.perf_counter()

    @staticmethod
    def _sum_level_summary_metric(
        level_summaries: List[Dict[str, Any]],
        key: str,
    ) -> float:
        return float(
            sum(float(item.get(key, 0.0)) for item in level_summaries if isinstance(item, dict))
        )

    def _compute_pair_costs_arrays(
        self, rows_add, cols_add, **kwargs
    ):
        """
        CN: 为所有新增支撑边应用与扫描一致的成本扰动。
        EN: Apply the same cost perturbation as the scans to every inserted support edge.
        """
        costs = self._compute_base_pair_costs_arrays(rows_add, cols_add, **kwargs)
        perturbation = getattr(self, "_metric_cost_perturbation", None)
        if perturbation is not None:
            from hello_ot.costs.metric_perturbation import _apply_metric_pair_perturbation

            costs = _apply_metric_pair_perturbation(costs, rows_add, cols_add, perturbation)
        return costs

    def _compute_base_pair_costs_arrays(
        self,
        rows_add: np.ndarray,
        cols_add: np.ndarray,
        support_order_cache: Optional[Dict[str, Any]] = None,
    ):
        """
        CN: 按当前 active-support 的 level_cache 计算一组边的成本。
        EN: Compute costs for edge pairs using the current active-support level_cache.
        """
        level_cache = self.active_support.level_cache
        cost_type = normalize_cost_type_name(level_cache.get('cost_type', 'l2^2'))
        if cost_type == 'lowrank':
            return _cost_pairs_lowrank_batched(
                level_cache,
                rows_add,
                cols_add,
                support_order_cache=support_order_cache,
            )
        elif cost_type == 'l1':
            if getattr(self, "_cost_precision", "fp32") == "fp64":
                return self._compute_fp64_metric_pair_costs(level_cache, rows_add, cols_add, "l1")
            return _cost_pairs_l1_batched(level_cache, rows_add, cols_add)
        elif cost_type == 'linf':
            if getattr(self, "_cost_precision", "fp32") == "fp64":
                return self._compute_fp64_metric_pair_costs(level_cache, rows_add, cols_add, "linf")
            return _cost_pairs_linf_batched(level_cache, rows_add, cols_add)
        elif cost_type == 'l2':
            if getattr(self, "_cost_precision", "fp32") == "fp64":
                return self._compute_fp64_metric_pair_costs(level_cache, rows_add, cols_add, "l2")
            return _cost_pairs_euclidean_batched(level_cache, rows_add, cols_add)
        return _cost_pairs_sqeuclidean_batched(level_cache, rows_add, cols_add)

    @staticmethod
    def _compute_fp64_metric_pair_costs(level_cache, rows, cols, cost_type: str):
        from hello_ot.costs.metric import pair_costs_metric

        return pair_costs_metric(
            source_points=level_cache["S"],
            target_points=level_cache["T"],
            rows=rows,
            cols=cols,
            cost_type=cost_type,
            compute_precision="fp64",
        )

    def _append_new_pairs_arrays(
        self,
        rows_add: np.ndarray,
        cols_add: np.ndarray,
        current_inner_iter: int,
        c_add=None,
    ):
        if self.active_support._array_size(rows_add) == 0:
            return

        if c_add is None:
            c_add = self._compute_pair_costs_arrays(rows_add, cols_add)

        self.active_support.add_evaluated_pairs(
            rows_add, cols_add, c_add, current_inner_iter)

        logger.debug(f"  [_append] Added {len(rows_add)} pairs.")

    def _refine_solution(self, coarse_sol, coarse_lvl_s, coarse_lvl_t, fine_lvl_s, fine_lvl_t):
        primal_coarse, duals_coarse = coarse_sol['primal'], coarse_sol['dual']

        # CN: 将 coarse dual potentials 广播到 fine level。
        # EN: Broadcast coarse dual potentials to the fine level.
        f_coarse, g_coarse = duals_coarse[:len(
            coarse_lvl_s.points)], duals_coarse[len(coarse_lvl_s.points):]
        duals_fine = np.concatenate(
            (f_coarse[fine_lvl_s.child_labels], g_coarse[fine_lvl_t.child_labels]))

        # CN: 用 Numba 加速 primal interpolation。
        # EN: Interpolate the primal solution with Numba acceleration.
        p_s, p_t = fine_lvl_s.child_labels, fine_lvl_t.child_labels
        n_fine_s, n_fine_t = len(p_s), len(p_t)
        n_coarse_s, n_coarse_t = primal_coarse.shape

        s_counts = np.bincount(p_s, minlength=n_coarse_s)
        t_counts = np.bincount(p_t, minlength=n_coarse_t)

        primal_coarse_coo = primal_coarse.tocoo()
        coarse_data = np.asarray(primal_coarse_coo.data, dtype=np.float64)

        offsets = _precalc_offsets(
            primal_coarse_coo.row, primal_coarse_coo.col, s_counts, t_counts)
        total_nnz = offsets[-1]
        out_rows = np.empty(total_nnz, dtype=np.int32)
        out_cols = np.empty(total_nnz, dtype=np.int32)
        out_data = np.empty(total_nnz, dtype=np.float32)

        # CN: 构建 coarse-to-fine 映射。
        # EN: Build coarse-to-fine maps.
        w_s = 1.0 / np.maximum(s_counts[p_s], 1.0)
        w_t = 1.0 / np.maximum(t_counts[p_t], 1.0)

        map_s = coo_matrix((w_s, (p_s, np.arange(n_fine_s))),
                           shape=(n_coarse_s, n_fine_s)).tocsr()
        map_t = coo_matrix((w_t, (p_t, np.arange(n_fine_t))),
                           shape=(n_coarse_t, n_fine_t)).tocsr()

        _numba_expand_kernel(
            primal_coarse_coo.row, primal_coarse_coo.col, coarse_data,
            map_s.indptr, map_s.indices, map_s.data,
            map_t.indptr, map_t.indices, map_t.data,
            offsets, out_rows, out_cols, out_data
        )

        refined_primal = coo_matrix(
            (out_data, (out_rows, out_cols)), shape=(n_fine_s, n_fine_t))
        return refined_primal, duals_fine
