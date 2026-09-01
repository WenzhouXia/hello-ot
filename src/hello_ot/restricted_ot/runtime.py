from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csc_matrix

from hello_ot._internal.core.result_utils import build_objective_history_by_level
from hello_ot._internal.lp.torch_restricted_ot import TorchRestrictedOTPDLP
from hello_ot.kernels.torch_scan import resolve_torch_device
from hello_ot._internal.types.base import BaseHierarchy, HierarchyLevel
from hello_ot._internal.types.runtime import LevelState
from hello_ot.refinement.budgeted_pruning import DualGapCleaning
from hello_ot.restricted_ot.backend import solve_lp as solve_restricted_lp
from hello_ot.utilities import trace_span
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState
from hello_ot.types import InitializationResult

if TYPE_CHECKING:
    from hello_ot._internal.core.solver import HierarchicalOTSolver
    from hello_ot._internal.trace import _ChromeTraceCollector

logger = logging.getLogger(__name__)


class _CompositeScanOnlyStrategy:
    """
    CN: 正式 HELLO 的占位 strategy；候选边必须来自复合 dual scan，禁止 legacy pricing fallback。
    EN: Formal HELLO placeholder strategy; candidates must come from composite dual scans and legacy pricing fallback is forbidden.
    """

    last_dual_feasibility_info: Optional[Dict[str, Any]] = None

    def generate(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(
            "legacy pricing fallback is unavailable in formal HELLO; "
            "use the resident-streamed composite scan"
        )

    def close(self) -> None:
        return None


class SingleLevelHierarchy(BaseHierarchy):
    """
    CN: restricted OT 使用的单层 hierarchy 容器。
    EN: Single-level hierarchy container used by restricted OT.
    """

    def __init__(self, level: HierarchyLevel):
        super().__init__([level])
        self.levels = [level]
        self.num_levels = 1
        self.build_time = 0.0

    def prolongate(
        self,
        coarse_potential: np.ndarray,
        coarse_level_idx: int,
        fine_level_idx: int,
    ) -> np.ndarray:
        if int(coarse_level_idx) != 0 or int(fine_level_idx) != 0:
            raise ValueError("SingleLevelHierarchy only supports level 0 prolongation.")
        return coarse_potential


def build_lp_backend(config: SolverRuntimeConfig) -> Any:
    """
    CN: 根据 HELLO runtime 配置构造 restricted-OT LP backend。
    EN: Build the restricted-OT LP backend from the HELLO runtime configuration.
    """
    if config.solver_engine == "cupdlpx":
        from hello_ot._internal.lp.cupdlpx import CuPDLPxSolver

        return CuPDLPxSolver(
            bound_objective_rescaling=config.bound_objective_rescaling,
            cupdlpx_python_pre_rescale=bool(config.cupdlpx_python_pre_rescale),
        )
    if config.solver_engine == "torch_pdlp":
        return TorchRestrictedOTPDLP(device=resolve_torch_device(config.torch_device))
    raise ValueError(f"unsupported HELLO LP backend: {config.solver_engine}")


def _pricing_and_pruning(
    config: SolverRuntimeConfig,
    *,
    pricing_index_pool: Optional[Any] = None,
) -> Tuple[Any, DualGapCleaning]:
    del pricing_index_pool
    strategy = _CompositeScanOnlyStrategy()
    pruning = DualGapCleaning(
        threshold_factor=float(config.cleaning_threshold_factor),
        primal_tol=float(config.cleaning_primal_tol),
    )
    return strategy, pruning


def _finish_restricted_solver(
    solver: HierarchicalOTSolver,
    config: SolverRuntimeConfig,
    *,
    cost_type: str,
) -> Tuple[HierarchicalOTSolver, SolverRuntimeConfig]:
    if config.backend == "native" and not torch.cuda.is_available():
        raise RuntimeError("HELLO native restricted OT requires CUDA.")
    solver._cost_type = str(cost_type)
    solver._cost_dot_scale = float(config.dot_scale)
    # CN: HELLO 的正式计时使用 API/阶段直接 wall-time；禁止同步式 RuntimeProfiler 进入求解热路径。
    # EN: Formal HELLO timing uses direct API/stage wall time; the synchronizing RuntimeProfiler is excluded from the hot path.
    solver._runtime_logging = config.normalized_runtime_logging()
    solver._report_added_violation_stats = bool(config.report_added_violation_stats)
    solver._added_violation_rel_threshold = float(config.added_violation_rel_threshold)
    solver._variable_bound_mode = str(config.variable_bound_mode)
    solver._matrix_value_mode = str(config.matrix_value_mode)
    solver._vector_sum_mode = str(config.vector_sum_mode)
    solver._cupdlpx_python_pre_rescale = bool(config.cupdlpx_python_pre_rescale)
    active_device = (
        "cuda"
        if config.backend == "native"
        else str(resolve_torch_device(config.torch_device))
    )
    solver._backend = str(config.backend)
    solver._use_gpu_active_support = bool(torch.device(active_device).type == "cuda")
    solver._active_support_device = str(active_device)
    lp_kwargs = dict(getattr(solver, "_lp_solver_kwargs", {}) or {})
    params = dict(lp_kwargs.get("solver_params", {}) or {})
    params["termination_norm"] = str(config.lp_termination_norm)
    lp_kwargs["solver_params"] = params
    lp_kwargs.update(
        {
            "use_gpu_lp_pipeline": bool(config.backend == "native"),
            "gpu_pipeline_device": str(active_device),
            "variable_bound_mode": str(config.variable_bound_mode),
            "matrix_value_mode": str(config.matrix_value_mode),
            "vector_sum_mode": str(config.vector_sum_mode),
            "cupdlpx_python_pre_rescale": bool(config.cupdlpx_python_pre_rescale),
        }
    )
    solver._lp_solver_kwargs = lp_kwargs
    return solver, config


def build_lowrank_restricted_solver(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_cost: np.ndarray,
    target_cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    config: SolverRuntimeConfig,
    pricing_index_pool: Optional[Any] = None,
    dot_scale: float = 1.0,
) -> Tuple[HierarchicalOTSolver, SolverRuntimeConfig]:
    """
    CN: 构造 low-rank cost 的单层 GPU restricted-OT solver。
    EN: Build a single-level GPU restricted-OT solver for a low-rank cost.
    """
    from hello_ot._internal.core.solver import HierarchicalOTSolver

    source_level = HierarchyLevel(
        level_idx=0,
        points=np.asarray(source_features, dtype=np.float32, order="C"),
        masses=np.asarray(source_mass, dtype=np.float64, order="C"),
        cost_vec=np.asarray(source_cost, dtype=np.float64, order="C"),
    )
    target_level = HierarchyLevel(
        level_idx=0,
        points=np.asarray(target_features, dtype=np.float32, order="C"),
        masses=np.asarray(target_mass, dtype=np.float64, order="C"),
        cost_vec=np.asarray(target_cost, dtype=np.float64, order="C"),
    )
    strategy, pruning = _pricing_and_pruning(
        config, pricing_index_pool=pricing_index_pool
    )
    solver = HierarchicalOTSolver(
        hierarchy_s=SingleLevelHierarchy(source_level),
        hierarchy_t=SingleLevelHierarchy(target_level),
        strategy=strategy,
        solver=build_lp_backend(config),
        cleaning_strategy=pruning,
        lp_solver_verbose=bool(config.lp_solver_verbose),
    )
    config.dot_scale = float(dot_scale)
    config.validate()
    return _finish_restricted_solver(solver, config, cost_type="lowrank")


def build_metric_restricted_solver(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    config: SolverRuntimeConfig,
    *,
    cost_type: str,
) -> Tuple[HierarchicalOTSolver, SolverRuntimeConfig]:
    """
    CN: 构造 metric cost 的单层 GPU restricted-OT solver。
    EN: Build a single-level GPU restricted-OT solver for a metric cost.
    """
    from hello_ot._internal.core.solver import HierarchicalOTSolver

    if str(cost_type) not in {"l1", "linf", "l2"}:
        raise ValueError("metric restricted OT supports cost_type in {'l1', 'linf', 'l2'}")
    source_level = HierarchyLevel(
        level_idx=0,
        points=np.asarray(source_points, dtype=np.float32, order="C"),
        masses=np.asarray(source_mass, dtype=np.float64, order="C"),
    )
    target_level = HierarchyLevel(
        level_idx=0,
        points=np.asarray(target_points, dtype=np.float32, order="C"),
        masses=np.asarray(target_mass, dtype=np.float64, order="C"),
    )
    strategy, pruning = _pricing_and_pruning(config)
    solver = HierarchicalOTSolver(
        hierarchy_s=SingleLevelHierarchy(source_level),
        hierarchy_t=SingleLevelHierarchy(target_level),
        strategy=strategy,
        solver=build_lp_backend(config),
        cleaning_strategy=pruning,
        lp_solver_verbose=bool(config.lp_solver_verbose),
    )
    return _finish_restricted_solver(solver, config, cost_type=str(cost_type))


def sparse_coupling_from_data(
    data: Optional[Dict[str, np.ndarray]], shape: Tuple[int, int]
) -> sp.coo_matrix:
    if not data:
        return sp.coo_matrix(shape, dtype=np.float64)
    return sp.coo_matrix(
        (data["values"], (data["rows"], data["cols"])),
        shape=shape,
        dtype=np.float64,
    )


def sparse_coupling_from_active_support(
    solver: HierarchicalOTSolver, shape: Tuple[int, int]
) -> sp.coo_matrix:
    if solver.active_support is None:
        return sp.coo_matrix(shape, dtype=np.float64)
    arrays = solver.active_support.export_numpy()
    positive = np.asarray(arrays["x_prev"], dtype=np.float64) > np.float64(1e-12)
    return sp.coo_matrix(
        (
            np.asarray(arrays["x_prev"], dtype=np.float64)[positive],
            (
                np.asarray(arrays["rows"], dtype=np.int32)[positive],
                np.asarray(arrays["cols"], dtype=np.int32)[positive],
            ),
        ),
        shape=shape,
        dtype=np.float64,
    )


def state_from_active_support(
    solver: HierarchicalOTSolver,
    *,
    n_source: int,
    n_target: int,
    dual: Optional[np.ndarray],
    fallback_result: Optional[Dict[str, Any]] = None,
) -> OTWarmStartState:
    coupling = sparse_coupling_from_active_support(solver, (n_source, n_target))
    if coupling.nnz == 0 and fallback_result is not None:
        coupling = sparse_coupling_from_data(
            fallback_result.get("sparse_coupling"), (n_source, n_target)
        )
    coupling = coupling.tocoo(copy=False)
    if dual is None and fallback_result is not None:
        dual = fallback_result.get("dual")
    return OTWarmStartState(
        rows=np.asarray(coupling.row, dtype=np.int32).copy(),
        cols=np.asarray(coupling.col, dtype=np.int32).copy(),
        x_prev=np.asarray(coupling.data, dtype=np.float64).copy(),
        dual_uv=None if dual is None else np.asarray(dual, dtype=np.float64).copy(),
        n_source=int(n_source),
        n_target=int(n_target),
    )


def extract_level_zero_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    CN: 提取 finest level 的轻量 LP 与 active-support 统计。
    EN: Extract compact LP and active-support statistics for the finest level.
    """
    level = next(
        (
            item
            for item in result.get("level_summaries", ())
            if int(item.get("level", -1)) == 0
        ),
        None,
    )
    if level is None:
        return {
            "level0_inner_iterations": -1,
            "level0_lp_time_total": float("nan"),
            "level0_pricing_time_total": float("nan"),
            "level0_total_time": float("nan"),
            "final_active_support_size": 0,
        }
    return {
        "level0_inner_iterations": int(level.get("iters", -1)),
        "level0_lp_time_total": float(level.get("lp_time", float("nan"))),
        "level0_pricing_time_total": float(level.get("pricing_time", float("nan"))),
        "level0_total_time": float(level.get("time", float("nan"))),
        "level0_lp_backend_peak_mem_mib": level.get("lp_backend_peak_mem_mib"),
        "level0_pricing_peak_mem_mib": level.get("pricing_peak_mem_mib"),
        "level0_dual_feas_peak_mem_mib": level.get("dual_feas_peak_mem_mib"),
        "final_active_support_size": int(level.get("support_final", 0)),
    }


def sum_level_summary_metric(
    level_summaries: Any, field: str
) -> float:
    return float(
        sum(
            float(item.get(field, 0.0) or 0.0)
            for item in level_summaries or ()
            if isinstance(item, dict)
        )
    )


def finalize_solve_output(
    *,
    distance: float,
    coupling: sp.coo_matrix,
    state: Optional[OTWarmStartState],
    dual_source: Optional[np.ndarray],
    level_summaries: Any,
    lp_solve_time_total: float,
    elapsed: float,
    extra_log_fields: Optional[Dict[str, Any]] = None,
    chrome_trace: Optional[Dict[str, Any]] = None,
    log: bool,
    return_coupling: bool,
    return_state: bool,
) -> Any:
    """
    CN: 将 restricted-OT 内部结果整理为迁移期求解返回格式。
    EN: Package internal restricted-OT results into the migration-time solve format.
    """
    if log:
        payload = {
            "distance": float(distance),
            "time": float(elapsed),
            "lp_solve_time_total": float(lp_solve_time_total),
            "level_summaries": level_summaries,
            "dual_source": dual_source,
            "sparse_coupling": coupling if return_coupling else None,
            "warm_start_state": state,
        }
        if extra_log_fields:
            payload.update(extra_log_fields)
        if chrome_trace is not None:
            payload["chrome_trace"] = chrome_trace
        return payload
    if return_coupling and return_state:
        return float(distance), coupling, state
    if return_coupling:
        return float(distance), coupling
    if return_state:
        return float(distance), state
    return float(distance)


def should_exact_flat(solver) -> bool:
    return solver.hierarchy_s.num_levels <= 1


def solve_exact_flat(
    solver,
    tolerance: Dict[str, float],
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot",
) -> Optional[Dict[str, Any]]:
    logger.info("Single level detected. Running exact solver.")
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level

    t_lp_start = time.perf_counter()
    primal_dense, dual_sol, result, solve_meta = solve_restricted_lp(
        solver,
        lvl_s,
        lvl_t,
        tolerance,
        verbose=solver.lp_solver_verbose,
        trace_collector=trace_collector,
        trace_prefix=f"{trace_prefix}.lp",
        trace_args={"level": 0, "iter": 0, "inner_iter": 1, "phase": "single_level_exact", "is_coarsest": True},
    )
    lp_solve_time_total = float(time.perf_counter() - t_lp_start)
    if not result.success:
        return None

    primal_coo = primal_dense.tocoo()
    mask = primal_coo.data > 1e-12
    primal_sparse = csc_matrix(
        (primal_coo.data[mask], (primal_coo.row[mask], primal_coo.col[mask])),
        shape=primal_dense.shape,
    )
    level_summary = {
        "level": 0,
        "n_source": int(primal_dense.shape[0]),
        "n_target": int(primal_dense.shape[1]),
        "iters": 1,
        "time": lp_solve_time_total,
        "objective": float(result.obj_val),
        "lp_time": lp_solve_time_total,
        "pricing_time": 0.0,
        "support_pre_lp_final": None,
        "support_final": int(primal_sparse.nnz),
        "stop_reason": None,
        "converged": True,
        "hit_itermax": False,
        "converged_before_itermax": True,
    }
    solutions = {0: {"history": [result.obj_val]}}

    return {
        "primal": primal_sparse,
        "dual": dual_sol,
        "final_obj": result.obj_val,
        "all_history": solutions,
        "objective_history_by_level": build_objective_history_by_level(solutions),
        "sparse_coupling": {
            "rows": primal_coo.row[mask],
            "cols": primal_coo.col[mask],
            "values": primal_coo.data[mask],
        },
        "level_summaries": [level_summary],
        "lp_solve_time_total": lp_solve_time_total,
    }


def init_run_state(solver, tolerance: Dict[str, float], **kwargs: Any) -> Dict[str, Any]:
    del kwargs
    return {
        "finest_idx": 0,
        "coarsest_idx": solver.hierarchy_s.num_levels - 1,
        "tolerance": tolerance,
    }


def get_level_indices(_solver, run_state: Dict[str, Any]):
    return range(run_state["coarsest_idx"], run_state["finest_idx"] - 1, -1)


def get_max_inner_iters(_solver, level_state: Dict[str, Any], max_inner_iter: int) -> int:
    if level_state.get("is_coarsest"):
        return 1
    return max_inner_iter


def initialize_level(
    solver,
    level_idx: int,
    cost_type: str,
    use_bfs_skeleton: bool,
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.init_level_state",
) -> Tuple[csc_matrix, np.ndarray]:
    fine_lvl_s = solver.hierarchy_s.levels[level_idx]
    fine_lvl_t = solver.hierarchy_t.levels[level_idx]
    coarse_sol = solver.solutions[level_idx + 1]

    n_s = len(fine_lvl_s.points)
    n_t = len(fine_lvl_t.points)
    init_components: Dict[str, float] = {}

    t0 = time.perf_counter()
    with trace_span(trace_collector, trace_prefix, "refine_solution", args={"level": int(level_idx)}):
        primal_curr, dual_curr = solver._refine_solution(
            coarse_sol,
            solver.hierarchy_s.levels[level_idx + 1],
            solver.hierarchy_t.levels[level_idx + 1],
            fine_lvl_s,
            fine_lvl_t,
        )
    init_components["refine_solution"] = time.perf_counter() - t0
    logger.debug("  [Level %s] Init: primal nnz=%s", level_idx, primal_curr.nnz)

    cost_type = solver._normalize_cost_type_name(cost_type)
    t0 = time.perf_counter()
    with trace_span(trace_collector, trace_prefix, "build_level_cache", args={"level": int(level_idx)}):
        if cost_type == "lowrank":
            level_cache = solver._prepare_level_cost_cache_lowrank(
                fine_lvl_s.points,
                fine_lvl_t.points,
                fine_lvl_s.cost_vec,
                fine_lvl_t.cost_vec,
                dot_scale=float(getattr(solver, "_cost_dot_scale", 1.0)),
            )
        elif cost_type == "l1":
            level_cache = solver._prepare_level_cost_cache_l1(fine_lvl_s.points, fine_lvl_t.points)
        elif cost_type == "linf":
            level_cache = solver._prepare_level_cost_cache_linf(fine_lvl_s.points, fine_lvl_t.points)
        elif cost_type == "l2":
            level_cache = solver._prepare_level_cost_cache_euclidean(fine_lvl_s.points, fine_lvl_t.points)
        else:
            level_cache = solver._prepare_level_cost_cache_sqeuclidean(fine_lvl_s.points, fine_lvl_t.points)
    init_components["build_level_cache"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with trace_span(trace_collector, trace_prefix, "init_active_support", args={"level": int(level_idx)}):
        solver.active_support = solver._create_active_support(
            level_cache, track_creation=solver.cleaning_strategy.needs_creation_iteration()
        )
    init_components["init_active_support"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    init_args = (primal_curr, (dual_curr[:n_s], dual_curr[n_s:]), level_cache)
    with trace_span(trace_collector, trace_prefix, "initial_pricing", args={"level": int(level_idx)}):
        init_cands = solver.strategy.generate(
            *init_args,
            level_idx=int(level_idx),
            inner_iter=-1,
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.initial_pricing.strategy",
        )
    init_components["initial_pricing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with trace_span(trace_collector, trace_prefix, "append_initial_candidates"):
        if cost_type == "lowrank":
            solver.active_support.add_pairs_placeholder(init_cands[0], init_cands[1], iter_idx=0)
        else:
            solver._append_new_pairs_arrays(init_cands[0], init_cands[1], 0)
    init_components["append_initial_candidates"] = time.perf_counter() - t0

    bfs_added = 0
    if use_bfs_skeleton:
        t0 = time.perf_counter()
        with trace_span(trace_collector, trace_prefix, "bfs_skeleton"):
            bfs_rows, bfs_cols = solver._northwest_corner(fine_lvl_s.masses, fine_lvl_t.masses)
            if cost_type == "lowrank":
                rows_add, cols_add, _, bfs_added = solver.active_support.filter_new_candidate_pairs(bfs_rows, bfs_cols)
                solver.active_support.add_pairs_placeholder(rows_add, cols_add, iter_idx=-1)
            else:
                solver._append_new_pairs_arrays(bfs_rows, bfs_cols, -1)
                bfs_added = int(len(bfs_rows))
        init_components["bfs_skeleton"] = time.perf_counter() - t0
        logger.debug("  [Level %s] Init BFS added %s edges.", level_idx, bfs_added)

    if cost_type == "lowrank":
        t0 = time.perf_counter()
        with trace_span(trace_collector, trace_prefix, "order_cache_build"):
            order_cache = solver.active_support.get_order_cache(n_source=n_s, n_target=n_t, device="cuda")
        init_components["order_cache_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with trace_span(trace_collector, trace_prefix, "compute_c_vec"):
            c_vec = solver._compute_pair_costs_arrays(
                solver.active_support.rows,
                solver.active_support.cols,
                support_order_cache=order_cache,
            )
            solver.active_support.replace_costs(c_vec)
        init_components["compute_c_vec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with trace_span(trace_collector, trace_prefix, "warm_start_fill"):
        primal_coo = primal_curr.tocoo()
        sorted_inc_keys, sorted_inc_pos = solver.active_support.sorted_keys_with_positions_numpy()
        x_prev_filled = solver._fill_x_prev(
            primal_coo.row,
            primal_coo.col,
            primal_coo.data,
            sorted_inc_keys,
            sorted_inc_pos,
            solver.active_support.export_numpy()["x_prev"],
            n_t,
        )
        solver.active_support.set_x_prev(x_prev_filled)
    init_components["warm_start_fill"] = time.perf_counter() - t0

    solver._print_profile_init_level(
        level_idx,
        init_components,
        primal_nnz=primal_curr.nnz,
        active_size=solver.active_support.size,
        bfs_added=bfs_added,
    )

    return primal_curr, dual_curr


def init_level_state(
    solver,
    level_idx: int,
    run_state: Dict[str, Any],
    tolerance: Dict[str, float],
    cost_type: str,
    use_bfs_skeleton: bool,
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.init_level_state",
) -> Optional[Dict[str, Any]]:
    coarsest_idx = run_state["coarsest_idx"]
    if level_idx == coarsest_idx:
        solver._runtime_log("progress", f"\n=== Solve The Coarsest Level {coarsest_idx} =====")
        return {
            "level_idx": level_idx,
            "is_coarsest": True,
            "level_s": solver.hierarchy_s.levels[level_idx],
            "level_t": solver.hierarchy_t.levels[level_idx],
            "t_level_start": time.perf_counter(),
            "cost_type": cost_type,
        }

    solver._runtime_log("progress", f"\n=== Solve Level {level_idx} =====")
    primal_curr, dual_curr = initialize_level(
        solver,
        level_idx,
        cost_type,
        use_bfs_skeleton,
        trace_collector=trace_collector,
        trace_prefix=f"{trace_prefix}.hello",
    )
    return {
        "level_idx": level_idx,
        "is_coarsest": False,
        "level_s": solver.hierarchy_s.levels[level_idx],
        "level_t": solver.hierarchy_t.levels[level_idx],
        "t_level_start": time.perf_counter(),
        "primal_curr": primal_curr,
        "dual_curr": dual_curr,
        "level_obj_hist": [],
        "level_lp_time": 0.0,
        "level_pricing_time": 0.0,
        "plateau_counter": 0,
        "_is_converged": False,
        "current_iter": 0,
        "completed_iters": 0,
        "tolerance": tolerance,
        "cost_type": cost_type,
    }


def initial(problem_def, algorithm_state, level_index: int):
    level_data = init_level_state(
        problem_def.solver,
        level_index,
        algorithm_state.run_state,
        problem_def.tolerance,
        problem_def.cost_type,
        problem_def.use_bfs_skeleton,
        trace_collector=problem_def.trace_collector,
        trace_prefix=f"{problem_def.trace_prefix}.solve.init_level_state",
    )
    return LevelState.from_legacy_data(level_index=level_index, max_inner_iter=0, data=level_data)


@dataclass
class InitializedLevel:
    """
    CN: 单层 refinement 进入首个 restricted LP 前的 GPU runtime；字段均为已有对象引用。
    EN: GPU runtime before the first restricted LP of one level; every field references an existing object.
    """

    solver: HierarchicalOTSolver
    cfg: SolverRuntimeConfig
    initialization: InitializationResult

    @property
    def active_support(self) -> Any:
        return self.initialization.active_support

    @property
    def dual_warm_start(self) -> Optional[Any]:
        return self.initialization.dual_warm_start

    @property
    def statistics(self) -> Mapping[str, Any]:
        return self.initialization.statistics


def _init_lowrank_warm_start_refinement(
    *,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    config: SolverRuntimeConfig,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    skip_initial_pricing: bool,
    trace_collector: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    pricing_index_pool: Optional[Any],
    warm_start_cost_vec_chunk_size: int,
    warm_start_cost_vec_feature_chunk_size: Optional[int],
    _consume_warm_start_gpu_state: bool = False,
    dot_scale: float = 1.0,
) -> InitializedLevel:
    """
    CN: 准备 lowrank warm-start refinement 的 solver 和 active support。
    EN: Prepare the solver and active support for lowrank warm-start refinement.
    """
    from hello_ot.initialization.initial_support import (
        _construct_initial_active_support,
        _validate_lowrank_warm_start,
    )
    from hello_ot.state import consume_gpu_warm_start as _drop_consumed_gpu_warm_start_state

    # CN: 先校验 warm-start 的形状和对偶信息，后续流程只保留需要继续传递的 dual_uv。
    # EN: Validate warm-start shapes and dual data first; only dual_uv is carried forward here.
    n_s = int(source_F.shape[0])
    n_t = int(target_G.shape[0])
    with (
        trace_collector.span(f"{trace_prefix}.validate_warm_start", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        rows, cols, x_prev, dual_uv = _validate_lowrank_warm_start(warm_start, n_s=n_s, n_t=n_t)
    del rows, cols, x_prev
    level_cache_holder: Dict[str, Any] = {}
    northwest_holder: Dict[str, Any] = {}

    # CN: 为单层 lowrank refinement 构建 restricted-OT solver，并复用 pricing index pool。
    # EN: Build the single-level low-rank restricted-OT solver and reuse the pricing index pool.
    with (
        trace_collector.span(f"{trace_prefix}.build_solver", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        solver, cfg = build_lowrank_restricted_solver(
            source_features=source_F,
            target_features=target_G,
            source_cost=source_cost_vec,
            target_cost=target_cost_vec,
            source_mass=np.asarray(source_mass, dtype=np.float64),
            target_mass=np.asarray(target_mass, dtype=np.float64),
            config=config,
            pricing_index_pool=pricing_index_pool,
            dot_scale=float(dot_scale),
        )

    # CN: 从 warm-start 播种 active support，并按常规路径补 NW basis 与 initial pricing。
    # EN: Construct the active support with the regular NW-basis and initial-pricing path.
    with (
        trace_collector.span(f"{trace_prefix}.init", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        initial_support_info = _construct_initial_active_support(
            solver,
            warm_start,
            skip_initial_pricing=bool(skip_initial_pricing),
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            cost_vec_chunk_size=int(warm_start_cost_vec_chunk_size),
            cost_vec_feature_chunk_size=warm_start_cost_vec_feature_chunk_size,
            level_cache_holder=level_cache_holder,
            northwest_holder=northwest_holder,
            consume_warm_start_gpu_state=bool(_consume_warm_start_gpu_state),
        )
        if bool(_consume_warm_start_gpu_state) and not bool(
            initial_support_info.get("consumed_warm_start_gpu_state", False)
        ):
            initial_support_info["consumed_warm_start_gpu_state"] = _drop_consumed_gpu_warm_start_state(
                warm_start,
                keep_dual=False,
            )

    return InitializedLevel(
        solver=solver,
        cfg=cfg,
        initialization=InitializationResult(
            active_support=solver.active_support,
            dual_warm_start=dual_uv,
            statistics=initial_support_info,
        ),
    )
