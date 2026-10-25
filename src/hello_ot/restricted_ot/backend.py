from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from hello_ot._internal.instrumentation.phase_names import (
    COMPONENT_LP_DIAG_EXTRACT,
    COMPONENT_LP_GET_SOLUTION,
    COMPONENT_LP_LOAD_DATA,
    COMPONENT_LP_PARAM_PACK,
    COMPONENT_LP_RESULT_PACK,
    COMPONENT_LP_WARM_START,
)
from hello_ot._internal.instrumentation.driver_memory import DriverMemoryTracker
from hello_ot._internal.instrumentation.memory_accounting import current_solve_memory_tracker
from hello_ot._internal.lp.device_pre_rescale import maybe_pre_rescale_device_csr_payload, should_direct_pre_rescale_values
from hello_ot._internal.lp.dispatch import solve_lp as dispatch_lp_solve
from hello_ot._internal.lp.restricted_ot_pdlp import RestrictedOTPDLP
from hello_ot._internal.cost_types import normalize_cost_type_name as _normalize_cost_type_name


def _is_torch_tensor(value: Any) -> bool:
    return bool(TORCH_AVAILABLE and torch.is_tensor(value))


def _parse_bool_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _infer_device(rows: Any, cols: Any, c: Any, masses_s: Any, masses_t: Any, requested_device: Optional[str]) -> "torch.device":
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for the HELLO GPU LP pipeline.")
    if requested_device is not None:
        return torch.device(requested_device)
    for value in (rows, cols, c, masses_s, masses_t):
        if _is_torch_tensor(value):
            return value.device
    return torch.device("cuda")


def _to_1d_cuda_tensor(value: Any, *, dtype, device: "torch.device") -> "torch.Tensor":
    if _is_torch_tensor(value):
        tensor = value.detach()
        if tensor.device != device:
            tensor = tensor.to(device=device)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous().view(-1)
    return torch.as_tensor(value, dtype=dtype, device=device).contiguous().view(-1)


def _build_device_ot_csr(
    *,
    rows: Any,
    cols: Any,
    c: Any,
    masses_s: Any,
    masses_t: Any,
    n_source: int,
    n_target: int,
    requested_device: Optional[str],
    support_order_cache: Optional[Dict[str, Any]] = None,
    variable_bound_mode: str = "explicit",
    matrix_value_mode: str = "explicit",
    cupdlpx_python_pre_rescale: bool = False,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for the HELLO GPU LP pipeline.")
    device = _infer_device(rows, cols, c, masses_s, masses_t, requested_device)
    if device.type != "cuda":
        raise ValueError(f"HELLO GPU LP pipeline requires a CUDA device, got {device}.")

    rows_t = _to_1d_cuda_tensor(rows, dtype=torch.int64, device=device)
    cols_t = _to_1d_cuda_tensor(cols, dtype=torch.int64, device=device)
    c_t = _to_1d_cuda_tensor(c, dtype=torch.float64, device=device)
    if rows_t.numel() != cols_t.numel() or rows_t.numel() != c_t.numel():
        raise ValueError("rows, cols, and c must have the same length for device LP pipeline.")

    n_vars = int(rows_t.numel())
    n_constraints = int(n_source) + int(n_target)
    col_ids = torch.arange(n_vars, device=device, dtype=torch.int32)

    cache_usable = (
        support_order_cache is not None
        and int(support_order_cache.get("n_vars", -1)) == n_vars
        and int(support_order_cache.get("n_source", -1)) == int(n_source)
        and int(support_order_cache.get("n_target", -1)) == int(n_target)
        and "source_order" in support_order_cache
        and "target_order" in support_order_cache
        and "source_rowptr" in support_order_cache
        and "target_rowptr" in support_order_cache
    )
    if cache_usable:
        source_order = support_order_cache["source_order"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        target_order = support_order_cache["target_order"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        source_rowptr = support_order_cache["source_rowptr"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        target_rowptr = support_order_cache["target_rowptr"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        row_ptr = torch.empty(n_constraints + 1, device=device, dtype=torch.int32)
        row_ptr[: int(n_source) + 1] = source_rowptr.to(dtype=torch.int32)
        if int(n_target) > 0:
            row_ptr[int(n_source) + 1 :] = (source_rowptr[-1] + target_rowptr[1:]).to(dtype=torch.int32)
    else:
        source_counts = torch.bincount(rows_t, minlength=int(n_source)).to(dtype=torch.int32)
        target_counts = torch.bincount(cols_t, minlength=int(n_target)).to(dtype=torch.int32)
        row_ptr = torch.empty(n_constraints + 1, device=device, dtype=torch.int32)
        row_ptr[0] = 0
        if int(n_source) > 0:
            row_ptr[1 : int(n_source) + 1] = torch.cumsum(source_counts, dim=0)
        if int(n_target) > 0:
            row_ptr[int(n_source) + 1 :] = row_ptr[int(n_source)] + torch.cumsum(target_counts, dim=0)
        source_order = torch.argsort(rows_t)
        target_order = torch.argsort(cols_t)
    col_ind = torch.cat([col_ids[source_order], col_ids[target_order]], dim=0).contiguous()
    matrix_value_mode = str(matrix_value_mode).strip().lower()
    if matrix_value_mode not in {"explicit", "implicit_aty", "implicit_ax", "implicit_both"}:
        raise ValueError("matrix_value_mode must be one of: explicit, implicit_aty, implicit_ax, implicit_both")
    a_explicit = matrix_value_mode in {"explicit", "implicit_aty"}
    direct_pre_rescale_values = should_direct_pre_rescale_values(bool(cupdlpx_python_pre_rescale), matrix_value_mode)
    values = None if (not a_explicit) or direct_pre_rescale_values else torch.ones(2 * n_vars, device=device, dtype=torch.float64)
    rhs = torch.cat(
        [
            _to_1d_cuda_tensor(masses_s, dtype=torch.float64, device=device),
            _to_1d_cuda_tensor(masses_t, dtype=torch.float64, device=device),
        ],
        dim=0,
    ).contiguous()
    variable_bound_mode = str(variable_bound_mode).strip().lower()
    if variable_bound_mode not in {"explicit", "constant"}:
        raise ValueError("variable_bound_mode must be one of: explicit, constant")
    if variable_bound_mode == "constant":
        lb = 0.0
        ub = float("inf")
    else:
        lb = torch.zeros(n_vars, device=device, dtype=torch.float64)
        ub = torch.full((n_vars,), float("inf"), device=device, dtype=torch.float64)

    payload = {
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        **({} if values is None else {"values": values}),
        "matrix_value_mode": matrix_value_mode,
        "c": c_t,
        "rhs": rhs,
        "lb": lb,
        "ub": ub,
        "variable_bound_mode": variable_bound_mode,
        "m": n_constraints,
        "n": n_vars,
        "n_eqs": n_constraints,
        "device": device,
        "order_cache_used": bool(cache_usable),
    }
    row_ids = None
    if direct_pre_rescale_values and a_explicit:
        row_ids = torch.cat(
            [
                rows_t[source_order],
                cols_t[target_order] + int(n_source),
            ],
            dim=0,
        ).to(dtype=torch.int64).contiguous()
    payload, pre_diag = maybe_pre_rescale_device_csr_payload(
        payload,
        requested=bool(cupdlpx_python_pre_rescale),
        row_ids=row_ids,
        source="hello_device_ot_csr_builder",
    )
    payload["cupdlpx_python_pre_rescale_builder_diag"] = pre_diag
    payload["cupdlpx_python_pre_rescale_requested"] = bool(cupdlpx_python_pre_rescale)
    return payload


def _to_numpy_1d(value: Any, dtype) -> np.ndarray:
    if _is_torch_tensor(value):
        tensor = value.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy().astype(dtype, copy=False).reshape(-1)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _array_length_1d(value: Any) -> int:
    if _is_torch_tensor(value):
        return int(value.numel())
    return int(np.asarray(value).size)


def _trace_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

@contextmanager
def _timed_trace_span(trace_collector: Optional[Any], name: str, *, args: Optional[Dict[str, Any]] = None):
    """
    CN: 结合 trace span 与 CUDA 同步，改善 restricted-LP 包装层的计时归因。
    EN: Combine a trace span with CUDA synchronization to improve timing attribution in the restricted-LP wrapper.
    """
    if trace_collector is None:
        yield
        return
    torch.cuda.synchronize()
    with trace_collector.span(name, "solve_ot", args=args):
        yield
    torch.cuda.synchronize()


def _cuda_device_index(device: Optional[str]) -> int:
    """
    CN: 从 cuda / cuda:idx 字符串中提取设备编号；解析失败时回退 0。
    EN: Extract the device index from cuda / cuda:idx strings; fall back to 0 on parse failure.
    """
    text = str(device or "").strip().lower()
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _build_lp_trace_args(
    base_args: Optional[Dict[str, Any]],
    *,
    n_source: int,
    n_target: int,
    n_vars: int,
    n_constraints: int,
    use_incremental: bool,
    use_gpu_lp_pipeline: bool,
    gpu_pipeline_device: Optional[str],
    warm_start_dual: Any,
    active_support_size: int,
) -> Dict[str, Any]:
    trace_args: Dict[str, Any] = {}
    if isinstance(base_args, dict):
        trace_args.update({str(key): _trace_scalar(value) for key, value in base_args.items()})
    trace_args.update(
        {
            "n_source": int(n_source),
            "n_target": int(n_target),
            "n_vars": int(n_vars),
            "n_constraints": int(n_constraints),
            "use_incremental": bool(use_incremental),
            "use_gpu_lp_pipeline": bool(use_gpu_lp_pipeline),
            "warm_start_dual_present": warm_start_dual is not None,
            "active_support_size": int(active_support_size),
        }
    )
    if gpu_pipeline_device is not None:
        trace_args["gpu_pipeline_device"] = str(gpu_pipeline_device)
    return trace_args


@dataclass(frozen=True)
class _LPBackendOptions:
    use_gpu_pipeline: bool
    gpu_device: Optional[str]
    variable_bound_mode: str
    matrix_value_mode: str
    vector_sum_mode: str
    cupdlpx_python_pre_rescale: bool
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class _LPProblem:
    rows: Any
    cols: Any
    costs: Any
    warm_start_primal: Any
    rows_result_pack: np.ndarray
    cols_result_pack: np.ndarray
    n_source: int
    n_target: int
    n_vars: int
    n_constraints: int
    active_support_size: int
    use_incremental: bool
    param_pack_time: float
    cost_build_time: float


@dataclass(frozen=True)
class _LPBackendPayload:
    A_csc: Any
    b_eq: Any
    lb: Any
    ub: Any
    costs: Any
    device_problem_data: Optional[Dict[str, Any]]
    matrix_build_time: float
    rhs_bounds_time: float


def _resolve_backend_options(solver: Any) -> _LPBackendOptions:
    """
    CN: 解析并验证 restricted-LP backend 选项。
    EN: Resolve and validate restricted-LP backend options.
    """
    kwargs = dict(getattr(solver, "_lp_solver_kwargs", {}))
    use_gpu_pipeline = bool(kwargs.pop("use_gpu_lp_pipeline", False))
    gpu_device = kwargs.pop("gpu_pipeline_device", None)
    variable_bound_mode = str(kwargs.pop("variable_bound_mode", "explicit")).strip().lower()
    if variable_bound_mode not in {"explicit", "constant"}:
        raise ValueError("variable_bound_mode must be one of: explicit, constant")
    matrix_value_mode = str(kwargs.pop("matrix_value_mode", "explicit")).strip().lower()
    if matrix_value_mode not in {"explicit", "implicit_aty", "implicit_ax", "implicit_both"}:
        raise ValueError(
            "matrix_value_mode must be one of: explicit, implicit_aty, implicit_ax, implicit_both"
        )
    if (
        matrix_value_mode != "explicit"
        and not use_gpu_pipeline
        and not isinstance(solver.lp_solver, RestrictedOTPDLP)
    ):
        raise ValueError("implicit matrix_value_mode requires GPU device-CSR LP pipeline.")
    vector_sum_mode = str(kwargs.get("vector_sum_mode", "resident_ones")).strip().lower()
    if vector_sum_mode not in {"resident_ones", "direct_reduce"}:
        raise ValueError("vector_sum_mode must be one of: resident_ones, direct_reduce")
    kwargs["vector_sum_mode"] = vector_sum_mode
    pre_rescale = _parse_bool_flag(
        kwargs.get("cupdlpx_python_pre_rescale", True), default=True
    )
    kwargs["cupdlpx_python_pre_rescale"] = pre_rescale
    return _LPBackendOptions(
        use_gpu_pipeline=use_gpu_pipeline,
        gpu_device=gpu_device,
        variable_bound_mode=variable_bound_mode,
        matrix_value_mode=matrix_value_mode,
        vector_sum_mode=vector_sum_mode,
        cupdlpx_python_pre_rescale=pre_rescale,
        kwargs=kwargs,
    )


def _assemble_lp_problem(
    solver: Any,
    lvl_s: Any,
    lvl_t: Any,
    options: _LPBackendOptions,
    *,
    trace_collector: Optional[Any],
    trace_prefix: str,
    components: Dict[str, float],
    diag: Dict[str, Any],
) -> _LPProblem:
    """
    CN: 从 active support 或完整笛卡尔积装配 restricted-OT 变量与成本。
    EN: Assemble restricted-OT variables and costs from active support or the full Cartesian product.
    """
    use_incremental = solver.active_support is not None and solver.active_support.size > 0
    n_source, n_target = len(lvl_s.points), len(lvl_t.points)
    started = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.param_pack"):
        if use_incremental:
            payload = solver.active_support.lp_payload(on_cpu=not options.use_gpu_pipeline)
            rows, cols = payload.rows, payload.cols
            costs, warm_start_primal = payload.costs, payload.primal
        else:
            warm_start_primal = None
            rows = cols = costs = None
    param_pack_time = time.perf_counter() - started
    if param_pack_time > 0.0:
        components[COMPONENT_LP_PARAM_PACK] = param_pack_time

    cost_build_time = 0.0
    if not use_incremental:
        started = time.perf_counter()
        with _timed_trace_span(trace_collector, f"{trace_prefix}.cost_build"):
            rows, cols = np.indices((n_source, n_target))
            rows, cols = rows.ravel(), cols.ravel()
            cost_type = _normalize_cost_type_name(
                getattr(solver, "_cost_type", "l2^2")
            )
            if cost_type == "lowrank":
                F, G, a, b = (
                    lvl_s.points,
                    lvl_t.points,
                    lvl_s.cost_vec,
                    lvl_t.cost_vec,
                )
                costs = (
                    a[:, None]
                    + b[None, :]
                    - float(getattr(solver, "_cost_dot_scale", 1.0)) * (F @ G.T)
                ).ravel()
            elif cost_type == "l1":
                costs = cdist(lvl_s.points, lvl_t.points, "cityblock").ravel()
            elif cost_type == "linf":
                costs = cdist(lvl_s.points, lvl_t.points, "chebyshev").ravel()
            elif cost_type == "l2":
                costs = cdist(lvl_s.points, lvl_t.points, "euclidean").ravel()
            else:
                costs = cdist(lvl_s.points, lvl_t.points, "sqeuclidean").ravel()
        cost_build_time = time.perf_counter() - started
        diag["cost_build_time"] = float(cost_build_time)
        if cost_build_time > 0.0:
            components["lp_cost_build"] = float(cost_build_time)

    n_vars = _array_length_1d(costs)
    return _LPProblem(
        rows=rows,
        cols=cols,
        costs=costs,
        warm_start_primal=warm_start_primal,
        rows_result_pack=_to_numpy_1d(rows, np.int32),
        cols_result_pack=_to_numpy_1d(cols, np.int32),
        n_source=n_source,
        n_target=n_target,
        n_vars=n_vars,
        n_constraints=n_source + n_target,
        active_support_size=(
            int(solver.active_support.size)
            if solver.active_support is not None
            else int(n_vars)
        ),
        use_incremental=use_incremental,
        param_pack_time=param_pack_time,
        cost_build_time=cost_build_time,
    )


def _assemble_backend_payload(
    solver: Any,
    lvl_s: Any,
    lvl_t: Any,
    problem: _LPProblem,
    options: _LPBackendOptions,
    *,
    trace_collector: Optional[Any],
    trace_prefix: str,
    components: Dict[str, float],
    diag: Dict[str, Any],
) -> _LPBackendPayload:
    """
    CN: 将二部图问题装配为 CPU CSC 或 GPU device-CSR backend payload。
    EN: Assemble the bipartite problem into a CPU CSC or GPU device-CSR backend payload.
    """
    device_problem_data = None
    started = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.matrix_build"):
        if options.use_gpu_pipeline:
            support_order_cache = None
            if problem.use_incremental and solver.active_support is not None:
                support_order_cache = solver.active_support.get_order_cache(
                    n_source=problem.n_source,
                    n_target=problem.n_target,
                    device=options.gpu_device or "cuda",
                )
            device_problem_data = _build_device_ot_csr(
                rows=problem.rows,
                cols=problem.cols,
                c=problem.costs,
                masses_s=lvl_s.masses,
                masses_t=lvl_t.masses,
                n_source=problem.n_source,
                n_target=problem.n_target,
                requested_device=options.gpu_device,
                support_order_cache=support_order_cache,
                variable_bound_mode=options.variable_bound_mode,
                matrix_value_mode=options.matrix_value_mode,
                cupdlpx_python_pre_rescale=options.cupdlpx_python_pre_rescale,
            )
            A_csc = None
            diag["lp_matrix_format"] = "device_csr"
            diag["lp_device"] = str(device_problem_data["device"])
            diag["lp_order_cache_used"] = bool(
                device_problem_data.get("order_cache_used", False)
            )
            diag["variable_bound_mode"] = str(
                device_problem_data.get(
                    "variable_bound_mode", options.variable_bound_mode
                )
            )
            diag["matrix_value_mode"] = str(
                device_problem_data.get("matrix_value_mode", options.matrix_value_mode)
            )
            if support_order_cache is not None and solver.active_support is not None:
                solver.active_support.clear_order_cache()
        else:
            data = np.ones(2 * problem.n_vars, dtype=np.float32)
            row_indices = np.empty(2 * problem.n_vars, dtype=np.int32)
            row_indices[0::2] = problem.rows
            row_indices[1::2] = problem.n_source + problem.cols
            indptr = np.arange(0, 2 * problem.n_vars + 1, 2, dtype=np.int32)
            A_csc = csc_matrix(
                (data, row_indices, indptr),
                shape=(problem.n_constraints, problem.n_vars),
            )
            diag["lp_matrix_format"] = "csc"
    matrix_build_time = time.perf_counter() - started
    diag["matrix_build_time"] = float(matrix_build_time)
    if matrix_build_time > 0.0:
        components["lp_matrix_build"] = float(matrix_build_time)

    started = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.rhs_bounds_pack"):
        if options.use_gpu_pipeline and device_problem_data is not None:
            b_eq = device_problem_data["rhs"]
            lb = device_problem_data["lb"]
            ub = device_problem_data["ub"]
            backend_costs = device_problem_data["c"]
        else:
            b_eq = np.concatenate([lvl_s.masses, lvl_t.masses])
            lb = np.zeros(problem.n_vars)
            ub = np.full(problem.n_vars, np.inf)
            backend_costs = problem.costs
    rhs_bounds_time = time.perf_counter() - started
    if rhs_bounds_time > 0.0:
        components["lp_rhs_bounds_pack"] = float(rhs_bounds_time)
    return _LPBackendPayload(
        A_csc=A_csc,
        b_eq=b_eq,
        lb=lb,
        ub=ub,
        costs=backend_costs,
        device_problem_data=device_problem_data,
        matrix_build_time=matrix_build_time,
        rhs_bounds_time=rhs_bounds_time,
    )


def _dispatch_torch_backend(
    solver: Any,
    lvl_s: Any,
    lvl_t: Any,
    problem: _LPProblem,
    *,
    tolerance: Any,
    warm_start_dual: Any,
    verbose: Any,
    components: Dict[str, float],
    diag: Dict[str, Any],
) -> Any:
    """
    CN: 将二部图端点直接交给 portable Torch restricted-OT backend。
    EN: Dispatch bipartite endpoints directly to the portable Torch restricted-OT backend.
    """
    started = time.perf_counter()
    result = solver.lp_solver.solve_restricted(
        rows=problem.rows,
        cols=problem.cols,
        costs=problem.costs,
        source_mass=lvl_s.masses,
        target_mass=lvl_t.masses,
        warm_start_primal=problem.warm_start_primal,
        warm_start_dual=warm_start_dual,
        tolerance=tolerance,
        verbose=bool(verbose),
    )
    duration = float(time.perf_counter() - started)
    components["lp_backend_solve"] = duration
    diag.update(
        {
            "lp_matrix_format": "torch_endpoints",
            "lp_device": str(solver.lp_solver.device),
            "backend_solve_wall_time": duration,
            "use_incremental": problem.use_incremental,
            "n_vars": problem.n_vars,
            "n_constraints": problem.n_constraints,
            "active_support_size": problem.active_support_size,
        }
    )
    for key, value in dict(result.solver_diag or {}).items():
        diag[f"solver_{key}"] = value
    return result


def _dispatch_general_backend(
    solver: Any,
    problem: _LPProblem,
    payload: _LPBackendPayload,
    options: _LPBackendOptions,
    *,
    tolerance: Any,
    warm_start_dual: Any,
    verbose: Any,
    trace_collector: Optional[Any],
    trace_prefix: str,
    backend_trace_args: Dict[str, Any],
    diag: Dict[str, Any],
) -> tuple[Any, float]:
    """
    CN: 调用通用 LP dispatch，并记录 backend 显存边界。
    EN: Call the general LP dispatch and record backend memory boundaries.
    """
    started = time.perf_counter()
    solve_memory_tracker = current_solve_memory_tracker()
    lp_mem_tracker = DriverMemoryTracker(
        device=_cuda_device_index(options.gpu_device),
        enabled=solve_memory_tracker is not None,
    )
    backend_trace_args["lp_entry_used_mem_mib"] = lp_mem_tracker.current_mib
    memory_metadata = {
        "trace_prefix": str(trace_prefix),
        "n_source": problem.n_source,
        "n_target": problem.n_target,
        "n_vars": problem.n_vars,
        "n_constraints": problem.n_constraints,
        "lp_matrix_format": diag.get("lp_matrix_format"),
        "lp_device": diag.get("lp_device"),
        "variable_bound_mode": options.variable_bound_mode,
        "matrix_value_mode": options.matrix_value_mode,
        "vector_sum_mode": options.vector_sum_mode,
        "cupdlpx_python_pre_rescale": options.cupdlpx_python_pre_rescale,
    }
    lp_memory_entry = (
        None
        if solve_memory_tracker is None
        else solve_memory_tracker.begin_lp(metadata=memory_metadata)
    )
    with _timed_trace_span(
        trace_collector,
        f"{trace_prefix}.backend_total",
        args=backend_trace_args,
    ):
        result = dispatch_lp_solve(
            solver.lp_solver,
            c=payload.costs,
            A_csc=payload.A_csc,
            b_eq=payload.b_eq,
            lb=payload.lb,
            ub=payload.ub,
            n_eqs=problem.n_constraints,
            warm_start_primal=problem.warm_start_primal,
            warm_start_dual=warm_start_dual,
            tolerance=tolerance,
            verbose=verbose,
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.backend",
            trace_args=backend_trace_args,
            device_problem_data=payload.device_problem_data,
            **options.kwargs,
        )
        solver_diag = getattr(result, "solver_diag", None)
        solver_diag = solver_diag if isinstance(solver_diag, dict) else {}
        reported_peak = solver_diag.get(
            "lp_backend_abs_peak_mem_mib", getattr(result, "peak_mem", None)
        )
        try:
            reported_peak_float = (
                None if reported_peak is None else float(reported_peak)
            )
        except (TypeError, ValueError):
            reported_peak_float = None
        reported_delta = solver_diag.get("lp_backend_delta_peak_mem_mib")
        try:
            reported_delta_float = (
                None if reported_delta is None else float(reported_delta)
            )
        except (TypeError, ValueError):
            reported_delta_float = None
        entry_memory = backend_trace_args["lp_entry_used_mem_mib"]
        if (
            reported_delta_float is None
            and reported_peak_float is not None
            and entry_memory is not None
        ):
            reported_delta_float = max(
                0.0, float(reported_peak_float) - float(entry_memory)
            )
        backend_trace_args["lp_backend_reported_peak_mem_mib"] = reported_peak_float
        backend_trace_args["lp_backend_abs_peak_mem_mib"] = reported_peak_float
        backend_trace_args["lp_delta_peak_mem_mib"] = reported_delta_float
        diag["lp_backend_abs_peak_mem_mib"] = reported_peak_float
        diag["lp_backend_entry_mem_mib"] = entry_memory
        diag["lp_backend_delta_peak_mem_mib"] = reported_delta_float
        diag["lp_backend_peak_mem_mib"] = reported_delta_float
        if solve_memory_tracker is not None and lp_memory_entry is not None:
            solve_memory_tracker.finish_lp(
                lp_memory_entry, delta_peak_mib=reported_delta_float
            )
    duration = time.perf_counter() - started
    diag["backend_solve_wall_time"] = float(duration)
    return result, duration


def _extract_backend_statistics(
    result: Any,
    *,
    trace_collector: Optional[Any],
    trace_prefix: str,
    components: Dict[str, float],
    diag: Dict[str, Any],
) -> float:
    """
    CN: 从 backend 结果中提取统一诊断字段和细分耗时。
    EN: Extract normalized diagnostics and component timings from a backend result.
    """
    started = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.diag_extract"):
        solver_diag = getattr(result, "solver_diag", None)
        if isinstance(solver_diag, dict):
            for key, value in solver_diag.items():
                diag[f"solver_{key}"] = value
            for key in (
                "matrix_value_mode",
                "implicit_ax_enabled",
                "implicit_aty_enabled",
                "implicit_ax_agg",
                "implicit_ax_row0_unique_p50",
                "implicit_ax_row1_unique_p50",
            ):
                if solver_diag.get(key) is not None:
                    diag[key] = solver_diag[key]
            timing_fields = (
                ("construct_time", "lp_backend_construct"),
                ("load_data_time", COMPONENT_LP_LOAD_DATA),
                ("warm_start_time", COMPONENT_LP_WARM_START),
                ("get_solution_time", COMPONENT_LP_GET_SOLUTION),
                ("native_solve_wall_time", "lp_backend_solve"),
            )
            for source, target in timing_fields:
                value = float(solver_diag.get(source, 0.0) or 0.0)
                if value > 0.0:
                    components[target] = value
    duration = time.perf_counter() - started
    if duration > 0.0:
        components[COMPONENT_LP_DIAG_EXTRACT] = float(duration)
    return duration


def _package_lp_result(
    result: Any,
    problem: _LPProblem,
    *,
    trace_collector: Optional[Any],
    trace_prefix: str,
    components: Dict[str, float],
    record_timing: bool,
) -> tuple[Any, Any, float]:
    """
    CN: 将 backend 向量封装为 SciPy primal coupling 与 NumPy dual potentials。
    EN: Package backend vectors as a SciPy primal coupling and NumPy dual potentials.
    """
    if not result.success:
        return None, None, 0.0
    started = time.perf_counter()
    span = (
        _timed_trace_span(trace_collector, f"{trace_prefix}.result_pack")
        if record_timing
        else nullcontext()
    )
    with span:
        result_x_host = _to_numpy_1d(result.x, np.float64)
        primal = csc_matrix(
            (result_x_host, (problem.rows_result_pack, problem.cols_result_pack)),
            shape=(problem.n_source, problem.n_target),
        )
        dual = None if result.y is None else _to_numpy_1d(result.y, np.float64)
    duration = time.perf_counter() - started if record_timing else 0.0
    if duration > 0.0:
        components[COMPONENT_LP_RESULT_PACK] = float(duration)
    return primal, dual, duration


def _finalize_lp_statistics(
    problem: _LPProblem,
    payload: _LPBackendPayload,
    *,
    backend_time: float,
    diag_extract_time: float,
    result_pack_time: float,
    components: Dict[str, float],
    diag: Dict[str, Any],
) -> None:
    """
    CN: 计算未归类 wrapper 开销并补齐稳定的结构诊断字段。
    EN: Compute unclassified wrapper overhead and complete stable structural diagnostics.
    """
    accounted = float(sum(components.values()))
    total = (
        problem.param_pack_time
        + problem.cost_build_time
        + payload.matrix_build_time
        + payload.rhs_bounds_time
        + backend_time
        + diag_extract_time
        + result_pack_time
    )
    residual = total - accounted
    if residual > 1e-9:
        components["lp_wrapper_overhead"] = float(residual)
    diag.update(
        {
            "use_incremental": problem.use_incremental,
            "n_vars": problem.n_vars,
            "n_constraints": problem.n_constraints,
            "active_support_size": problem.active_support_size,
        }
    )


def solve_lp(
    solver,
    lvl_s,
    lvl_t,
    tolerance,
    warm_start_dual=None,
    verbose=0,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_args: Optional[Dict[str, Any]] = None,
):
    """
    CN: 按装配、dispatch、统计和结果封装四个步骤求解当前 restricted OT。
    EN: Solve the current restricted OT through separate assembly, dispatch, statistics, and result-packaging steps.
    """
    components: Dict[str, float] = {}
    diag: Dict[str, Any] = {}
    options = _resolve_backend_options(solver)
    problem = _assemble_lp_problem(
        solver,
        lvl_s,
        lvl_t,
        options,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        components=components,
        diag=diag,
    )
    backend_trace_args = _build_lp_trace_args(
        trace_args,
        n_source=problem.n_source,
        n_target=problem.n_target,
        n_vars=problem.n_vars,
        n_constraints=problem.n_constraints,
        use_incremental=problem.use_incremental,
        use_gpu_lp_pipeline=options.use_gpu_pipeline,
        gpu_pipeline_device=options.gpu_device,
        warm_start_dual=warm_start_dual,
        active_support_size=problem.active_support_size,
    )
    backend_trace_args["vector_sum_mode"] = options.vector_sum_mode
    backend_trace_args["cupdlpx_python_pre_rescale"] = (
        options.cupdlpx_python_pre_rescale
    )

    if isinstance(solver.lp_solver, RestrictedOTPDLP):
        result = _dispatch_torch_backend(
            solver,
            lvl_s,
            lvl_t,
            problem,
            tolerance=tolerance,
            warm_start_dual=warm_start_dual,
            verbose=verbose,
            components=components,
            diag=diag,
        )
        primal, dual, _ = _package_lp_result(
            result,
            problem,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            components=components,
            record_timing=False,
        )
        return primal, dual, result, {"components": components, "diag": diag}

    payload = _assemble_backend_payload(
        solver,
        lvl_s,
        lvl_t,
        problem,
        options,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        components=components,
        diag=diag,
    )
    result, backend_time = _dispatch_general_backend(
        solver,
        problem,
        payload,
        options,
        tolerance=tolerance,
        warm_start_dual=warm_start_dual,
        verbose=verbose,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        backend_trace_args=backend_trace_args,
        diag=diag,
    )
    diag_extract_time = _extract_backend_statistics(
        result,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        components=components,
        diag=diag,
    )
    primal, dual, result_pack_time = _package_lp_result(
        result,
        problem,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        components=components,
        record_timing=True,
    )
    _finalize_lp_statistics(
        problem,
        payload,
        backend_time=backend_time,
        diag_extract_time=diag_extract_time,
        result_pack_time=result_pack_time,
        components=components,
        diag=diag,
    )
    return primal, dual, result, {"components": components, "diag": diag}
