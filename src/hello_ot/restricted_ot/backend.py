from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
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
from hello_ot._internal.lp.torch_restricted_ot import TorchRestrictedOTPDLP
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
    CN: 将当前 active-support OT 子问题包装成 LP，并调用底层 backend 求解。
    EN: Wrap the current active-support OT subproblem as an LP and solve it with the selected backend.

    CN: 这个函数负责三层工作：
    EN: This function is responsible for three layers of work:
    1. CN: 从当前 active support 或完整笛卡尔积中整理 LP 的列集合与 cost 向量。
       EN: Collect the LP columns and cost vector either from the current active support or from the full Cartesian product.
    2. CN: 按 CPU CSC 或 GPU device-CSR 的形式构造约束矩阵与边界向量。
       EN: Build the constraint matrix and bound vectors either as CPU CSC or GPU device-CSR.
    3. CN: 调用 dispatch 层进入具体 backend，并重新打包 primal/dual 输出。
       EN: Call the concrete backend through dispatch and repack the primal/dual outputs.
    """
    # CN: components/diag 分别汇总“阶段耗时分解”和“结构化诊断信息”，最后一起返回给上层。
    # EN: components/diag accumulate the phase timing breakdown and structured diagnostics that are returned upstream.
    components: Dict[str, float] = {}
    diag: Dict[str, Any] = {}

    # CN: use_incremental 表示当前是否已有 active support；若为真，就只在该 working set 上组装 restricted LP。
    # EN: use_incremental indicates whether an active support already exists; when true, the restricted LP is assembled only on that working set.
    use_incremental = solver.active_support is not None and solver.active_support.size > 0
    n_source, n_target = len(lvl_s.points), len(lvl_t.points)
    lp_backend_kwargs = dict(getattr(solver, "_lp_solver_kwargs", {}))

    # CN: GPU pipeline 打开时，LP 将直接以 device-CSR 形式传给 backend；否则回落到 CPU CSC。
    # EN: When the GPU pipeline is enabled, the LP is passed to the backend as device-CSR; otherwise it falls back to CPU CSC.
    use_gpu_lp_pipeline = bool(lp_backend_kwargs.pop("use_gpu_lp_pipeline", False))
    gpu_pipeline_device = lp_backend_kwargs.pop("gpu_pipeline_device", None)
    variable_bound_mode = str(lp_backend_kwargs.pop("variable_bound_mode", "explicit")).strip().lower()
    if variable_bound_mode not in {"explicit", "constant"}:
        raise ValueError("variable_bound_mode must be one of: explicit, constant")
    matrix_value_mode = str(lp_backend_kwargs.pop("matrix_value_mode", "explicit")).strip().lower()
    if matrix_value_mode not in {"explicit", "implicit_aty", "implicit_ax", "implicit_both"}:
        raise ValueError("matrix_value_mode must be one of: explicit, implicit_aty, implicit_ax, implicit_both")
    if (
        matrix_value_mode != "explicit"
        and not use_gpu_lp_pipeline
        and not isinstance(solver.lp_solver, TorchRestrictedOTPDLP)
    ):
        raise ValueError("implicit matrix_value_mode requires GPU device-CSR LP pipeline.")
    vector_sum_mode = str(lp_backend_kwargs.get("vector_sum_mode", "resident_ones")).strip().lower()
    if vector_sum_mode not in {"resident_ones", "direct_reduce"}:
        raise ValueError("vector_sum_mode must be one of: resident_ones, direct_reduce")
    lp_backend_kwargs["vector_sum_mode"] = vector_sum_mode
    cupdlpx_python_pre_rescale = _parse_bool_flag(lp_backend_kwargs.get("cupdlpx_python_pre_rescale", True), default=True)
    lp_backend_kwargs["cupdlpx_python_pre_rescale"] = cupdlpx_python_pre_rescale
    device_problem_data = None
    support_order_cache = None

    # CN: 第一阶段是“参数整理”：
    # EN: The first phase is "parameter packing":
    # CN: 如果已有 active support，就直接从其中取出 restricted LP 的 rows/cols/c/x_prev；
    # EN: if an active support already exists, read rows/cols/c/x_prev directly from it;
    # CN: 否则仅记录 warm_start_primal 为空，稍后会走完整 Cartesian product 的 full LP 构造。
    # EN: otherwise record that warm_start_primal is empty and later build the full LP from the Cartesian product.
    t_param_pack = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.param_pack"):
        if use_incremental:
            if solver.active_support.is_torch_backend and not use_gpu_lp_pipeline:
                active_payload = solver.active_support.export_numpy()
                rows = active_payload["rows"]
                cols = active_payload["cols"]
                c = active_payload["c_vec"]
                warm_start_primal = active_payload["x_prev"]
            else:
                rows, cols, c = solver.active_support.rows, solver.active_support.cols, solver.active_support.c_vec
                warm_start_primal = solver.active_support.x_prev
        else:
            warm_start_primal = None
    param_pack_dt = time.perf_counter() - t_param_pack
    if param_pack_dt > 0.0:
        components[COMPONENT_LP_PARAM_PACK] = param_pack_dt

    # CN: 当没有 active support 时，这里显式构造 full LP 的列集合与 cost 向量。
    # EN: When no active support exists, explicitly build the full LP column set and cost vector here.
    # CN: 这一步决定变量空间是“完整 n_source x n_target”还是“restricted support”。
    # EN: This step decides whether the variable space is the full n_source x n_target grid or a restricted support.
    if not use_incremental:
        t_cost = time.perf_counter()
        with _timed_trace_span(trace_collector, f"{trace_prefix}.cost_build"):
            rows, cols = np.indices((n_source, n_target))
            rows, cols = rows.ravel(), cols.ravel()
            cost_type = _normalize_cost_type_name(getattr(solver, "_cost_type", "l2^2"))
            if cost_type == "lowrank":
                F, G, a, b = lvl_s.points, lvl_t.points, lvl_s.cost_vec, lvl_t.cost_vec
                c = (
                    a[:, None]
                    + b[None, :]
                    - float(getattr(solver, "_cost_dot_scale", 1.0)) * (F @ G.T)
                ).ravel()
            elif cost_type == "l1":
                c = cdist(lvl_s.points, lvl_t.points, "cityblock").ravel()
            elif cost_type == "linf":
                c = cdist(lvl_s.points, lvl_t.points, "chebyshev").ravel()
            elif cost_type == "l2":
                c = cdist(lvl_s.points, lvl_t.points, "euclidean").ravel()
            else:
                c = cdist(lvl_s.points, lvl_t.points, "sqeuclidean").ravel()
        cost_build_dt = time.perf_counter() - t_cost
        diag["cost_build_time"] = float(cost_build_dt)
        if cost_build_dt > 0.0:
            components["lp_cost_build"] = float(cost_build_dt)


    # CN: 下面这些量是后续 LP backend 和 trace/diagnostics 都会共用的核心维度与列元数据。
    # EN: The quantities below are core dimensions and column metadata reused by both the backend and diagnostics/trace.
    n_vars = _array_length_1d(c)
    n_constraints = n_source + n_target
    rows_result_pack = _to_numpy_1d(rows, np.int32)
    cols_result_pack = _to_numpy_1d(cols, np.int32)
    active_support_size = int(solver.active_support.size) if solver.active_support is not None else int(n_vars)
    backend_trace_args = _build_lp_trace_args(
        trace_args,
        n_source=n_source,
        n_target=n_target,
        n_vars=n_vars,
        n_constraints=n_constraints,
        use_incremental=use_incremental,
        use_gpu_lp_pipeline=use_gpu_lp_pipeline,
        gpu_pipeline_device=gpu_pipeline_device,
        warm_start_dual=warm_start_dual,
        active_support_size=active_support_size,
    )
    backend_trace_args["vector_sum_mode"] = str(vector_sum_mode)
    backend_trace_args["cupdlpx_python_pre_rescale"] = bool(cupdlpx_python_pre_rescale)

    # CN: Torch restricted-OT backend 直接消费二部图端点，禁止为 portable 路径构造通用 CSR/CSC。
    # EN: The Torch restricted-OT backend consumes bipartite endpoints directly and never builds generic CSR/CSC.
    if isinstance(solver.lp_solver, TorchRestrictedOTPDLP):
        backend_started = time.perf_counter()
        result = solver.lp_solver.solve_restricted(
            rows=rows,
            cols=cols,
            costs=c,
            source_mass=lvl_s.masses,
            target_mass=lvl_t.masses,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            tolerance=tolerance,
            verbose=bool(verbose),
        )
        backend_duration = float(time.perf_counter() - backend_started)
        components["lp_backend_solve"] = backend_duration
        solver_diag = dict(result.solver_diag or {})
        diag.update(
            {
                "lp_matrix_format": "torch_endpoints",
                "lp_device": str(solver.lp_solver.device),
                "backend_solve_wall_time": backend_duration,
                "use_incremental": bool(use_incremental),
                "n_vars": int(n_vars),
                "n_constraints": int(n_constraints),
                "active_support_size": int(active_support_size),
            }
        )
        for key, value in solver_diag.items():
            diag[f"solver_{key}"] = value
        primal_sol = None
        dual_sol = None
        if result.success:
            result_x_host = _to_numpy_1d(result.x, np.float64)
            primal_sol = csc_matrix(
                (result_x_host, (rows_result_pack, cols_result_pack)),
                shape=(n_source, n_target),
            )
            dual_sol = None if result.y is None else _to_numpy_1d(result.y, np.float64)
        return primal_sol, dual_sol, result, {"components": components, "diag": diag}

    # CN: 第二阶段是“矩阵构造”：
    # EN: The second phase is "matrix construction":
    # CN: GPU pipeline 下把 (rows, cols, c) 直接转成 device-CSR；
    # EN: under the GPU pipeline, convert (rows, cols, c) directly into device-CSR;
    # CN: 否则构造 CPU 侧 CSC 矩阵，后续由 backend 再做加载。
    # EN: otherwise build a CPU-side CSC matrix that the backend will load later.
    t_matrix = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.matrix_build"):
        if use_gpu_lp_pipeline:
            if use_incremental and solver.active_support is not None:
                support_order_cache = solver.active_support.get_order_cache(
                    n_source=n_source,
                    n_target=n_target,
                    device=gpu_pipeline_device or "cuda",
                )
            device_problem_data = _build_device_ot_csr(
                rows=rows,
                cols=cols,
                c=c,
                masses_s=lvl_s.masses,
                masses_t=lvl_t.masses,
                n_source=n_source,
                n_target=n_target,
                requested_device=gpu_pipeline_device,
                support_order_cache=support_order_cache,
                variable_bound_mode=variable_bound_mode,
                matrix_value_mode=matrix_value_mode,
                cupdlpx_python_pre_rescale=bool(cupdlpx_python_pre_rescale),
            )
            A_csc = None
            diag["lp_matrix_format"] = "device_csr"
            diag["lp_device"] = str(device_problem_data["device"])
            diag["lp_order_cache_used"] = bool(device_problem_data.get("order_cache_used", False))
            diag["variable_bound_mode"] = str(device_problem_data.get("variable_bound_mode", variable_bound_mode))
            diag["matrix_value_mode"] = str(device_problem_data.get("matrix_value_mode", matrix_value_mode))
            if support_order_cache is not None and solver.active_support is not None:
                # CN: device CSR 已经拥有所需 row_ptr/col_ind；释放 active-support 上的排序缓存。
                # EN: The device CSR now owns row_ptr/col_ind, so release the active-support sort cache.
                solver.active_support.clear_order_cache()
                support_order_cache = None
        else:
            data = np.ones(2 * n_vars, dtype=np.float32)
            row_indices = np.empty(2 * n_vars, dtype=np.int32)
            row_indices[0::2] = rows
            row_indices[1::2] = n_source + cols
            indptr = np.arange(0, 2 * n_vars + 1, 2, dtype=np.int32)
            A_csc = csc_matrix((data, row_indices, indptr), shape=(n_constraints, n_vars))
            diag["lp_matrix_format"] = "csc"
    matrix_build_dt = time.perf_counter() - t_matrix
    diag["matrix_build_time"] = float(matrix_build_dt)
    if matrix_build_dt > 0.0:
        components["lp_matrix_build"] = float(matrix_build_dt)

    # CN: 第三阶段是整理 RHS / bounds / objective：
    # EN: The third phase packages the RHS / bounds / objective:
    # CN: GPU pipeline 下直接复用 device_problem_data 中已经在 GPU 上的向量；
    # EN: under the GPU pipeline, reuse the already-on-GPU vectors in device_problem_data;
    # CN: 否则在 CPU 上拼出 b_eq、lb、ub 与 c_backend。
    # EN: otherwise assemble b_eq, lb, ub, and c_backend on CPU.
    t_rhs = time.perf_counter()
    with _timed_trace_span(trace_collector, f"{trace_prefix}.rhs_bounds_pack"):
        if use_gpu_lp_pipeline and device_problem_data is not None:
            b_eq = device_problem_data["rhs"]
            lb = device_problem_data["lb"]
            ub = device_problem_data["ub"]
            c_backend = device_problem_data["c"]
        else:
            b_eq = np.concatenate([lvl_s.masses, lvl_t.masses])
            lb, ub = np.zeros(n_vars), np.full(n_vars, np.inf)
            c_backend = c
    rhs_bounds_dt = time.perf_counter() - t_rhs
    if rhs_bounds_dt > 0.0:
        components["lp_rhs_bounds_pack"] = float(rhs_bounds_dt)

    # CN: 第四阶段是真正进入 dispatch/backend：
    # EN: The fourth phase enters the dispatch/backend layer:
    # CN: 正式 HELLO 路径在这里调用内置 CuPDLPx backend。
    # EN: The formal HELLO path calls the bundled CuPDLPx backend here.
    t_backend = time.perf_counter()
    solve_memory_tracker = current_solve_memory_tracker()
    lp_mem_tracker = DriverMemoryTracker(
        device=_cuda_device_index(gpu_pipeline_device),
        enabled=solve_memory_tracker is not None,
    )
    backend_trace_args["lp_entry_used_mem_mib"] = lp_mem_tracker.current_mib
    memory_metadata = {
        "trace_prefix": str(trace_prefix),
        "n_source": int(n_source),
        "n_target": int(n_target),
        "n_vars": int(n_vars),
        "n_constraints": int(n_constraints),
        "lp_matrix_format": diag.get("lp_matrix_format"),
        "lp_device": diag.get("lp_device"),
        "variable_bound_mode": str(variable_bound_mode),
        "matrix_value_mode": str(matrix_value_mode),
        "vector_sum_mode": str(vector_sum_mode),
        "cupdlpx_python_pre_rescale": bool(cupdlpx_python_pre_rescale),
    }
    lp_memory_entry = (
        None
        if solve_memory_tracker is None
        else solve_memory_tracker.begin_lp(metadata=memory_metadata)
    )
    with _timed_trace_span(trace_collector, f"{trace_prefix}.backend_total", args=backend_trace_args):
        result = dispatch_lp_solve(
            solver.lp_solver,
            c=c_backend,
            A_csc=A_csc,
            b_eq=b_eq,
            lb=lb,
            ub=ub,
            n_eqs=n_constraints,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            tolerance=tolerance,
            verbose=verbose,
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.backend",
            trace_args=backend_trace_args,
            device_problem_data=device_problem_data,
            **lp_backend_kwargs,
        )
        solver_diag_for_peak = getattr(result, "solver_diag", None)
        if not isinstance(solver_diag_for_peak, dict):
            solver_diag_for_peak = {}
        reported_peak = solver_diag_for_peak.get("lp_backend_abs_peak_mem_mib", getattr(result, "peak_mem", None))
        try:
            reported_peak_float = None if reported_peak is None else float(reported_peak)
        except (TypeError, ValueError):
            reported_peak_float = None
        reported_delta = solver_diag_for_peak.get("lp_backend_delta_peak_mem_mib")
        try:
            reported_delta_float = None if reported_delta is None else float(reported_delta)
        except (TypeError, ValueError):
            reported_delta_float = None
        if reported_delta_float is None and reported_peak_float is not None and backend_trace_args["lp_entry_used_mem_mib"] is not None:
            reported_delta_float = max(0.0, float(reported_peak_float) - float(backend_trace_args["lp_entry_used_mem_mib"]))
        backend_trace_args["lp_backend_reported_peak_mem_mib"] = reported_peak_float
        backend_trace_args["lp_backend_abs_peak_mem_mib"] = reported_peak_float
        backend_trace_args["lp_delta_peak_mem_mib"] = reported_delta_float
        diag["lp_backend_abs_peak_mem_mib"] = reported_peak_float
        diag["lp_backend_entry_mem_mib"] = backend_trace_args["lp_entry_used_mem_mib"]
        diag["lp_backend_delta_peak_mem_mib"] = reported_delta_float
        diag["lp_backend_peak_mem_mib"] = reported_delta_float
        if solve_memory_tracker is not None and lp_memory_entry is not None:
            solve_memory_tracker.finish_lp(lp_memory_entry, delta_peak_mib=reported_delta_float)
    backend_total_dt = time.perf_counter() - t_backend
    diag["backend_solve_wall_time"] = float(backend_total_dt)

    # CN: 第五阶段从 backend 的 solver_diag 中提取统一诊断字段，并把已知的耗时项记入 components。
    # EN: The fifth phase extracts a normalized solver_diag from the backend result and records known timing components.
    t_diag_extract = time.perf_counter()
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
                    diag[key] = solver_diag.get(key)
            load_data_time = float(solver_diag.get("load_data_time", 0.0) or 0.0)
            warm_start_time = float(solver_diag.get("warm_start_time", 0.0) or 0.0)
            get_solution_time = float(solver_diag.get("get_solution_time", 0.0) or 0.0)
            native_solve_time = float(solver_diag.get("native_solve_wall_time", 0.0) or 0.0)
            construct_time = float(solver_diag.get("construct_time", 0.0) or 0.0)
            if construct_time > 0.0:
                components["lp_backend_construct"] = construct_time
            if load_data_time > 0.0:
                components[COMPONENT_LP_LOAD_DATA] = load_data_time
            if warm_start_time > 0.0:
                components[COMPONENT_LP_WARM_START] = warm_start_time
            if get_solution_time > 0.0:
                components[COMPONENT_LP_GET_SOLUTION] = get_solution_time
            if native_solve_time > 0.0:
                components["lp_backend_solve"] = native_solve_time
    diag_extract_dt = time.perf_counter() - t_diag_extract
    if diag_extract_dt > 0.0:
        components[COMPONENT_LP_DIAG_EXTRACT] = float(diag_extract_dt)

    # CN: 第六阶段把 backend 返回向量打包成 primal 稀疏矩阵与 dual 向量。
    # EN: The sixth phase repacks backend vectors into a sparse primal matrix and dual vector.
    primal_sol = None
    dual_sol = None
    result_pack_dt = 0.0
    if result.success:
        t_result_pack = time.perf_counter()
        with _timed_trace_span(trace_collector, f"{trace_prefix}.result_pack"):
            result_rows = rows_result_pack
            result_cols = cols_result_pack
            # CN: 轻量输出契约使用 SciPy/NumPy；device-native 结果仍留在 result 中供 GPU warm start 复用。
            # EN: The compact output contract uses SciPy/NumPy while retaining device-native results for GPU warm start.
            result_x_host = _to_numpy_1d(result.x, np.float64)
            primal_sol = csc_matrix((result_x_host, (result_rows, result_cols)), shape=(n_source, n_target))
            dual_sol = None if result.y is None else _to_numpy_1d(result.y, np.float64)
        result_pack_dt = time.perf_counter() - t_result_pack
        if result_pack_dt > 0.0:
            components[COMPONENT_LP_RESULT_PACK] = float(result_pack_dt)

    # CN: residual 表示没有被前面细分组件显式归类的 wrapper 开销。
    # EN: residual captures wrapper overhead that was not explicitly assigned to the earlier fine-grained components.
    accounted = float(sum(components.values()))
    residual = (
        param_pack_dt
        + float(diag.get("cost_build_time", 0.0))
        + matrix_build_dt
        + rhs_bounds_dt
        + backend_total_dt
        + diag_extract_dt
        + result_pack_dt
        - accounted
    )
    if residual > 1e-9:
        components["lp_wrapper_overhead"] = float(residual)
    diag["use_incremental"] = bool(use_incremental)
    diag["n_vars"] = int(n_vars)
    diag["n_constraints"] = int(n_constraints)
    diag["active_support_size"] = int(active_support_size)

    # CN: 返回 restricted-OT 的 primal、dual、backend result 与统计。
    # EN: Return restricted-OT primal, dual, backend result, and statistics.
    # CN: primal_sol/dual_sol 是上层真正消费的解，result 是 backend 原始求解结果，meta 里带组件分解与诊断信息。
    # EN: primal_sol/dual_sol are the solutions consumed upstream, result is the raw backend solve result, and meta carries the component breakdown plus diagnostics.
    return primal_sol, dual_sol, result, {"components": components, "diag": diag}
