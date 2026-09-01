from __future__ import annotations

import importlib
import math
import time
from typing import Any, Dict

import numpy as np
import torch

from hello_ot.kernels.scan_contract import (
    plan_resident_side,
    plan_stream_chunk_rows,
    reusable_cuda_memory_bytes,
)

_CUDA_CPP_DUAL_FEASIBILITY_EXT: Any = None
_CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR: str | None = None
_CUDA_CPP_DUAL_FEASIBILITY_EXT_MODULE = (
    "hello_ot._native.inner_product_scan.hierot_inner_product_scan_ext"
)
_STREAMING_RAW_QUERY_TILE = 1024
_STREAMING_RAW_DB_TILE = 8192


def _load_cuda_cpp_dual_feasibility_extension() -> Any:
    global _CUDA_CPP_DUAL_FEASIBILITY_EXT, _CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR
    if _CUDA_CPP_DUAL_FEASIBILITY_EXT is not None:
        return _CUDA_CPP_DUAL_FEASIBILITY_EXT
    if _CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR is not None:
        raise RuntimeError(_CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR)
    try:
        _CUDA_CPP_DUAL_FEASIBILITY_EXT = importlib.import_module(_CUDA_CPP_DUAL_FEASIBILITY_EXT_MODULE)
        return _CUDA_CPP_DUAL_FEASIBILITY_EXT
    except Exception as exc:  # noqa: BLE001
        _CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR = (
            f"Failed to import installed CUDA C++ dual feasibility extension "
            f"{_CUDA_CPP_DUAL_FEASIBILITY_EXT_MODULE!r}. Reinstall hello-ot so the extension is built. "
            f"Original error: {exc!r}"
        )
        raise RuntimeError(_CUDA_CPP_DUAL_FEASIBILITY_EXT_ERROR) from exc


def _cuda_float32_tensor(data: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(data):
        return data.to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(data, dtype=torch.float32, device=device).contiguous()


def _cuda_float64_tensor(data: Any, device: torch.device) -> torch.Tensor:
    """
    CN: 将 scalar cost/dual 数据规范化为 CUDA float64 tensor。
    EN: Normalize scalar cost/dual data to a CUDA float64 tensor.
    """
    if torch.is_tensor(data):
        return data.to(device=device, dtype=torch.float64).contiguous()
    return torch.as_tensor(data, dtype=torch.float64, device=device).contiguous()


def _data_shape(data: Any) -> tuple[int, ...]:
    if torch.is_tensor(data):
        return tuple(int(v) for v in data.shape)
    return tuple(int(v) for v in np.shape(data))


def _row_slice(data: Any, start: int, end: int) -> Any:
    return data[int(start) : int(end)]


def _cuda_device_for_lowrank(source_F: Any, target_G: Any, gpu_id: int | None) -> torch.device:
    if torch.is_tensor(source_F) and source_F.is_cuda:
        return source_F.device
    if torch.is_tensor(target_G) and target_G.is_cuda:
        return target_G.device
    return torch.device(f"cuda:{0 if gpu_id is None else int(gpu_id)}")


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return 0 if device.index is None else int(device.index)


def cuda_cpp_lowrank_dual_feasibility_infeasibility_streaming_raw(
    source_F: Any,
    target_G: Any,
    source_cost_vec: Any,
    target_cost_vec: Any,
    dual_uv: Any,
    *,
    gpu_id: int | None = None,
    diagnostics: Dict[str, Any] | None = None,
    dot_scale: float = 1.0,
) -> float:
    """
    CN: 使用 raw feature/cost/dual 的 CUDA C++ streaming 扫描，避免同时构造 source/target augmented feature。
    CN: 双线性代价约定为 c_ij = source_cost_i + target_cost_j - dot_scale * source_F_i @ target_G_j。
    EN: Use CUDA C++ streaming over raw feature/cost/dual and avoid building both source/target augmented features.
    EN: The bilinear cost convention is c_ij = source_cost_i + target_cost_j - dot_scale * source_F_i @ target_G_j.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA C++ lowrank dual feasibility requires CUDA to be available.")
    source_shape = _data_shape(source_F)
    target_shape = _data_shape(target_G)
    if len(source_shape) != 2 or len(target_shape) != 2:
        raise ValueError("source_F and target_G must be 2D for lowrank dual feasibility check.")
    n_source, source_dim = int(source_shape[0]), int(source_shape[1])
    n_target, target_dim = int(target_shape[0]), int(target_shape[1])
    if source_dim != target_dim:
        raise ValueError("source_F and target_G dimensions must match for lowrank dual feasibility check.")
    if n_source <= 0 or n_target <= 0:
        return 0.0
    dual_shape = _data_shape(dual_uv)
    dual_numel = int(np.prod(dual_shape)) if dual_shape else 1
    if dual_numel != n_source + n_target:
        raise ValueError("dual_uv length must match n_source + n_target for lowrank dual feasibility check.")

    device = _cuda_device_for_lowrank(source_F, target_G, gpu_id)
    device_index = _device_index(device)
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device_index)
    memory_plan = plan_resident_side(
        source_count=int(n_source),
        target_count=int(n_target),
        feature_dim=int(source_dim),
        topk_bucket=1,
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    source_is_resident = memory_plan.resident_side == "source"
    if source_is_resident:
        resident_feat_data = source_F
        resident_cost_data = source_cost_vec
        resident_dual_data = _row_slice(dual_uv, 0, n_source)
        stream_feat_data = target_G
        stream_cost_data = target_cost_vec
        stream_dual_data = _row_slice(dual_uv, n_source, n_source + n_target)
        resident_side = "source"
    else:
        resident_feat_data = target_G
        resident_cost_data = target_cost_vec
        resident_dual_data = _row_slice(dual_uv, n_source, n_source + n_target)
        stream_feat_data = source_F
        stream_cost_data = source_cost_vec
        stream_dual_data = _row_slice(dual_uv, 0, n_source)
        resident_side = "target"

    if diagnostics is not None:
        diagnostics.update(
            {
                "source_F_is_cuda": bool(torch.is_tensor(source_F) and source_F.is_cuda),
                "target_G_is_cuda": bool(torch.is_tensor(target_G) and target_G.is_cuda),
                "resident_side": str(resident_side),
            }
        )

    resident_feat = _cuda_float32_tensor(resident_feat_data, device)
    resident_cost = _cuda_float64_tensor(resident_cost_data, device).view(-1)
    resident_dual = _cuda_float64_tensor(resident_dual_data, device).view(-1)
    n_stream = n_target if source_is_resident else n_source
    full_feature_rows = max(int(n_source), int(n_target))
    chunk_rows = plan_stream_chunk_rows(
        memory_plan,
        streamed_count=int(n_stream),
        feature_dim=int(source_dim),
        max_rows=min(int(n_stream), 65536),
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "chunk_rows": int(chunk_rows),
                "full_feature_rows": int(full_feature_rows),
                "full_feature_mem_mib": float(full_feature_rows) * float(source_dim) * 4.0 / float(1024 * 1024),
            }
        )
    total_stats = torch.zeros((5,), dtype=torch.float64, device=device)
    ext = _load_cuda_cpp_dual_feasibility_extension()

    for chunk_start in range(0, n_stream, chunk_rows):
        chunk_end = min(n_stream, chunk_start + chunk_rows)
        stream_feat = _cuda_float32_tensor(_row_slice(stream_feat_data, chunk_start, chunk_end), device)
        stream_cost = _cuda_float64_tensor(_row_slice(stream_cost_data, chunk_start, chunk_end), device).view(-1)
        stream_dual = _cuda_float64_tensor(_row_slice(stream_dual_data, chunk_start, chunk_end), device).view(-1)
        stats = ext.raw_lowrank_stats_only(
            stream_feat,
            resident_feat,
            stream_cost,
            resident_cost,
            stream_dual,
            resident_dual,
            int(_STREAMING_RAW_QUERY_TILE),
            int(_STREAMING_RAW_DB_TILE),
            float(dot_scale),
        )
        total_stats.add_(stats)
        del stats, stream_feat, stream_cost, stream_dual

    stats_cpu = total_stats.detach().cpu().double()
    numerator_sq = float(stats_cpu[0].item())
    denom_sq = float(stats_cpu[1].item())
    positive_count = int(round(float(stats_cpu[2].item())))
    max_violation = float(stats_cpu[3].item())
    cost_linf = float(stats_cpu[4].item())
    if diagnostics is not None:
        diagnostics["dual_feasibility_num_sq"] = numerator_sq
        diagnostics["dual_feasibility_den_sq"] = denom_sq
        diagnostics["dual_feasibility_positive_count"] = int(positive_count)
        diagnostics["dual_feasibility_max_violation"] = float(max_violation)
        diagnostics["dual_feasibility_cost_linf"] = float(cost_linf)
        diagnostics["relative_linf_dual_feasibility"] = float(max_violation) / (1.0 + float(cost_linf))
        diagnostics["scan_backend"] = "custom"
        diagnostics["score_family"] = "inner_product"
        diagnostics["resident_bytes"] = int(memory_plan.resident_bytes)
        diagnostics["certificate_mode"] = "l2_and_linf"
        diagnostics["pairwise_passes"] = 1
    denom = 1.0 + math.sqrt(max(denom_sq, 0.0))
    return float(math.sqrt(max(numerator_sq, 0.0)) / denom)


def lowrank_dual_feasibility_infeasibility(
    source_F: Any,
    target_G: Any,
    source_cost_vec: Any,
    target_cost_vec: Any,
    dual_uv: Any,
    *,
    block_elements: int = 64 * 1024 * 1024,
    gpu_id: int | None = None,
    diagnostics: Dict[str, Any] | None = None,
    dot_scale: float = 1.0,
) -> Any:
    """
    CN: 用 CUDA C++ 精确扫描 lowrank 完整问题，计算 L2-style 相对 dual feasibility 违反量。
    CN: 双线性代价约定为 c_ij = source_cost_i + target_cost_j - dot_scale * source_F_i @ target_G_j。
    EN: Use CUDA C++ to exactly scan the full lowrank problem and compute the L2-style relative dual feasibility violation.
    EN: The bilinear cost convention is c_ij = source_cost_i + target_cost_j - dot_scale * source_F_i @ target_G_j.
    """
    _ = block_elements
    return cuda_cpp_lowrank_dual_feasibility_infeasibility_streaming_raw(
        source_F,
        target_G,
        source_cost_vec,
        target_cost_vec,
        dual_uv,
        gpu_id=gpu_id,
        diagnostics=diagnostics,
        dot_scale=float(dot_scale),
    )


def check_convergence(
    history,
    criterion: str,
    tolerance: Dict[str, float],
    plateau_counter: int,
    objective_plateau_iters: int,
):
    if len(history) < 2:
        return False, plateau_counter

    is_converged = False
    if criterion == "objective":
        diff = abs(history[-2] - history[-1]) / (abs(history[-2]) + 1e-9)
        if diff < tolerance["objective"]:
            plateau_counter += 1
        else:
            plateau_counter = 0

        if plateau_counter >= int(objective_plateau_iters):
            is_converged = True

    return is_converged, plateau_counter


def _missing_dual_feasibility_error() -> RuntimeError:
    return RuntimeError(
        "convergence_criterion='dual_feasibility' requires pricing to report dual_feasibility. "
        "Use the warm-start fused lowrank feasibility pricing scan or keep record_pricing_dual_feasibility enabled for strategies that report it."
    )


def update_convergence_state(
    level_state: Dict[str, Any],
    step_pack: Dict[str, Any],
    *,
    criterion: str,
    tolerance: Dict[str, float],
    objective_plateau_iters: int,
) -> bool:
    history = level_state.get("level_obj_hist", [])
    plateau_counter = int(level_state.get("plateau_counter", 0))
    is_converged, plateau_counter = check_convergence(
        history,
        criterion,
        tolerance,
        plateau_counter,
        objective_plateau_iters,
    )
    pricing_info = step_pack.get("pricing_info")
    if not isinstance(pricing_info, dict):
        pricing_info = {}
    dual_feasibility = pricing_info.get("dual_feasibility")
    dual_feasibility_tol = float(
        step_pack.get("dual_feasibility_tol", tolerance.get("dual_feasibility", 1e-6))
    )
    if str(criterion) == "dual_feasibility" and dual_feasibility is None:
        raise _missing_dual_feasibility_error()
    dual_feasibility_passed = (
        None
        if dual_feasibility is None
        else bool(float(dual_feasibility) <= float(dual_feasibility_tol))
    )
    if str(criterion) == "dual_feasibility":
        is_converged = bool(dual_feasibility_passed)
    level_state["plateau_counter"] = plateau_counter
    level_state["_is_converged"] = is_converged
    step_pack["convergence_info"] = {
        "signed_rel_obj_change": (
            (float(history[-1]) - float(history[-2])) / (abs(float(history[-2])) + 1e-9)
        )
        if len(history) >= 2
        else None,
        "is_converged": bool(is_converged),
        "criterion": str(criterion),
        "plateau_counter": int(plateau_counter),
        "required_plateau": 1 if criterion == "objective" else None,
        "objective_tol": float(tolerance.get("objective"))
        if isinstance(tolerance, dict) and "objective" in tolerance
        else None,
        "dual_feasibility": None if dual_feasibility is None else float(dual_feasibility),
        "dual_feasibility_source": pricing_info.get("dual_feasibility_source"),
        "dual_feasibility_tol": float(dual_feasibility_tol),
        "dual_feasibility_passed": dual_feasibility_passed,
    }
    iteration_records = level_state.setdefault("iteration_records", [])
    iteration_records.append(
        {
            "inner_iter": int(level_state.get("current_iter", len(iteration_records))) + 1,
            "objective": None if not history else float(history[-1]),
            "convergence_info": dict(step_pack["convergence_info"]),
        }
    )
    return bool(is_converged)


def should_stop_iteration(_solver, level_state, _run_state, max_inner_iter: int, _step_pack) -> bool:
    if level_state.get("is_coarsest"):
        return True
    if bool(level_state.get("_is_converged", False)):
        return True
    return level_state["current_iter"] + 1 >= max_inner_iter


def should_stop_inner(problem_def, algorithm_state, level_state, step_result) -> bool:
    conv_dt = 0.0
    if not level_state.data.get("is_coarsest"):
        t_conv = time.perf_counter()
        if isinstance(step_result.data, dict):
            step_result.data["dual_feasibility_tol"] = float(
                problem_def.extra_kwargs.get("dual_feasibility_tol", 1e-6)
            )
        update_convergence_state(
            level_state.data,
            step_result.data,
            criterion=problem_def.convergence_criterion,
            tolerance=problem_def.tolerance,
            objective_plateau_iters=problem_def.objective_plateau_iters,
        )
        conv_dt = time.perf_counter() - t_conv
    t0 = time.perf_counter()
    result = should_stop_iteration(
        problem_def.solver,
        level_state.data,
        algorithm_state.run_state,
        problem_def.max_inner_iter,
        step_result.data,
    )
    return bool(result)
