from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MATRIX_VALUE_MODES = {"explicit", "implicit_aty", "implicit_ax", "implicit_both"}
IMPLICIT_MATRIX_VALUE_MODES = {"implicit_aty", "implicit_ax", "implicit_both"}
A_EXPLICIT_MATRIX_VALUE_MODES = {"explicit", "implicit_aty"}


def should_direct_pre_rescale_values(requested: bool, matrix_value_mode: str) -> bool:
    return bool(requested) and str(matrix_value_mode).strip().lower() in IMPLICIT_MATRIX_VALUE_MODES


def maybe_pre_rescale_device_csr_payload(
    payload: Dict[str, Any],
    *,
    requested: bool,
    row_ids: Optional["torch.Tensor"] = None,
    source: str = "builder",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        return payload, {"effective": False, "skip_reason": "torch_unavailable"}
    matrix_value_mode = str(payload.get("matrix_value_mode", "explicit")).strip().lower()
    diag: Dict[str, Any] = {
        "requested": bool(requested),
        "effective": False,
        "skip_reason": None,
        "source": str(source),
        "matrix_value_mode_in": matrix_value_mode,
    }
    if not requested:
        diag["skip_reason"] = "disabled"
        return payload, diag
    if matrix_value_mode not in IMPLICIT_MATRIX_VALUE_MODES:
        diag["skip_reason"] = "matrix_value_mode_not_implicit"
        return payload, diag
    if bool(payload.get("has_precomputed_rescaling", False)):
        diag["effective"] = True
        diag["skip_reason"] = None
        diag["already_precomputed"] = True
        return payload, diag

    variable_bound_mode = str(payload.get("variable_bound_mode", "explicit")).strip().lower()
    if variable_bound_mode == "constant":
        lb_const = float(payload["lb"])
        ub_const = float(payload["ub"])
        if lb_const != 0.0 or not np.isinf(ub_const) or ub_const < 0.0:
            diag["skip_reason"] = "non_default_constant_bounds"
            return payload, diag

    required = ("row_ptr", "col_ind", "c", "rhs")
    if not all(torch.is_tensor(payload.get(key)) and payload[key].is_cuda for key in required):
        diag["skip_reason"] = "not_cuda_tensor_input"
        return payload, diag

    t0 = time.perf_counter()
    row_ptr = payload["row_ptr"]
    col_ind = payload["col_ind"]
    c = payload["c"]
    rhs = payload["rhs"]
    n_vars = int(payload["n"])
    n_cons = int(payload["m"])
    row_counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.float64)
    col_counts = torch.bincount(col_ind.to(dtype=torch.int64), minlength=n_vars).to(dtype=torch.float64)
    eps = 1e-12
    constraint_rescaling = torch.where(row_counts < eps, torch.ones_like(row_counts), torch.sqrt(row_counts)).contiguous()
    variable_rescaling = torch.where(col_counts < eps, torch.ones_like(col_counts), torch.sqrt(col_counts)).contiguous()

    scaled = dict(payload)
    scaled["c"] = (c / variable_rescaling).contiguous()
    scaled["rhs"] = (rhs / constraint_rescaling).contiguous()
    if variable_bound_mode == "explicit":
        scaled["lb"] = (payload["lb"] * variable_rescaling).contiguous()
        scaled["ub"] = (payload["ub"] * variable_rescaling).contiguous()

    if matrix_value_mode in A_EXPLICIT_MATRIX_VALUE_MODES:
        if row_ids is None:
            row_ids = torch.repeat_interleave(
                torch.arange(n_cons, device=row_ptr.device, dtype=torch.int64),
                (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.int64),
            )
        row_ids = row_ids.to(device=row_ptr.device, dtype=torch.int64).contiguous()
        scaled["values"] = (1.0 / (constraint_rescaling[row_ids] * variable_rescaling[col_ind.to(dtype=torch.int64)])).contiguous()
    else:
        scaled.pop("values", None)

    original_objective_vector_norm = float(torch.linalg.vector_norm(c).detach().cpu().item())
    original_constraint_bound_norm = float(torch.linalg.vector_norm(rhs).detach().cpu().item())
    scaled.update(
        {
            "constraint_rescaling": constraint_rescaling,
            "variable_rescaling": variable_rescaling,
            "constraint_bound_rescaling": 1.0,
            "objective_vector_rescaling": 1.0,
            "original_objective_vector_norm": original_objective_vector_norm,
            "original_constraint_bound_norm": original_constraint_bound_norm,
            "has_precomputed_rescaling": True,
            "cupdlpx_python_pre_rescale_direct": True,
            "cupdlpx_python_pre_rescale_source": str(source),
            "cupdlpx_python_pre_rescale_matrix_value_mode_in": matrix_value_mode,
            "cupdlpx_python_pre_rescale_matrix_value_mode_out": matrix_value_mode,
            "cupdlpx_python_pre_rescale_vector_bytes": int((n_cons + n_vars) * 8),
            "cupdlpx_python_pre_rescale_time": float(time.perf_counter() - t0),
            "cupdlpx_pre_rescale_original_rhs": rhs,
        }
    )
    diag.update(
        {
            "effective": True,
            "skip_reason": None,
            "matrix_value_mode_out": matrix_value_mode,
            "time": scaled["cupdlpx_python_pre_rescale_time"],
            "vector_bytes": scaled["cupdlpx_python_pre_rescale_vector_bytes"],
        }
    )
    return scaled, diag


def combine_precomputed_rescaling_metadata(
    out: Dict[str, Any],
    payloads: list[Dict[str, Any]],
    *,
    source: str,
) -> Dict[str, Any]:
    if not payloads or not all(bool(payload.get("has_precomputed_rescaling", False)) for payload in payloads):
        return out
    device = out["device"]
    out["constraint_rescaling"] = torch.cat([payload["constraint_rescaling"] for payload in payloads], dim=0).contiguous()
    out["variable_rescaling"] = torch.cat([payload["variable_rescaling"] for payload in payloads], dim=0).contiguous()
    obj_norm_sq = sum(float(payload.get("original_objective_vector_norm", 0.0)) ** 2 for payload in payloads)
    rhs_norm_sq = sum(float(payload.get("original_constraint_bound_norm", 0.0)) ** 2 for payload in payloads)
    original_rhs_parts = [
        payload.get("cupdlpx_pre_rescale_original_rhs")
        for payload in payloads
        if torch.is_tensor(payload.get("cupdlpx_pre_rescale_original_rhs"))
    ]
    out.update(
        {
            "constraint_bound_rescaling": 1.0,
            "objective_vector_rescaling": 1.0,
            "original_objective_vector_norm": float(obj_norm_sq ** 0.5),
            "original_constraint_bound_norm": float(rhs_norm_sq ** 0.5),
            "has_precomputed_rescaling": True,
            "cupdlpx_python_pre_rescale_direct": True,
            "cupdlpx_python_pre_rescale_source": str(source),
            "cupdlpx_python_pre_rescale_matrix_value_mode_in": str(out.get("matrix_value_mode", "explicit")),
            "cupdlpx_python_pre_rescale_matrix_value_mode_out": str(out.get("matrix_value_mode", "explicit")),
            "cupdlpx_python_pre_rescale_vector_bytes": int(
                (int(out["m"]) + int(out["n"])) * 8
            ),
        }
    )
    if len(original_rhs_parts) == len(payloads):
        out["cupdlpx_pre_rescale_original_rhs"] = torch.cat(original_rhs_parts, dim=0).to(device=device).contiguous()
    return out
