from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np
from scipy import sparse as sp

from ..instrumentation.driver_memory import DriverMemoryTracker
from ..instrumentation.memory_accounting import current_memory_recorder
from .wrapper import LPSolver, SolverResult

try:
    from hello_ot._native import pycupdlpx

    CUPDLPX_AVAILABLE = True
except ImportError:
    CUPDLPX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

BoundObjectiveRescalingMode = Union[bool, Literal["auto"]]


def _trace_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class CuPDLPxSolver(LPSolver):
    """统一的 cuPDLPx wrapper。

    支持两种 LP 形式：
    - `primal`: 标准形式，变量是 primal flow
    - `dual`: 结构化 OT 的 dual 形式，变量是势能 (u, v)
    """

    supports_tree_lp_form = True
    supports_device_csr = True
    supports_approx_pruning = True
    supports_native_support_stop = True

    def __init__(
        self,
        *,
        bound_objective_rescaling: BoundObjectiveRescalingMode = "auto",
        cupdlpx_python_pre_rescale: bool = True,
    ) -> None:
        self.bound_objective_rescaling = self._normalize_bound_objective_rescaling_mode(bound_objective_rescaling)
        self.cupdlpx_python_pre_rescale = self._normalize_bool_flag(cupdlpx_python_pre_rescale, default=True)

    @staticmethod
    def _normalize_bool_flag(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (np.bool_,)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _cupdlpx_dump_on_failure_enabled() -> bool:
        enabled = str(os.environ.get("HIEROT_CUPDLPX_DUMP_ON_FAILURE", "")).strip().lower()
        return enabled in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _normalize_bound_objective_rescaling_mode(value: Any) -> BoundObjectiveRescalingMode:
        if isinstance(value, bool):
            return value
        if isinstance(value, (np.bool_,)):
            return bool(value)
        text = str(value).strip().lower()
        if text == "auto":
            return "auto"
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError("bound_objective_rescaling must be one of: False, True, 'auto'")

    @staticmethod
    def _normalize_termination_norm(value: Any) -> Literal["l2", "linf"]:
        """
        CN: 规范化 termination-only residual norm；该参数不改变 cuPDLPx 的算法启发式。
        EN: Normalize the termination-only residual norm without changing cuPDLPx algorithmic heuristics.
        """
        norm = str(value if value is not None else "l2").strip().lower()
        if norm not in {"l2", "linf"}:
            raise ValueError("termination_norm must be one of: l2, linf")
        return norm

    @staticmethod
    def _copy_if_present(mapping: Optional[Dict[str, Any]], key: str):
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
        return None

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return bool(TORCH_AVAILABLE and torch.is_tensor(value))

    @classmethod
    def _to_numpy_float32(cls, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if cls._is_torch_tensor(value):
            tensor = value.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            return tensor.to(dtype=torch.float32).contiguous().numpy()
        return np.asarray(value, dtype=np.float32)

    @classmethod
    def _to_numpy_float64(cls, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if cls._is_torch_tensor(value):
            tensor = value.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            return tensor.to(dtype=torch.float64).contiguous().numpy()
        return np.asarray(value, dtype=np.float64)

    @classmethod
    def _normalize_continuation_state(cls, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        out: Dict[str, Any] = {}
        for key in (
            "enabled",
            "reuse_step_size",
            "apply_restart_on_entry",
            "step_size",
            "primal_weight",
            "primal_weight_error_sum",
            "primal_weight_last_error",
            "best_primal_weight",
            "best_primal_dual_residual_gap",
            "previous_restart_dual_residual",
            "previous_restart_gap",
            "total_count",
            "inner_count",
        ):
            if key in payload and payload.get(key) is not None:
                out[key] = payload.get(key)
        for key in (
            "anchor_primal_unscaled",
            "anchor_dual_unscaled",
            "current_primal_unscaled",
            "current_dual_unscaled",
        ):
            if key in payload and payload.get(key) is not None:
                out[key] = cls._to_numpy_float64(payload.get(key))
        return out if out else None

    @classmethod
    def _normalize_device_problem_data(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for device LP pipeline.")
        if not isinstance(payload, dict):
            raise TypeError("device_problem_data must be a dict.")

        matrix_value_mode = str(payload.get("matrix_value_mode", "explicit")).strip().lower()
        if matrix_value_mode not in {"explicit", "implicit_aty", "implicit_ax", "implicit_both"}:
            raise ValueError("device_problem_data matrix_value_mode must be one of: explicit, implicit_aty, implicit_ax, implicit_both.")
        required = ["row_ptr", "col_ind", "c", "rhs", "lb", "ub", "m", "n", "n_eqs"]
        if matrix_value_mode in {"explicit", "implicit_aty"}:
            required.append("values")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"device_problem_data missing keys: {missing}")

        requested_device = payload.get("device")
        inferred_device = None
        for key in ("row_ptr", "col_ind", "values", "c", "rhs", "lb", "ub"):
            value = payload.get(key)
            if cls._is_torch_tensor(value):
                inferred_device = value.device
                break
        if requested_device is not None:
            device = torch.device(requested_device)
        elif inferred_device is not None:
            device = inferred_device
        else:
            device = torch.device("cuda")
        if device.type != "cuda":
            raise ValueError(f"device LP pipeline requires a CUDA device, got {device}.")

        variable_bound_mode = str(payload.get("variable_bound_mode", "explicit")).strip().lower()
        if variable_bound_mode not in {"explicit", "constant"}:
            raise ValueError("device_problem_data variable_bound_mode must be 'explicit' or 'constant'.")

        def _to_tensor(name: str, value: Any, dtype) -> "torch.Tensor":
            if cls._is_torch_tensor(value):
                tensor = value.detach()
                if tensor.device != device:
                    tensor = tensor.to(device=device)
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                return tensor
            return torch.as_tensor(value, dtype=dtype, device=device).contiguous()

        normalized = {
            "row_ptr": _to_tensor("row_ptr", payload["row_ptr"], torch.int32),
            "col_ind": _to_tensor("col_ind", payload["col_ind"], torch.int32),
            "values": None if matrix_value_mode in {"implicit_ax", "implicit_both"} else _to_tensor("values", payload["values"], torch.float64),
            "c": _to_tensor("c", payload["c"], torch.float64),
            "rhs": _to_tensor("rhs", payload["rhs"], torch.float64),
            "m": int(payload["m"]),
            "n": int(payload["n"]),
            "n_eqs": int(payload["n_eqs"]),
            "device": device,
            "variable_bound_mode": variable_bound_mode,
            "matrix_value_mode": matrix_value_mode,
        }
        if matrix_value_mode in {"implicit_ax", "implicit_both"}:
            col_degree = torch.bincount(normalized["col_ind"].to(dtype=torch.int64), minlength=int(normalized["n"]))
            if not bool(torch.all(col_degree == 2).detach().cpu().item()):
                raise ValueError("implicit_ax and implicit_both require GPU device-CSR with col_deg == 2.")
        if variable_bound_mode == "explicit":
            normalized["lb"] = _to_tensor("lb", payload["lb"], torch.float64)
            normalized["ub"] = _to_tensor("ub", payload["ub"], torch.float64)
        else:
            if cls._is_torch_tensor(payload["lb"]) or cls._is_torch_tensor(payload["ub"]):
                raise ValueError("constant device variable bounds require scalar lb and ub.")
            normalized["lb"] = float(payload["lb"])
            normalized["ub"] = float(payload["ub"])
        if bool(payload.get("has_precomputed_rescaling", False)):
            normalized["constraint_rescaling"] = _to_tensor(
                "constraint_rescaling",
                payload["constraint_rescaling"],
                torch.float64,
            )
            normalized["variable_rescaling"] = _to_tensor(
                "variable_rescaling",
                payload["variable_rescaling"],
                torch.float64,
            )
            normalized["constraint_bound_rescaling"] = float(payload.get("constraint_bound_rescaling", 1.0))
            normalized["objective_vector_rescaling"] = float(payload.get("objective_vector_rescaling", 1.0))
            normalized["original_objective_vector_norm"] = float(payload["original_objective_vector_norm"])
            normalized["original_constraint_bound_norm"] = float(payload["original_constraint_bound_norm"])
            objective_scalar = float(normalized["objective_vector_rescaling"])
            constraint_scalar = float(normalized["constraint_bound_rescaling"])
            if objective_scalar == 0.0 or constraint_scalar == 0.0:
                raise ValueError("precomputed rescaling scalar factors must be nonzero")
            if payload.get("original_objective_vector_linf_norm") is None:
                original_c = (
                    normalized["c"]
                    * normalized["variable_rescaling"]
                    / objective_scalar
                )
                normalized["original_objective_vector_linf_norm"] = float(
                    torch.linalg.vector_norm(original_c, ord=float("inf")).detach().cpu().item()
                )
            else:
                normalized["original_objective_vector_linf_norm"] = float(
                    payload["original_objective_vector_linf_norm"]
                )
            if payload.get("original_constraint_bound_linf_norm") is None:
                original_rhs = (
                    normalized["rhs"]
                    * normalized["constraint_rescaling"]
                    / constraint_scalar
                )
                normalized["original_constraint_bound_linf_norm"] = float(
                    torch.linalg.vector_norm(original_rhs, ord=float("inf")).detach().cpu().item()
                )
            else:
                normalized["original_constraint_bound_linf_norm"] = float(
                    payload["original_constraint_bound_linf_norm"]
                )
            normalized["has_precomputed_rescaling"] = True
            for key in (
                "cupdlpx_python_pre_rescale_direct",
                "cupdlpx_python_pre_rescale_source",
                "cupdlpx_python_pre_rescale_matrix_value_mode_in",
                "cupdlpx_python_pre_rescale_matrix_value_mode_out",
                "cupdlpx_python_pre_rescale_vector_bytes",
                "cupdlpx_python_pre_rescale_time",
                "cupdlpx_python_pre_rescale_builder_diag",
                "cupdlpx_python_pre_rescale_requested",
                "cupdlpx_pre_rescale_original_rhs",
            ):
                if key in payload:
                    normalized[key] = payload[key]
        return normalized

    @classmethod
    def _maybe_python_pre_rescale_device_problem(
        cls,
        normalized: Dict[str, Any],
        params: Dict[str, Any],
        *,
        requested: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        diag: Dict[str, Any] = {
            "cupdlpx_python_pre_rescale_requested": bool(requested),
            "cupdlpx_python_pre_rescale_effective": False,
            "cupdlpx_python_pre_rescale_skip_reason": None,
            "cupdlpx_python_bound_objective_rescaling_effective": False,
        }
        if not requested:
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "disabled"
            return normalized, params, diag
        if bool(normalized.get("has_precomputed_rescaling", False)):
            solve_params = dict(params)
            if cls._normalize_bool_flag(
                solve_params.get("bound_objective_rescaling", False),
                default=False,
            ):
                normalized, scalar_diag = (
                    cls._append_bound_objective_rescaling_to_precomputed_problem(
                        normalized
                    )
                )
                diag.update(scalar_diag)
            solve_params["has_pock_chambolle_alpha"] = False
            solve_params["l_inf_ruiz_iterations"] = 0
            solve_params["bound_objective_rescaling"] = False
            matrix_value_mode = str(normalized.get("matrix_value_mode", "explicit")).strip().lower()
            diag.update(
                {
                    "cupdlpx_python_pre_rescale_effective": True,
                    "cupdlpx_python_pre_rescale_skip_reason": None,
                    "cupdlpx_python_pre_rescale_time": float(normalized.get("cupdlpx_python_pre_rescale_time", 0.0) or 0.0),
                    "cupdlpx_python_pre_rescale_matrix_value_mode_in": str(
                        normalized.get("cupdlpx_python_pre_rescale_matrix_value_mode_in", matrix_value_mode)
                    ),
                    "cupdlpx_python_pre_rescale_matrix_value_mode_out": str(
                        normalized.get("cupdlpx_python_pre_rescale_matrix_value_mode_out", matrix_value_mode)
                    ),
                    "cupdlpx_python_pre_rescale_vector_bytes": int(
                        normalized.get("cupdlpx_python_pre_rescale_vector_bytes", (int(normalized["m"]) + int(normalized["n"])) * 8)
                    ),
                    "cupdlpx_python_pre_rescale_native_rescale_disabled": True,
                    "cupdlpx_python_pre_rescale_direct": bool(normalized.get("cupdlpx_python_pre_rescale_direct", False)),
                    "cupdlpx_python_pre_rescale_source": str(normalized.get("cupdlpx_python_pre_rescale_source", "input_payload")),
                    "cupdlpx_python_pre_rescale_original_objective_vector_norm": float(normalized["original_objective_vector_norm"]),
                    "cupdlpx_python_pre_rescale_original_constraint_bound_norm": float(normalized["original_constraint_bound_norm"]),
                    "cupdlpx_python_pre_rescale_original_objective_vector_linf_norm": float(normalized["original_objective_vector_linf_norm"]),
                    "cupdlpx_python_pre_rescale_original_constraint_bound_linf_norm": float(normalized["original_constraint_bound_linf_norm"]),
                }
            )
            if "cupdlpx_python_pre_rescale_builder_diag" in normalized:
                diag["cupdlpx_python_pre_rescale_builder_diag"] = normalized["cupdlpx_python_pre_rescale_builder_diag"]
            return normalized, solve_params, diag
        if not TORCH_AVAILABLE:
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "torch_unavailable"
            return normalized, params, diag
        matrix_value_mode = str(normalized.get("matrix_value_mode", "explicit")).strip().lower()
        diag["cupdlpx_python_pre_rescale_matrix_value_mode_in"] = matrix_value_mode
        if matrix_value_mode not in {"implicit_aty", "implicit_ax", "implicit_both"}:
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "matrix_value_mode_not_implicit"
            return normalized, params, diag
        if int(params.get("l_inf_ruiz_iterations", 0) or 0) != 0:
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "ruiz_rescale_enabled"
            return normalized, params, diag
        if not cls._normalize_bool_flag(params.get("has_pock_chambolle_alpha", False), default=False):
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "pock_chambolle_disabled"
            return normalized, params, diag
        if not all(cls._is_torch_tensor(normalized.get(key)) and normalized[key].is_cuda for key in ("row_ptr", "col_ind", "c", "rhs")):
            diag["cupdlpx_python_pre_rescale_skip_reason"] = "not_cuda_tensor_input"
            return normalized, params, diag
        if matrix_value_mode == "implicit_aty":
            values = normalized.get("values")
            if not (cls._is_torch_tensor(values) and bool(torch.all(values == 1.0).detach().cpu().item())):
                diag["cupdlpx_python_pre_rescale_skip_reason"] = "implicit_aty_values_not_all_ones"
                return normalized, params, diag
        variable_bound_mode = str(normalized.get("variable_bound_mode", "explicit")).strip().lower()
        if variable_bound_mode == "constant":
            lb_const = float(normalized["lb"])
            ub_const = float(normalized["ub"])
            if lb_const != 0.0 or not np.isinf(ub_const) or ub_const < 0.0:
                diag["cupdlpx_python_pre_rescale_skip_reason"] = "non_default_constant_bounds"
                return normalized, params, diag

        t0 = time.perf_counter()
        n_vars = int(normalized["n"])
        n_cons = int(normalized["m"])
        row_ptr = normalized["row_ptr"]
        col_ind = normalized["col_ind"]
        row_counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.float64)
        col_counts = torch.bincount(col_ind.to(dtype=torch.int64), minlength=n_vars).to(dtype=torch.float64)
        eps = 1e-12
        constraint_rescaling = torch.where(row_counts < eps, torch.ones_like(row_counts), torch.sqrt(row_counts)).contiguous()
        variable_rescaling = torch.where(col_counts < eps, torch.ones_like(col_counts), torch.sqrt(col_counts)).contiguous()
        c_scaled = (normalized["c"] / variable_rescaling).contiguous()
        rhs_scaled = (normalized["rhs"] / constraint_rescaling).contiguous()
        if variable_bound_mode == "explicit":
            lb_scaled = (normalized["lb"] * variable_rescaling).contiguous()
            ub_scaled = (normalized["ub"] * variable_rescaling).contiguous()
        else:
            lb_scaled = normalized["lb"]
            ub_scaled = normalized["ub"]
        original_objective_vector_norm = float(torch.linalg.vector_norm(normalized["c"]).detach().cpu().item())
        original_constraint_bound_norm = float(torch.linalg.vector_norm(normalized["rhs"]).detach().cpu().item())
        original_objective_vector_linf_norm = float(
            torch.linalg.vector_norm(normalized["c"], ord=float("inf")).detach().cpu().item()
        )
        original_constraint_bound_linf_norm = float(
            torch.linalg.vector_norm(normalized["rhs"], ord=float("inf")).detach().cpu().item()
        )

        pre_scaled = dict(normalized)
        values_scaled = None
        if matrix_value_mode == "implicit_aty":
            row_ids = torch.repeat_interleave(
                torch.arange(n_cons, device=row_ptr.device, dtype=torch.int64),
                (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.int64),
            )
            values_scaled = (1.0 / (constraint_rescaling[row_ids] * variable_rescaling[col_ind.to(dtype=torch.int64)])).contiguous()
        pre_scaled.update(
            {
                "values": values_scaled,
                "matrix_value_mode": matrix_value_mode,
                "c": c_scaled,
                "rhs": rhs_scaled,
                "lb": lb_scaled,
                "ub": ub_scaled,
                "constraint_rescaling": constraint_rescaling,
                "variable_rescaling": variable_rescaling,
                "constraint_bound_rescaling": 1.0,
                "objective_vector_rescaling": 1.0,
                "original_objective_vector_norm": original_objective_vector_norm,
                "original_constraint_bound_norm": original_constraint_bound_norm,
                "original_objective_vector_linf_norm": original_objective_vector_linf_norm,
                "original_constraint_bound_linf_norm": original_constraint_bound_linf_norm,
                "has_precomputed_rescaling": True,
                "python_pre_rescale_original_matrix_value_mode": matrix_value_mode,
            }
        )
        solve_params = dict(params)
        if cls._normalize_bool_flag(
            solve_params.get("bound_objective_rescaling", False),
            default=False,
        ):
            pre_scaled, scalar_diag = (
                cls._append_bound_objective_rescaling_to_precomputed_problem(
                    pre_scaled
                )
            )
            diag.update(scalar_diag)
        solve_params["has_pock_chambolle_alpha"] = False
        solve_params["l_inf_ruiz_iterations"] = 0
        solve_params["bound_objective_rescaling"] = False
        vector_bytes = int((n_cons + n_vars) * 8)
        diag.update(
            {
                "cupdlpx_python_pre_rescale_effective": True,
                "cupdlpx_python_pre_rescale_skip_reason": None,
                "cupdlpx_python_pre_rescale_time": float(time.perf_counter() - t0),
                "cupdlpx_python_pre_rescale_matrix_value_mode_out": matrix_value_mode,
                "cupdlpx_python_pre_rescale_vector_bytes": vector_bytes,
                "cupdlpx_python_pre_rescale_native_rescale_disabled": True,
                "cupdlpx_python_pre_rescale_original_objective_vector_norm": original_objective_vector_norm,
                "cupdlpx_python_pre_rescale_original_constraint_bound_norm": original_constraint_bound_norm,
                "cupdlpx_python_pre_rescale_original_objective_vector_linf_norm": original_objective_vector_linf_norm,
                "cupdlpx_python_pre_rescale_original_constraint_bound_linf_norm": original_constraint_bound_linf_norm,
            }
        )
        return pre_scaled, solve_params, diag

    @classmethod
    def _append_bound_objective_rescaling_to_precomputed_problem(
        cls,
        normalized: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        CN: 在 GPU degree-scaled LP 上追加 bound/objective 标量缩放，并组合解恢复元数据。
        EN: Append bound/objective scalar scaling to a GPU degree-scaled LP and compose solution-unscaling metadata.
        """
        if not bool(normalized.get("has_precomputed_rescaling", False)):
            raise ValueError(
                "bound/objective scalar scaling requires precomputed degree scaling"
            )
        if not all(
            cls._is_torch_tensor(normalized.get(key)) for key in ("c", "rhs")
        ):
            raise TypeError(
                "precomputed bound/objective scalar scaling requires torch tensors"
            )

        objective_norm = float(
            torch.linalg.vector_norm(normalized["c"]).detach().cpu().item()
        )
        constraint_bound_norm = float(
            torch.linalg.vector_norm(normalized["rhs"]).detach().cpu().item()
        )
        constraint_bound_rescaling = 1.0 / (constraint_bound_norm + 1.0)
        objective_vector_rescaling = 1.0 / (objective_norm + 1.0)

        scaled = dict(normalized)
        scaled["c"] = (
            normalized["c"] * objective_vector_rescaling
        ).contiguous()
        scaled["rhs"] = (
            normalized["rhs"] * constraint_bound_rescaling
        ).contiguous()
        if str(normalized.get("variable_bound_mode", "explicit")) == "explicit":
            scaled["lb"] = (
                normalized["lb"] * constraint_bound_rescaling
            ).contiguous()
            scaled["ub"] = (
                normalized["ub"] * constraint_bound_rescaling
            ).contiguous()
        scaled["constraint_bound_rescaling"] = float(
            normalized.get("constraint_bound_rescaling", 1.0)
        ) * constraint_bound_rescaling
        scaled["objective_vector_rescaling"] = float(
            normalized.get("objective_vector_rescaling", 1.0)
        ) * objective_vector_rescaling
        return scaled, {
            "cupdlpx_python_bound_objective_rescaling_effective": True,
            "cupdlpx_python_bound_objective_constraint_norm_before": constraint_bound_norm,
            "cupdlpx_python_bound_objective_objective_norm_before": objective_norm,
            "cupdlpx_python_bound_objective_constraint_rescaling": constraint_bound_rescaling,
            "cupdlpx_python_bound_objective_objective_rescaling": objective_vector_rescaling,
        }

    @staticmethod
    def _extract_metrics(
        result_dict: Dict[str, Any],
        sol_dict: Optional[Dict[str, Any]],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if result_dict.get("termination_rel_primal_res") is not None:
            return (
                result_dict.get("termination_rel_primal_res"),
                result_dict.get("termination_rel_dual_res"),
                result_dict.get("rel_obj_gap"),
            )
        save_info = sol_dict.get("SaveInfo") if isinstance(sol_dict, dict) else None
        if isinstance(sol_dict, dict):
            if save_info == 2:
                primal_feas = sol_dict.get("PrimalFeasAvgRel")
                dual_feas = sol_dict.get("DualFeasAvgRel")
                gap = sol_dict.get("RelObjGapAverage")
            else:
                primal_feas = sol_dict.get("PrimalFeasRel")
                dual_feas = sol_dict.get("DualFeasRel")
                gap = sol_dict.get("RelObjGap")
        else:
            primal_feas = result_dict.get("PrimalFeasRel")
            dual_feas = result_dict.get("DualFeasRel")
            gap = result_dict.get("RelObjGap")
        return primal_feas, dual_feas, gap

    @staticmethod
    def _build_params(
        tolerance: Dict[str, float],
        verbose: int,
        extra_params: Optional[Dict[str, Any]] = None,
        *,
        use_device_problem_data: bool = False,
        bound_objective_rescaling: bool = False,
    ) -> Dict[str, Any]:
        """
        CN: 为 host-CSR 和 device-CSR 两条载入路径统一构造 cuPDLPx 参数，避免步长策略分叉导致的收敛行为漂移。
        EN: Build a unified cuPDLPx parameter set for both host-CSR and device-CSR load paths so that step-size policy does not diverge between them.
        """
        params = {
            "verbose": verbose,
            # Disable native timing summary unless explicitly verbose to avoid
            # uncontrollable C++ stdout noise and timing jitter.
            "verbose_time": 1 if int(verbose) else 0,
            "print_summary": bool(verbose),
            "termination_evaluation_frequency": 200,
            "termination_norm": "l2",
            "eps_optimal_absolute": tolerance.get("objective"),
            "eps_optimal_relative": tolerance.get("objective"),
            "eps_feasible_absolute_primal": tolerance.get("primal"),
            "eps_feasible_relative_primal": tolerance.get("primal"),
            "eps_feasible_absolute_dual": tolerance.get("dual"),
            "eps_feasible_relative_dual": tolerance.get("dual"),
            # CN: 这一组 step_size_method 默认值是 host/device 两条路径统一依赖的通用配置，
            # CN: 需要与 native cupdlpx 默认值和 pybind 暴露出的默认字典保持一致，避免未来再次出现路径分叉。
            # EN: This "step_size_method" default bundle is the common configuration used by both host and device paths,
            # EN: and must stay aligned with the native cupdlpx defaults and the pybind-exposed defaults to avoid future path divergence.
            "step_size_method": 3,
            "step_size_safety": 0.998,
            "power_max_iterations": 5000,
            "power_tolerance": 1e-4,
            "hybrid_refine_iterations": 100,
            "stepsize_power_reference": False,
            "stepsize_reference_max_iterations": 5000,
            "stepsize_reference_tolerance": 1e-4,
            "l_inf_ruiz_iterations": 0,
            "bound_objective_rescaling": bool(bound_objective_rescaling),
            "has_pock_chambolle_alpha": True,
            "enable_objective_gap_divergence_check": True,
            "vector_sum_mode": "resident_ones",
        }
        if isinstance(extra_params, dict):
            params.update(extra_params)
        params["termination_norm"] = CuPDLPxSolver._normalize_termination_norm(
            params.get("termination_norm", "l2")
        )
        return params

    @classmethod
    def _maybe_set_init_sol_primal_layout(
        cls,
        solver,
        warm_start_primal,
        warm_start_dual,
        n_primal: int,
        n_dual: int,
    ) -> None:
        primal_ok = warm_start_primal is not None and len(warm_start_primal) == n_primal
        dual_ok = warm_start_dual is not None and len(warm_start_dual) == n_dual
        if not (primal_ok or dual_ok):
            return
        x0 = cls._to_numpy_float64(warm_start_primal) if primal_ok else np.zeros(n_primal, dtype=np.float64)
        y0 = cls._to_numpy_float64(warm_start_dual) if dual_ok else np.zeros(n_dual, dtype=np.float64)
        solver.setInitSol(x0, y0)

    @classmethod
    def _maybe_set_init_sol_dual_layout(
        cls,
        solver,
        warm_start_primal,
        warm_start_dual,
        n_dual: int,
        n_primal: int,
    ) -> None:
        dual_ok = warm_start_dual is not None and len(warm_start_dual) == n_dual
        primal_ok = warm_start_primal is not None and len(warm_start_primal) == n_primal
        if not (dual_ok or primal_ok):
            return
        x0 = cls._to_numpy_float64(warm_start_dual) if dual_ok else np.zeros(n_dual, dtype=np.float64)
        y0 = cls._to_numpy_float64(warm_start_primal) if primal_ok else np.zeros(n_primal, dtype=np.float64)
        solver.setInitSol(x0, y0)

    def solve(
        self,
        c,
        A_csc,
        b_eq,
        lb,
        ub,
        n_eqs,
        tolerance: Dict[str, float] = {"objective": 1e-6, "primal": 1e-6, "dual": 1e-6},
        warm_start_primal=None,
        warm_start_dual=None,
        verbose=0,
        lp_form: str = "primal",
        dual_form_data: Optional[Dict[str, Any]] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        trace_collector=None,
        trace_prefix: str = "lp_backend",
        trace_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SolverResult:
        if not CUPDLPX_AVAILABLE:
            raise ImportError("pycupdlpx module not installed.")

        form = str(lp_form).strip().lower()
        if form not in {"primal", "dual"}:
            raise ValueError(f"Unknown lp_form: {lp_form!r}, expected 'primal' or 'dual'")
        device_problem_data = kwargs.pop("device_problem_data", None)
        solver_params_local = dict(solver_params or {})
        memory_accounting_component = solver_params_local.pop("memory_accounting_component", None)
        memory_accounting_metadata = solver_params_local.pop("memory_accounting_metadata", None)
        vector_sum_mode_arg = kwargs.pop("vector_sum_mode", None)
        vector_sum_mode = str(
            vector_sum_mode_arg
            if vector_sum_mode_arg is not None
            else solver_params_local.get("vector_sum_mode", "resident_ones")
        ).strip().lower()
        if vector_sum_mode not in {"resident_ones", "direct_reduce"}:
            raise ValueError("vector_sum_mode must be one of: resident_ones, direct_reduce")
        if device_problem_data is not None and form != "primal":
            raise ValueError("device_problem_data is only supported for primal LP form.")
        cupdlpx_python_pre_rescale_requested = self._normalize_bool_flag(
            kwargs.pop(
                "cupdlpx_python_pre_rescale",
                solver_params_local.get("cupdlpx_python_pre_rescale", self.cupdlpx_python_pre_rescale),
            ),
            default=True,
        )

        solver_diag: Dict[str, Any] = {
            "lp_form": form,
            "cupdlpx_python_pre_rescale_requested": bool(cupdlpx_python_pre_rescale_requested),
            "cupdlpx_python_pre_rescale_effective": False,
            "cupdlpx_python_pre_rescale_skip_reason": None if device_problem_data is not None else "non_device_csr_path",
        }

        base_trace_args: Dict[str, Any] = {}
        if isinstance(trace_args, dict):
            base_trace_args.update({str(key): _trace_scalar(value) for key, value in trace_args.items()})

        def _trace_span(name: str, *, args: Optional[Dict[str, Any]] = None):
            if trace_collector is None:
                return nullcontext()
            merged_args = dict(base_trace_args)
            if isinstance(args, dict):
                merged_args.update({str(key): _trace_scalar(value) for key, value in args.items()})
            return trace_collector.span(name, "solve_ot", args=merged_args)

        t0 = time.perf_counter()
        with _trace_span(f"{trace_prefix}.construct_solver"):
            solver = pycupdlpx.cupdlpx()
        solver_diag["construct_time"] = float(time.perf_counter() - t0)

        def _as_float64_or_none(arr):
            if arr is None:
                return None
            return np.asarray(arr, dtype=np.float64)

        def _to_host_numpy(arr, *, dtype=None):
            if arr is None:
                return None
            if self._is_torch_tensor(arr):
                tensor = arr.detach()
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                return tensor.contiguous().numpy()
            return np.asarray(arr)

        dump_on_failure_enabled = self._cupdlpx_dump_on_failure_enabled()
        dump_lp_input_builder: Optional[Callable[[], Dict[str, Any]]] = None
        dump_lp_input_cache: Optional[Dict[str, Any]] = None
        pre_rescale_diag: Dict[str, Any] = {
            "cupdlpx_python_pre_rescale_requested": bool(
                cupdlpx_python_pre_rescale_requested
            ),
            "cupdlpx_python_pre_rescale_effective": False,
            "cupdlpx_python_pre_rescale_skip_reason": "non_device_csr_path",
        }
        params = self._build_params(
            tolerance=tolerance,
            verbose=verbose,
            extra_params=solver_params_local,
            use_device_problem_data=device_problem_data is not None,
            bound_objective_rescaling=self.bound_objective_rescaling is True,
        )
        params.pop("cupdlpx_python_pre_rescale", None)
        params["vector_sum_mode"] = vector_sum_mode
        solver_diag["termination_norm"] = str(params["termination_norm"])
        original_params = dict(params)

        if form == "primal":
            n_primal = int(len(c)) if c is not None else 0
            t0 = time.perf_counter()
            with _trace_span(
                f"{trace_prefix}.load_data",
                args={
                    "lp_form": form,
                    "n_eqs": int(n_eqs),
                },
            ):
                if device_problem_data is not None:
                    normalized = self._normalize_device_problem_data(device_problem_data)
                    load_normalized, params, pre_rescale_diag = self._maybe_python_pre_rescale_device_problem(
                        normalized,
                        params,
                        requested=cupdlpx_python_pre_rescale_requested,
                    )
                    solver_diag.update(pre_rescale_diag)
                    solver.loadData_device_csr(
                        load_normalized["row_ptr"],
                        load_normalized["col_ind"],
                        load_normalized["values"],
                        load_normalized["m"],
                        load_normalized["n"],
                        load_normalized["c"],
                        load_normalized["rhs"],
                        load_normalized["lb"],
                        load_normalized["ub"],
                        load_normalized["n_eqs"],
                        load_normalized["variable_bound_mode"],
                        load_normalized["matrix_value_mode"],
                        load_normalized.get("constraint_rescaling", None),
                        load_normalized.get("variable_rescaling", None),
                        float(load_normalized.get("constraint_bound_rescaling", 1.0)),
                        float(load_normalized.get("objective_vector_rescaling", 1.0)),
                        load_normalized.get("original_objective_vector_norm", None),
                        load_normalized.get("original_constraint_bound_norm", None),
                        bool(load_normalized.get("has_precomputed_rescaling", False)),
                        load_normalized.get("original_objective_vector_linf_norm", None),
                        load_normalized.get("original_constraint_bound_linf_norm", None),
                    )
                    n_primal = normalized["n"]
                    solver_diag["device_lp_pipeline"] = True
                    solver_diag["device_lp_pipeline_device"] = str(normalized["device"])
                    solver_diag["variable_bound_mode"] = str(load_normalized["variable_bound_mode"])
                    solver_diag["matrix_value_mode"] = str(load_normalized["matrix_value_mode"])
                    solver_diag["input_matrix_value_mode"] = str(normalized["matrix_value_mode"])
                    solver_diag["implicit_ax_enabled"] = str(load_normalized["matrix_value_mode"]) in {"implicit_ax", "implicit_both"}
                    solver_diag["implicit_aty_enabled"] = str(load_normalized["matrix_value_mode"]) in {"implicit_aty", "implicit_both"}

                    if dump_on_failure_enabled:
                        def _build_device_dump_lp_input(
                            effective_problem=load_normalized,
                        ) -> Dict[str, Any]:
                            payload = {
                                "row_ptr": _to_host_numpy(
                                    effective_problem["row_ptr"]
                                ).astype(np.int32, copy=False),
                                "col_ind": _to_host_numpy(
                                    effective_problem["col_ind"]
                                ).astype(np.int32, copy=False),
                                "values": (
                                    None
                                    if effective_problem.get("values") is None
                                    else _to_host_numpy(
                                        effective_problem["values"]
                                    ).astype(np.float64, copy=False)
                                ),
                                "c": _to_host_numpy(effective_problem["c"]).astype(
                                    np.float64, copy=False
                                ),
                                "rhs": _to_host_numpy(
                                    effective_problem["rhs"]
                                ).astype(np.float64, copy=False),
                                "m": int(effective_problem["m"]),
                                "n": int(effective_problem["n"]),
                                "n_eqs": int(effective_problem["n_eqs"]),
                                "device": str(effective_problem["device"]),
                                "variable_bound_mode": str(
                                    effective_problem["variable_bound_mode"]
                                ),
                                "matrix_value_mode": str(
                                    effective_problem["matrix_value_mode"]
                                ),
                                "has_precomputed_rescaling": bool(
                                    effective_problem.get(
                                        "has_precomputed_rescaling", False
                                    )
                                ),
                            }
                            if payload["variable_bound_mode"] == "constant":
                                payload["lb"] = float(effective_problem["lb"])
                                payload["ub"] = float(effective_problem["ub"])
                            else:
                                payload["lb"] = _to_host_numpy(
                                    effective_problem["lb"]
                                ).astype(np.float64, copy=False)
                                payload["ub"] = _to_host_numpy(
                                    effective_problem["ub"]
                                ).astype(np.float64, copy=False)
                            if payload["has_precomputed_rescaling"]:
                                payload.update(
                                    {
                                        "constraint_rescaling": _to_host_numpy(
                                            effective_problem[
                                                "constraint_rescaling"
                                            ]
                                        ).astype(np.float64, copy=False),
                                        "variable_rescaling": _to_host_numpy(
                                            effective_problem[
                                                "variable_rescaling"
                                            ]
                                        ).astype(np.float64, copy=False),
                                        "constraint_bound_rescaling": float(
                                            effective_problem.get(
                                                "constraint_bound_rescaling", 1.0
                                            )
                                        ),
                                        "objective_vector_rescaling": float(
                                            effective_problem.get(
                                                "objective_vector_rescaling", 1.0
                                            )
                                        ),
                                        "original_objective_vector_norm": float(
                                            effective_problem[
                                                "original_objective_vector_norm"
                                            ]
                                        ),
                                        "original_constraint_bound_norm": float(
                                            effective_problem[
                                                "original_constraint_bound_norm"
                                            ]
                                        ),
                                        "original_objective_vector_linf_norm": float(
                                            effective_problem[
                                                "original_objective_vector_linf_norm"
                                            ]
                                        ),
                                        "original_constraint_bound_linf_norm": float(
                                            effective_problem[
                                                "original_constraint_bound_linf_norm"
                                            ]
                                        ),
                                    }
                                )
                            return {
                                "dump_format_version": 2,
                                "effective_problem": payload,
                                "warm_start_primal": _to_host_numpy(
                                    warm_start_primal
                                ),
                                "warm_start_dual": _to_host_numpy(
                                    warm_start_dual
                                ),
                                "device_lp_pipeline": True,
                            }

                        dump_lp_input_builder = _build_device_dump_lp_input
                else:
                    solver.loadData(A=A_csc, c=c, rhs=b_eq, lb=lb, ub=ub, nEqs=n_eqs)
                    solver_diag["device_lp_pipeline"] = False

                    if dump_on_failure_enabled:
                        def _build_host_dump_lp_input() -> Dict[str, Any]:
                            return {
                                "c": c,
                                "A_csc": A_csc if sp.isspmatrix_csc(A_csc) else A_csc.tocsc(),
                                "b_eq": b_eq,
                                "lb": lb,
                                "ub": ub,
                                "n_eqs": n_eqs,
                                "warm_start_primal": _to_host_numpy(warm_start_primal),
                                "warm_start_dual": _to_host_numpy(warm_start_dual),
                                "device_lp_pipeline": False,
                            }

                        dump_lp_input_builder = _build_host_dump_lp_input
            solver_diag["load_data_time"] = float(time.perf_counter() - t0)
            try:
                solver_diag["load_data_breakdown"] = dict(solver.getLoadStats())
            except Exception:
                pass
            t0 = time.perf_counter()
            with _trace_span(
                f"{trace_prefix}.warm_start",
                args={
                    "lp_form": form,
                    "warm_start_primal_present": warm_start_primal is not None,
                    "warm_start_dual_present": warm_start_dual is not None,
                },
            ):
                self._maybe_set_init_sol_primal_layout(
                    solver,
                    warm_start_primal=warm_start_primal,
                    warm_start_dual=warm_start_dual,
                    n_primal=n_primal,
                    n_dual=n_eqs,
                )
            solver_diag["warm_start_time"] = float(time.perf_counter() - t0)
        else:
            payload = dual_form_data or {}
            minus_AT = payload.get("minus_AT")
            neg_minus_AT = payload.get("neg_minus_AT")
            minus_c = payload.get("minus_c")
            minus_q = payload.get("minus_q")

            if minus_AT is None or minus_c is None:
                raise ValueError("dual_form_data must contain 'minus_AT' and 'minus_c'")

            if neg_minus_AT is None:
                minus_AT_csc = minus_AT if sp.isspmatrix_csc(minus_AT) else minus_AT.tocsc()
                A_load = -minus_AT_csc
            else:
                A_load = neg_minus_AT if sp.isspmatrix_csc(neg_minus_AT) else neg_minus_AT.tocsc()
                minus_AT_csc = minus_AT if sp.isspmatrix_csc(minus_AT) else minus_AT.tocsc()

            minus_c = np.asarray(minus_c, dtype=np.float64)
            if minus_q is None:
                if b_eq is None:
                    raise ValueError("dual_form_data missing 'minus_q' and fallback b_eq is None")
                minus_q = -np.asarray(b_eq, dtype=np.float64)
            else:
                minus_q = np.asarray(minus_q, dtype=np.float64)

            if minus_AT_csc.shape[0] != minus_c.shape[0]:
                raise ValueError(
                    f"dual form shape mismatch: minus_AT rows={minus_AT_csc.shape[0]} vs minus_c={minus_c.shape[0]}"
                )
            if minus_AT_csc.shape[1] != minus_q.shape[0]:
                raise ValueError(
                    f"dual form shape mismatch: minus_AT cols={minus_AT_csc.shape[1]} vs minus_q={minus_q.shape[0]}"
                )

            n_dual = minus_q.shape[0]
            n_primal = minus_AT_csc.shape[0]
            lb_dual = np.full(n_dual, -np.inf, dtype=np.float64)
            ub_dual = np.full(n_dual, np.inf, dtype=np.float64)

            t0 = time.perf_counter()
            with _trace_span(
                f"{trace_prefix}.load_data",
                args={
                    "lp_form": form,
                    "n_eqs": 0,
                },
            ):
                solver.loadData(
                    A=A_load,
                    c=minus_q,
                    rhs=-minus_c,
                    lb=lb_dual,
                    ub=ub_dual,
                    nEqs=0,
                )

                if dump_on_failure_enabled:
                    def _build_dual_dump_lp_input() -> Dict[str, Any]:
                        return {
                            "c": minus_q,
                            "A_csc": A_load if sp.isspmatrix_csc(A_load) else A_load.tocsc(),
                            "b_eq": -minus_c,
                            "lb": lb_dual,
                            "ub": ub_dual,
                            "n_eqs": 0,
                            "warm_start_primal": _to_host_numpy(warm_start_dual),
                            "warm_start_dual": _to_host_numpy(warm_start_primal),
                            "device_lp_pipeline": False,
                        }

                    dump_lp_input_builder = _build_dual_dump_lp_input
            solver_diag["load_data_time"] = float(time.perf_counter() - t0)
            try:
                solver_diag["load_data_breakdown"] = dict(solver.getLoadStats())
            except Exception:
                pass
            t0 = time.perf_counter()
            with _trace_span(
                f"{trace_prefix}.warm_start",
                args={
                    "lp_form": form,
                    "warm_start_primal_present": warm_start_primal is not None,
                    "warm_start_dual_present": warm_start_dual is not None,
                },
            ):
                self._maybe_set_init_sol_dual_layout(
                    solver,
                    warm_start_primal=warm_start_primal,
                    warm_start_dual=warm_start_dual,
                    n_dual=n_dual,
                    n_primal=n_primal,
                )
            solver_diag["warm_start_time"] = float(time.perf_counter() - t0)

        solver_diag["vector_sum_mode"] = vector_sum_mode
        logger.info(f"  使用容差值: {tolerance}")
        logger.info(
            "  cuPDLPx termination-only norm=%s (algorithm residual norm=l2)",
            params["termination_norm"],
        )
        n_vars_log = int(device_problem_data["n"]) if isinstance(device_problem_data, dict) and device_problem_data.get("n") is not None else len(c)
        logger.info(f"  调用 cuPDLPx (lp_form={form}, n_vars={n_vars_log})...")

        native_trace_args = {
            "solver_iterations": None,
            "solver_runtime_sec": None,
            "solver_wall_time_sec": None,
            "termination_reason": None,
            "success": None,
            "lp_solve_kind": str(base_trace_args.get("lp_solve_kind", "normal")),
            "approx_prune_stage": base_trace_args.get("approx_prune_stage"),
            **base_trace_args,
            "lp_form": form,
            "n_eqs": int(n_eqs),
            "n_vars_solver_input": int(n_vars_log),
            "warm_start_primal_present": warm_start_primal is not None,
            "warm_start_dual_present": warm_start_dual is not None,
            "device_lp_pipeline": bool(device_problem_data is not None),
        }
        if isinstance(device_problem_data, dict) and device_problem_data.get("device") is not None:
            native_trace_args["device_lp_pipeline_device"] = str(device_problem_data.get("device"))

        native_memory_context = nullcontext()
        native_memory_span = None
        native_memory_enabled = False
        native_memory_metadata: Dict[str, Any] = {}
        if memory_accounting_component is not None:
            native_memory_recorder = current_memory_recorder()
            if native_memory_recorder is not None:
                native_memory_enabled = True
                native_memory_metadata.update(base_trace_args)
                if isinstance(memory_accounting_metadata, dict):
                    native_memory_metadata.update(
                        {str(key): _trace_scalar(value) for key, value in memory_accounting_metadata.items()}
                    )
                native_memory_metadata.setdefault("trace_prefix", str(trace_prefix))
                native_memory_metadata["memory_accounting_entry"] = "pre_native_solve"
                native_memory_context = native_memory_recorder.component(
                    str(memory_accounting_component),
                    metadata=native_memory_metadata,
                )

        pre_native_driver_used_mib = None
        pre_native_torch_alloc_mib = None
        pre_native_delta_peak_mib = None

        t0 = time.perf_counter()
        if trace_collector is None:
            native_solve_span = nullcontext()
        else:
            native_solve_span = trace_collector.span(f"{trace_prefix}.native_solve", "solve_ot", args=native_trace_args)
        with native_solve_span:
            with native_memory_context as native_memory_span:
                native_memory_device = 0
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    if isinstance(device_problem_data, dict) and device_problem_data.get("device") is not None:
                        parsed_device = torch.device(device_problem_data["device"])
                        native_memory_device = (
                            int(parsed_device.index)
                            if parsed_device.index is not None
                            else int(torch.cuda.current_device())
                        )
                    else:
                        native_memory_device = int(torch.cuda.current_device())
                pre_native_driver_tracker = DriverMemoryTracker(
                    device=native_memory_device,
                    enabled=native_memory_enabled,
                )
                pre_native_driver_used_mib = pre_native_driver_tracker.current_mib
                if native_memory_enabled and TORCH_AVAILABLE and torch.cuda.is_available():
                    pre_native_torch_alloc_mib = float(torch.cuda.memory_allocated(torch.cuda.current_device())) / float(1024 ** 2)
                result_dict = solver.solve(params)
                peak_gpu_mem = result_dict.get("peak_gpu_mem_mib", None) if isinstance(result_dict, dict) else None
                try:
                    peak_gpu_mem_float = None if peak_gpu_mem is None else float(peak_gpu_mem)
                except (TypeError, ValueError):
                    peak_gpu_mem_float = None
                pre_native_delta_peak_mib = (
                    None
                    if peak_gpu_mem_float is None or pre_native_driver_used_mib is None
                    else max(0.0, float(peak_gpu_mem_float) - float(pre_native_driver_used_mib))
                )
                if native_memory_span is not None:
                    implicit_metadata = {}
                    if isinstance(result_dict, dict):
                        implicit_metadata = {
                            "matrix_value_mode": str(result_dict.get("matrix_value_mode", solver_diag.get("matrix_value_mode", "explicit"))),
                            "implicit_ax_enabled": bool(result_dict.get("implicit_ax_enabled", solver_diag.get("implicit_ax_enabled", False))),
                            "implicit_aty_enabled": bool(result_dict.get("implicit_aty_enabled", solver_diag.get("implicit_aty_enabled", False))),
                            "implicit_ax_agg": str(result_dict.get("implicit_ax_agg", "none")),
                            "implicit_ax_row0_unique_p50": result_dict.get("implicit_ax_row0_unique_p50"),
                            "implicit_ax_row1_unique_p50": result_dict.get("implicit_ax_row1_unique_p50"),
                        }
                    native_memory_span.report_delta_peak_mib(pre_native_delta_peak_mib, source="lp_backend_pre_native")
                    native_memory_span.update_metadata(
                        {
                            "pre_native_driver_used_mib": pre_native_driver_used_mib,
                            "pre_native_torch_alloc_mib": pre_native_torch_alloc_mib,
                            "lp_backend_abs_peak_mem_mib": peak_gpu_mem_float,
                            "lp_backend_pre_native_delta_peak_mib": pre_native_delta_peak_mib,
                            **implicit_metadata,
                        }
                    )
                solver_diag["pre_native_driver_used_mib"] = pre_native_driver_used_mib
                solver_diag["pre_native_torch_alloc_mib"] = pre_native_torch_alloc_mib
                solver_diag["lp_backend_pre_native_delta_peak_mib"] = pre_native_delta_peak_mib
            native_trace_args["solver_iterations"] = int(result_dict.get("iterations", 0) or 0)
            native_trace_args["solver_runtime_sec"] = float(result_dict.get("runtime_sec", 0.0) or 0.0)
            native_trace_args["solver_wall_time_sec"] = float(time.perf_counter() - t0)
            native_trace_args["termination_reason"] = result_dict.get("termination_reason")
            native_trace_args["success"] = bool(result_dict.get("success", False))
            native_trace_args["peak_gpu_mem_mib"] = float(result_dict.get("peak_gpu_mem_mib", 0.0) or 0.0)
            native_trace_args["vector_sum_mode"] = str(result_dict.get("vector_sum_mode", vector_sum_mode))
            native_trace_args["matrix_value_mode"] = str(result_dict.get("matrix_value_mode", solver_diag.get("matrix_value_mode", "explicit")))
            native_trace_args["implicit_ax_enabled"] = bool(result_dict.get("implicit_ax_enabled", solver_diag.get("implicit_ax_enabled", False)))
            native_trace_args["implicit_aty_enabled"] = bool(result_dict.get("implicit_aty_enabled", solver_diag.get("implicit_aty_enabled", False)))
            native_trace_args["implicit_ax_agg"] = str(result_dict.get("implicit_ax_agg", "none"))
            native_trace_args["implicit_ax_row0_unique_p50"] = float(result_dict.get("implicit_ax_row0_unique_p50", float("nan")))
            native_trace_args["implicit_ax_row1_unique_p50"] = float(result_dict.get("implicit_ax_row1_unique_p50", float("nan")))
            native_trace_args["support_stop_obj_rel_change"] = float(result_dict.get("support_stop_obj_rel_change", float("nan")) or 0.0)
            native_trace_args["termination_norm"] = str(result_dict.get("termination_norm", params["termination_norm"]))
            native_trace_args["termination_rel_primal_res"] = float(result_dict.get("termination_rel_primal_res", float("nan")))
            native_trace_args["termination_rel_dual_res"] = float(result_dict.get("termination_rel_dual_res", float("nan")))
            native_trace_args["native_wall_time_sec"] = float(time.perf_counter() - t0)
        solver_diag["native_solve_wall_time"] = float(time.perf_counter() - t0)

        def _maybe_dump_failed_lp(*, stage: str) -> None:
            if not dump_on_failure_enabled:
                return
            if not isinstance(result_dict, dict):
                return
            termination = result_dict.get("termination_reason")
            if termination in {"OPTIMAL", "OPTIMAL_WITH_SUPPORT_LIMIT"}:
                return
            termination_filter = str(
                os.environ.get("HIEROT_CUPDLPX_DUMP_TERMINATION", "")
            ).strip()
            if termination_filter:
                allowed_terminations = {
                    item.strip()
                    for item in termination_filter.split(",")
                    if item.strip()
                }
                if termination not in allowed_terminations:
                    return
            elif termination == "SUPPORT_LIMIT_REACHED":
                return
            if dump_lp_input_builder is None:
                logger.warning(
                    "  [cupdlpx dump] skip stage=%s because LP input is unavailable",
                    stage,
                )
                return
            try:
                from ..instrumentation.dump import (
                    DEFAULT_CUPDLPX_DUMP_ROOT,
                    dump_cupdlpx_device_lp_v2,
                    dump_cupdlpx_lp,
                )

                nonlocal dump_lp_input_cache
                if dump_lp_input_cache is None:
                    dump_lp_input_cache = dump_lp_input_builder()
                dump_lp_input = dump_lp_input_cache
                dump_root = Path(
                    os.environ.get(
                        "HIEROT_CUPDLPX_DUMP_ROOT",
                        str(DEFAULT_CUPDLPX_DUMP_ROOT),
                    )
                )
                dump_params = dict(pycupdlpx.make_default_params())
                dump_params.update(params)
                dump_params["dump_stage"] = stage
                dump_params["trace_args"] = dict(base_trace_args)
                if int(dump_lp_input.get("dump_format_version", 1)) == 2:
                    dump_dir = dump_cupdlpx_device_lp_v2(
                        effective_problem=dump_lp_input["effective_problem"],
                        warm_start_primal=dump_lp_input["warm_start_primal"],
                        warm_start_dual=dump_lp_input["warm_start_dual"],
                        tolerance=tolerance,
                        original_params=original_params,
                        effective_params=dump_params,
                        pre_rescale_diag=pre_rescale_diag,
                        trace_args=base_trace_args,
                        result_dict=result_dict,
                        sol_dict=sol_dict,
                        dump_root=dump_root,
                    )
                else:
                    dump_dir = dump_cupdlpx_lp(
                        c=np.asarray(dump_lp_input["c"]),
                        A_csc=dump_lp_input["A_csc"],
                        b_eq=np.asarray(dump_lp_input["b_eq"]),
                        lb=np.asarray(dump_lp_input["lb"]),
                        ub=np.asarray(dump_lp_input["ub"]),
                        n_eqs=int(dump_lp_input["n_eqs"]),
                        warm_start_primal=dump_lp_input["warm_start_primal"],
                        warm_start_dual=dump_lp_input["warm_start_dual"],
                        tolerance=tolerance,
                        params=dump_params,
                        result_dict=result_dict,
                        sol_dict=sol_dict,
                        dump_root=dump_root,
                    )
                logger.warning(
                    "  [cupdlpx dump] stage=%s termination=%s dumped_to: %s",
                    stage,
                    termination,
                    dump_dir,
                )
            except Exception as exc:
                logger.warning(
                    "  [cupdlpx dump] failed stage=%s error=%r",
                    stage,
                    exc,
                )

        if (
            isinstance(result_dict, dict)
            and result_dict.get("termination_reason") == "NUMERICAL_DIVERGENCE"
            and self.bound_objective_rescaling == "auto"
            and not bool(params.get("bound_objective_rescaling", False))
        ):
            logger.warning("  NUMERICAL_DIVERGENCE; retry with bound_objective_rescaling=True")
            retry_solver_params = dict(solver_params_local)
            retry_solver_params["bound_objective_rescaling"] = True
            retry_kwargs = dict(kwargs)
            if device_problem_data is not None:
                retry_kwargs["device_problem_data"] = device_problem_data
            retry_solver = type(self)(bound_objective_rescaling=True)
            retry_result = retry_solver.solve(
                c,
                A_csc,
                b_eq,
                lb,
                ub,
                n_eqs,
                tolerance=tolerance,
                warm_start_primal=warm_start_primal,
                warm_start_dual=warm_start_dual,
                verbose=verbose,
                lp_form=form,
                dual_form_data=dual_form_data,
                solver_params=retry_solver_params,
                trace_collector=trace_collector,
                trace_prefix=trace_prefix,
                trace_args=trace_args,
                **retry_kwargs,
            )
            retry_diag = dict(retry_result.solver_diag or {})
            retry_diag.update(
                {
                    "fallback_bound_objective_rescaling": True,
                    "fallback_first_termination_reason": result_dict.get("termination_reason"),
                    "fallback_first_iterations": int(result_dict.get("iterations", 0) or 0),
                    "fallback_first_runtime_sec": float(result_dict.get("runtime_sec", 0.0) or 0.0),
                    "fallback_first_rel_obj_gap": float(result_dict.get("rel_obj_gap") or float("nan")),
                }
            )
            retry_result.solver_diag = retry_diag
            return retry_result

        if isinstance(result_dict, dict) and result_dict.get("trace_snapshots") is not None:
            solver_diag["trace_snapshots"] = result_dict.get("trace_snapshots")
        if isinstance(result_dict, dict):
            solver_diag["termination_reason"] = result_dict.get("termination_reason")
            solver_diag["iterations"] = int(result_dict.get("iterations", 0) or 0)
        if isinstance(result_dict, dict) and result_dict.get("support_stop_obj_rel_change") is not None:
            solver_diag["support_stop_obj_rel_change"] = float(result_dict.get("support_stop_obj_rel_change"))
        if isinstance(result_dict, dict) and result_dict.get("support_stop_last_eval_nnz") is not None:
            solver_diag["support_stop_last_eval_nnz"] = int(result_dict.get("support_stop_last_eval_nnz"))
        if isinstance(result_dict, dict) and result_dict.get("support_limit_mode") is not None:
            solver_diag["support_limit_mode"] = str(result_dict.get("support_limit_mode"))
        if isinstance(result_dict, dict):
            for key in (
                "termination_norm",
                "termination_abs_primal_res",
                "termination_rel_primal_res",
                "termination_abs_dual_res",
                "termination_rel_dual_res",
                "termination_objective_vector_norm",
                "termination_constraint_bound_norm",
                "matrix_value_mode",
                "implicit_ax_enabled",
                "implicit_aty_enabled",
                "implicit_ax_agg",
                "implicit_ax_row0_unique_p50",
                "implicit_ax_row1_unique_p50",
            ):
                if result_dict.get(key) is not None:
                    solver_diag[key] = result_dict.get(key)
            solver_diag["algorithm_abs_primal_res_l2"] = result_dict.get("abs_primal_res")
            solver_diag["algorithm_rel_primal_res_l2"] = result_dict.get("rel_primal_res")
            solver_diag["algorithm_abs_dual_res_l2"] = result_dict.get("abs_dual_res")
            solver_diag["algorithm_rel_dual_res_l2"] = result_dict.get("rel_dual_res")
            solver_diag["relative_primal_dual_gap"] = result_dict.get("rel_obj_gap")
        continuation_state = self._normalize_continuation_state(
            result_dict.get("continuation_state") if isinstance(result_dict, dict) else None
        )
        if continuation_state is not None:
            solver_diag["continuation_state"] = continuation_state

        try:
            t0 = time.perf_counter()
            with _trace_span(f"{trace_prefix}.get_solution"):
                sol_dict = solver.getSolution()
            solver_diag["get_solution_time"] = float(time.perf_counter() - t0)
        except Exception:
            sol_dict = None
            solver_diag["get_solution_time"] = 0.0

        peak_mem = result_dict.get("peak_gpu_mem_mib", 0.0)
        try:
            peak_mem_float = float(peak_mem or 0.0)
        except (TypeError, ValueError):
            peak_mem_float = 0.0
        entry_mem = (
            pre_native_driver_used_mib
            if pre_native_driver_used_mib is not None and memory_accounting_component is not None
            else base_trace_args.get("lp_entry_used_mem_mib")
        )
        try:
            entry_mem_float = None if entry_mem is None else float(entry_mem)
        except (TypeError, ValueError):
            entry_mem_float = None
        delta_peak = None if entry_mem_float is None else max(0.0, float(peak_mem_float) - float(entry_mem_float))
        solver_diag["lp_backend_abs_peak_mem_mib"] = float(peak_mem_float)
        solver_diag["lp_backend_entry_mem_mib"] = entry_mem_float
        solver_diag["lp_backend_delta_peak_mem_mib"] = delta_peak
        solver_diag["lp_backend_peak_mem_mib"] = delta_peak
        duration = result_dict.get("runtime_sec", 0.0)
        iterations = result_dict.get("iterations", 0)
        is_optimal = result_dict.get("termination_reason") in {"OPTIMAL", "OPTIMAL_WITH_SUPPORT_LIMIT"}
        primal_feas, dual_feas, gap = self._extract_metrics(result_dict, sol_dict)
        logger.info(f"  [GPU Mem] C++ Solver Peak Usage: {peak_mem:.2f} MiB")
        if duration:
            logger.info(
                f"  Solve Done, Time = {duration:.2f}, nIter = {iterations:,}, iter/sec = {iterations / duration:.2f}"
            )
        else:
            logger.info(f"  Solve Done, Time = {duration:.2f}, nIter = {iterations:,}")
        if not is_optimal:
            _maybe_dump_failed_lp(stage="final")
            logger.warning(f"  警告: 求解失败或未达到最优。原因: {result_dict.get('termination_reason')}")

        if form == "primal":
            primal_sol = self._copy_if_present(sol_dict, "x")
            if primal_sol is None:
                primal_sol = self._copy_if_present(result_dict, "x")
            dual_sol = self._copy_if_present(sol_dict, "y")
            if dual_sol is None:
                dual_sol = self._copy_if_present(result_dict, "y")
        else:
            primal_sol = self._copy_if_present(sol_dict, "y")
            if primal_sol is None:
                primal_sol = self._copy_if_present(result_dict, "y")
            dual_sol = self._copy_if_present(sol_dict, "x")
            if dual_sol is None:
                dual_sol = self._copy_if_present(result_dict, "x")

        t0 = time.perf_counter()
        primal_sol = _as_float64_or_none(primal_sol)
        dual_sol = _as_float64_or_none(dual_sol)
        solver_diag["extract_solution_time"] = float(time.perf_counter() - t0)
        solver_diag["runtime_sec"] = float(duration)

        obj_val = (
            sol_dict.get("PrimalObj")
            if isinstance(sol_dict, dict) and sol_dict.get("PrimalObj") is not None
            else result_dict.get("primal_objective", 0.0)
        )
        dual_obj_val = (
            sol_dict.get("DualObj")
            if isinstance(sol_dict, dict) and sol_dict.get("DualObj") is not None
            else result_dict.get("dual_objective")
        )
        iter_count = (
            sol_dict.get("iters")
            if isinstance(sol_dict, dict) and sol_dict.get("iters") is not None
            else result_dict.get("iterations", 0)
        )

        return SolverResult(
            success=is_optimal,
            x=primal_sol,
            y=dual_sol,
            obj_val=obj_val,
            dual_obj_val=dual_obj_val,
            duration=duration,
            iterations=iter_count,
            peak_mem=peak_mem,
            termination_reason=result_dict.get("termination_reason"),
            primal_feas=primal_feas,
            dual_feas=dual_feas,
            gap=gap,
            solver_diag=solver_diag,
        )
