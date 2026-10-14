from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.sparse import csc_matrix

from hello_ot.progress import report_sparsification
from hello_ot.refinement.support_budget import (
    POSITIVE_SUPPORT_TOLERANCE,
    primal_support_cap,
)
from hello_ot.restricted_ot.support_sparsification import sparsify_transport_support


def _numpy_vector(values: Any) -> np.ndarray:
    """
    CN: 将 LP 向量转换为连续 FP64 主机数组。
    EN: Convert an LP vector to a contiguous FP64 host array.
    """
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    return np.ascontiguousarray(values, dtype=np.float64).reshape(-1)


def _numpy_array(values: Any, dtype: Any) -> np.ndarray:
    """
    CN: 将设备数组转换为指定 dtype 的连续一维 NumPy 数组。
    EN: Convert a device array to a contiguous one-dimensional NumPy array of the requested dtype.
    """
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.ascontiguousarray(values, dtype=dtype).reshape(-1)


def _replace_vector_like(reference: Any, values: np.ndarray) -> Any:
    """
    CN: 用与 backend 原结果相同的设备和 dtype 重建向量。
    EN: Rebuild a vector with the device and dtype of the original backend result.
    """
    if torch.is_tensor(reference):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    return np.asarray(values, dtype=np.asarray(reference).dtype)


def _positive_nnz(values: Any) -> int:
    """
    CN: 在 LP 输出所在设备统计 positive support，仅返回一个主机标量。
    EN: Count positive support on the LP output device and return only a host scalar.
    """
    if torch.is_tensor(values):
        return int(torch.count_nonzero(values > POSITIVE_SUPPORT_TOLERANCE).item())
    return int(np.count_nonzero(np.asarray(values) > POSITIVE_SUPPORT_TOLERANCE))


def sparsify_restricted_lp_output(
    solver: Any,
    lvl_s: Any,
    lvl_t: Any,
    primal: Any,
    dual: np.ndarray,
    backend_result: Any,
) -> tuple[Any, np.ndarray, Any]:
    """
    CN: 按论文 cap 稀疏化过密的 restricted-LP primal，并同步修正 primal--dual 输出。
    EN: Sparsify an over-dense restricted-LP primal at the paper cap and synchronize the primal-dual output.
    """
    n_source = int(len(lvl_s.points))
    n_target = int(len(lvl_t.points))
    beta = float(solver._support_budget_factor)
    gamma = int(solver._pricing_topk)
    target_nnz = primal_support_cap(beta, gamma, n_source, n_target)
    input_nnz = _positive_nnz(backend_result.x)
    if input_nnz <= target_nnz:
        return primal, dual, backend_result

    values = _numpy_vector(backend_result.x)
    support = {
        "rows": _numpy_array(solver.active_support.rows, np.int64),
        "cols": _numpy_array(solver.active_support.cols, np.int64),
        "c_vec": _numpy_array(solver.active_support.c_vec, np.float64),
    }
    engine = "cpp_lct" if str(solver._backend) == "native" else "numba_lct"
    result = sparsify_transport_support(
        support["rows"],
        support["cols"],
        values,
        support["c_vec"],
        n_source=n_source,
        n_target=n_target,
        target_nnz=target_nnz,
        zero_tolerance=POSITIVE_SUPPORT_TOLERANCE,
        engine=engine,
    )

    values_new = result.values
    dual_new = np.asarray(dual, dtype=np.float64).reshape(-1).copy()
    primal_objective = float(np.dot(support["c_vec"], values_new))
    source_mass = np.asarray(lvl_s.masses, dtype=np.float64).reshape(-1)
    target_mass = np.asarray(lvl_t.masses, dtype=np.float64).reshape(-1)
    dual_objective = float(
        np.dot(source_mass, dual_new[:n_source])
        + np.dot(target_mass, dual_new[n_source:])
    )
    dual_shift = 0.0
    if primal_objective < dual_objective:
        total_mass = float(np.sum(source_mass))
        dual_shift = float((dual_objective - primal_objective) / total_mass)
        dual_new[:n_source] -= 0.5 * dual_shift
        dual_new[n_source:] -= 0.5 * dual_shift
        dual_objective = float(
            np.dot(source_mass, dual_new[:n_source])
            + np.dot(target_mass, dual_new[n_source:])
        )

    relative_gap = abs(primal_objective - dual_objective) / (
        1.0 + abs(primal_objective) + abs(dual_objective)
    )
    backend_result.x = _replace_vector_like(backend_result.x, values_new)
    backend_result.y = _replace_vector_like(backend_result.y, dual_new)
    backend_result.obj_val = primal_objective
    backend_result.dual_obj_val = dual_objective
    backend_result.gap = relative_gap
    if isinstance(backend_result.solver_diag, dict):
        backend_result.solver_diag["relative_primal_dual_gap"] = relative_gap

    primal_new = csc_matrix(
        (values_new, (support["rows"], support["cols"])),
        shape=(n_source, n_target),
    )
    report_sparsification(
        solver._verbosity,
        input_nnz=result.input_nnz,
        output_nnz=result.output_nnz,
        target_nnz=result.target_nnz,
        cycle_updates=result.cycle_updates,
        wall_time=result.wall_time,
        dual_shift=dual_shift,
    )
    return primal_new, dual_new, backend_result


__all__ = ["sparsify_restricted_lp_output"]
