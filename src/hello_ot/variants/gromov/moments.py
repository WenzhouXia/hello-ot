from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class GWStaticTerms:
    """
    CN: 平方欧氏 GW 目标的静态低秩项。
    EN: Static low-rank terms of squared-Euclidean GW objective.
    """

    u_s: np.ndarray
    v_s: np.ndarray
    u_t: np.ndarray
    v_t: np.ndarray
    constant: float


def _torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_torch_float32(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _sqeuclidean_lr_factors_torch(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CN: 将点云 X 解析分解为低秩距离矩阵因子 ||x_i - x_j||^2 = <u_i, v_j>。
    EN: Factorize point cloud X into rank-(d+2) factors ||x_i - x_j||^2 = <u_i, v_j>.
    """
    sqnorm = torch.sum(X * X, dim=1, keepdim=True)
    ones = torch.ones_like(sqnorm)
    scale = float(math.sqrt(2.0))
    u = torch.cat([ones, sqnorm, scale * X], dim=1)
    v = torch.cat([sqnorm, ones, -scale * X], dim=1)
    return u, v


def _rowwise_quadratic_form_torch(u: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.sum((u @ M) * u, dim=1)


def _sqeuclidean_cross_moments_from_sparse_gpu(
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    coupling_coo: sp.coo_matrix,
) -> torch.Tensor:
    """
    CN: 在 GPU 上高效累加稀疏 coupling 的低秩二阶交叉矩 v_s.T @ P @ v_t。
    EN: Accumulate cross-moment matrix v_s.T @ P @ v_t on GPU from sparse coupling.
    """
    device = v_s.device
    rows = torch.from_numpy(np.asarray(coupling_coo.row, dtype=np.int64)).to(device)
    cols = torch.from_numpy(np.asarray(coupling_coo.col, dtype=np.int64)).to(device)
    values = torch.from_numpy(np.asarray(coupling_coo.data, dtype=np.float32)).to(device)

    v_s_sampled = v_s[rows] * values.unsqueeze(1)
    v_t_sampled = v_t[cols]
    return v_s_sampled.T @ v_t_sampled


def linearized_sqeuclidean_gw_factors(
    source_X: np.ndarray,
    target_X: np.ndarray,
    coupling: Optional[sp.spmatrix],
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CN: 构造平方欧氏 GW 在当前 coupling 下的线性化双线性 OT 子问题代价因子。
    EN: Construct bilinear cost factors of linearized GW subproblem around current coupling.
    """
    device = _torch_device()
    source_X_gpu = _as_torch_float32(source_X, device)
    target_X_gpu = _as_torch_float32(target_X, device)
    source_mass_gpu = _as_torch_float32(source_mass, device)
    target_mass_gpu = _as_torch_float32(target_mass, device)

    u_s, v_s = _sqeuclidean_lr_factors_torch(source_X_gpu)
    u_t, v_t = _sqeuclidean_lr_factors_torch(target_X_gpu)

    mass_moment_s = v_s.T @ (source_mass_gpu.unsqueeze(1) * v_s)
    mass_moment_t = v_t.T @ (target_mass_gpu.unsqueeze(1) * v_t)

    a_vec_gpu = _rowwise_quadratic_form_torch(u_s, mass_moment_s)
    b_vec_gpu = _rowwise_quadratic_form_torch(u_t, mass_moment_t)

    if coupling is None:
        p_s = v_s.T @ source_mass_gpu.unsqueeze(1)
        p_t = v_t.T @ target_mass_gpu.unsqueeze(1)
        cross_core_gpu = p_s @ p_t.T
    elif sp.issparse(coupling):
        cross_core_gpu = _sqeuclidean_cross_moments_from_sparse_gpu(
            v_s, v_t, coupling.tocoo(copy=False)
        )
    else:
        dense_p = _as_torch_float32(np.asarray(coupling), device)
        cross_core_gpu = v_s.T @ (dense_p @ v_t)

    k_gpu = 2.0 * cross_core_gpu
    f_gpu = u_s @ k_gpu
    g_gpu = u_t

    f_cpu = f_gpu.detach().cpu().numpy()
    g_cpu = g_gpu.detach().cpu().numpy()
    a_vec_cpu = a_vec_gpu.detach().cpu().numpy()
    b_vec_cpu = b_vec_gpu.detach().cpu().numpy()

    return f_cpu, g_cpu, a_vec_cpu, b_vec_cpu


def compute_sqeuclidean_gw_static_terms(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> GWStaticTerms:
    """
    CN: 预计算平方欧氏 GW 目标函数的静态自相关项。
    EN: Precompute static self-correlation terms of squared-Euclidean GW objective.
    """
    device = _torch_device()
    source_X_gpu = _as_torch_float32(source_X, device)
    target_X_gpu = _as_torch_float32(target_X, device)
    source_mass_gpu = _as_torch_float32(source_mass, device)
    target_mass_gpu = _as_torch_float32(target_mass, device)

    u_s_gpu, v_s_gpu = _sqeuclidean_lr_factors_torch(source_X_gpu)
    u_t_gpu, v_t_gpu = _sqeuclidean_lr_factors_torch(target_X_gpu)

    mass_moment_s = v_s_gpu.T @ (source_mass_gpu.unsqueeze(1) * v_s_gpu)
    mass_moment_t = v_t_gpu.T @ (target_mass_gpu.unsqueeze(1) * v_t_gpu)

    a_vec_gpu = _rowwise_quadratic_form_torch(u_s_gpu, mass_moment_s)
    b_vec_gpu = _rowwise_quadratic_form_torch(u_t_gpu, mass_moment_t)
    constant_gpu = torch.dot(source_mass_gpu, a_vec_gpu) + torch.dot(target_mass_gpu, b_vec_gpu)

    return GWStaticTerms(
        u_s=u_s_gpu.detach().cpu().numpy(),
        v_s=v_s_gpu.detach().cpu().numpy(),
        u_t=u_t_gpu.detach().cpu().numpy(),
        v_t=v_t_gpu.detach().cpu().numpy(),
        constant=float(constant_gpu.item()),
    )


def compute_sqeuclidean_gw_objective(
    terms: GWStaticTerms,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    coupling: Optional[sp.spmatrix],
) -> float:
    """
    CN: 利用低秩解析结构在 O((n+m)d) 复杂度下精确评估平方欧氏 GW 目标函数值。
    EN: Evaluate squared-Euclidean GW objective value exactly in O((n+m)d) time using low-rank factors.
    """
    device = _torch_device()
    v_s = _as_torch_float32(terms.v_s, device)
    u_t = _as_torch_float32(terms.u_t, device)
    u_s = _as_torch_float32(terms.u_s, device)
    v_t = _as_torch_float32(terms.v_t, device)
    source_mass_gpu = _as_torch_float32(source_mass, device)
    target_mass_gpu = _as_torch_float32(target_mass, device)

    if coupling is None:
        left = (v_s.T @ source_mass_gpu.unsqueeze(1)) @ (u_t.T @ target_mass_gpu.unsqueeze(1)).T
        right = (u_s.T @ source_mass_gpu.unsqueeze(1)) @ (v_t.T @ target_mass_gpu.unsqueeze(1)).T
    elif sp.issparse(coupling):
        coupling_coo = coupling.tocoo(copy=False)
        left = _sqeuclidean_cross_moments_from_sparse_gpu(v_s, u_t, coupling_coo)
        right = _sqeuclidean_cross_moments_from_sparse_gpu(u_s, v_t, coupling_coo)
    else:
        dense_p = _as_torch_float32(np.asarray(coupling), device)
        left = v_s.T @ (dense_p @ u_t)
        right = u_s.T @ (dense_p @ v_t)

    cross_term = torch.sum(left * right)
    obj = torch.tensor(terms.constant, device=device, dtype=torch.float32) - 2.0 * cross_term
    return float(obj.item())


__all__ = [
    "GWStaticTerms",
    "compute_sqeuclidean_gw_objective",
    "compute_sqeuclidean_gw_static_terms",
    "linearized_sqeuclidean_gw_factors",
]
