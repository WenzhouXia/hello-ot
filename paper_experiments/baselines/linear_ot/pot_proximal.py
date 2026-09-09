from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .problem import LinearOTProblem
from .result import LinearOTResult, TransportEvaluation


@dataclass(frozen=True)
class ImplicitProximalTransport:
    """
    CN: 保存 online IPOT 的紧凑对偶表示及已分块计算的 scalar evaluation。
    EN: Store a compact online IPOT dual representation and its blockwise scalar evaluation.
    """

    source_potential: np.ndarray
    target_potential: np.ndarray
    regularization: float
    batch_size: int
    evaluation: TransportEvaluation
    rounded_objective: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.evaluation.transport_shape


def solve_pot_proximal_point(
    problem: LinearOTProblem,
    *,
    backend: str = "dense",
    max_iterations: int = 1_000,
    tolerance: float = 1.0e-5,
    inner_iterations: int = 1,
    inner_regularization: float = 1.0e-3,
    dtype_name: str = "float32",
) -> LinearOTResult:
    """
    CN: 使用仅修正已知符号错误的 POT 0.9.7.post1 log-domain IPOT，支持 dense 与 online keops 后端。
    EN: Use POT 0.9.7.post1 log-domain IPOT with only the known sign error corrected, supporting dense and online keops backends.

    CN: `inner_regularization` 是相对于 population std(C) 的无量纲系数，
    实际 proximal 参数为 `inner_regularization * std(C)`。
    EN: `inner_regularization` is dimensionless relative to population std(C),
    and the actual proximal parameter is `inner_regularization * std(C)`.
    """
    if problem.cost_type != "l2^2":
        raise ValueError("pot_proximal_point currently supports only cost_type='l2^2'.")
    try:
        import ot
        import torch
    except Exception as exc:
        raise ImportError("POT and PyTorch are required for pot_proximal_point.") from exc
    if getattr(getattr(ot, "batch", None), "solve_batch", None) is None:
        raise RuntimeError(f"pot_proximal_point requires POT >= 0.9.7, got {getattr(ot, '__version__', 'unknown')}.")
    if not torch.cuda.is_available():
        raise RuntimeError("pot_proximal_point requires a CUDA GPU.")
    if int(max_iterations) <= 0 or int(inner_iterations) <= 0:
        raise ValueError("max_iterations and inner_iterations must be positive.")
    if float(tolerance) <= 0.0 or float(inner_regularization) <= 0.0:
        raise ValueError("tolerance and inner_regularization must be positive.")

    backend_mode = str(backend).strip().lower()
    if backend_mode not in {"dense", "keops"}:
        raise ValueError(f"backend must be 'dense' or 'keops', got {backend!r}.")

    device = torch.device("cuda")
    dtype = torch.float32 if str(dtype_name) == "float32" else torch.float64
    if str(dtype_name) not in {"float32", "float64"}:
        raise ValueError("dtype_name must be 'float32' or 'float64'.")

    # CN: 清空 warmup 遗留的异步工作，再从 CPU 点云构造代价开始计时。
    # EN: Drain asynchronous warmup work, then time from CPU point-cloud cost construction.
    torch.cuda.synchronize(device)
    solve_t0 = time.perf_counter()

    if backend_mode == "keops":
        return _solve_pot_proximal_keops(
            problem,
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
            inner_iterations=int(inner_iterations),
            inner_regularization=float(inner_regularization),
            dtype_name=str(dtype_name),
            device=device,
            dtype=dtype,
            solve_t0=solve_t0,
        )

    cost = _squared_l2_cost(problem.source_points, problem.target_points)
    cost_std = _population_cost_std(cost)
    relative_inner_regularization = float(inner_regularization)
    actual_inner_regularization = relative_inner_regularization * cost_std
    cost_t = torch.as_tensor(cost, dtype=dtype, device=device).unsqueeze(0)
    source_mass_t = torch.as_tensor(problem.source_mass, dtype=dtype, device=device).unsqueeze(0)
    target_mass_t = torch.as_tensor(problem.target_mass, dtype=dtype, device=device).unsqueeze(0)
    torch.cuda.synchronize(device)
    output = _corrected_log_domain_ipot(
        cost_t,
        source_mass_t,
        target_mass_t,
        max_iter=int(max_iterations),
        tol=float(tolerance),
        inner_iter=int(inner_iterations),
        inner_reg=actual_inner_regularization,
    )
    plan_t = output["T"][0]
    objective_t = torch.sum(plan_t * cost_t[0])
    source_marginal_l2_t = torch.linalg.vector_norm(torch.sum(plan_t, dim=1) - source_mass_t[0], ord=2)
    source_marginal_l1_t = torch.sum(torch.abs(torch.sum(plan_t, dim=1) - source_mass_t[0]))
    target_marginal_l1_t = torch.sum(torch.abs(torch.sum(plan_t, dim=0) - target_mass_t[0]))
    plan = plan_t.detach().cpu().numpy()
    objective = float(objective_t.detach().cpu().item())
    source_marginal_l2 = float(source_marginal_l2_t.detach().cpu().item())
    source_marginal_l1 = float(source_marginal_l1_t.detach().cpu().item())
    target_marginal_l1 = float(target_marginal_l1_t.detach().cpu().item())
    torch.cuda.synchronize(device)
    runtime_sec = float(time.perf_counter() - solve_t0)
    log_n_iter = int(output["n_iters"])
    completed_iterations = log_n_iter + 1
    converged = bool(output["converged"])
    return LinearOTResult(
        method="pot_proximal_point",
        solver_objective=objective,
        runtime_sec=runtime_sec,
        transport_kind="dense",
        transport=plan,
        converged=converged,
        status="converged" if converged else "iteration_limit",
        diagnostics={
            "pot_version": str(getattr(ot, "__version__", "")),
            "torch_version": str(getattr(torch, "__version__", "")),
            "max_iterations": int(max_iterations),
            "completed_iterations": int(completed_iterations),
            "tolerance": float(tolerance),
            "source_marginal_l2_error": source_marginal_l2,
            "source_marginal_l1_error": source_marginal_l1,
            "target_marginal_l1_error": target_marginal_l1,
            "inner_iterations": int(inner_iterations),
            "inner_regularization": actual_inner_regularization,
            "inner_regularization_relative": relative_inner_regularization,
            "inner_regularization_actual": actual_inner_regularization,
            "inner_regularization_scale": "population_std_squared_l2_cost",
            "cost_std": cost_std,
            "convergence_criterion": "pot_source_marginal_l2_checked_every_10_iterations",
            "dtype": str(dtype_name),
            "backend": "dense",
            "solver_backend": "corrected_pot_style_log_domain_ipot.torch_cuda",
            "pot_0_9_7_sign_correction": "log_T_minus_C_over_beta",
        },
    )


def _solve_pot_proximal_keops(
    problem: LinearOTProblem,
    *,
    max_iterations: int,
    tolerance: float,
    inner_iterations: int,
    inner_regularization: float,
    dtype_name: str,
    device: torch.device,
    dtype: torch.dtype,
    solve_t0: float,
) -> LinearOTResult:
    """
    CN: 基于 PyKeOps 的 O(N) 显存 online IPOT 实现。
    EN: O(N) memory online IPOT implementation based on PyKeOps.
    """
    try:
        from pykeops.torch import LazyTensor
    except Exception as exc:
        raise ImportError("PyKeOps is required for backend='keops'.") from exc

    import ot
    import torch

    n, m = problem.shape

    source_t = torch.as_tensor(problem.source_points, dtype=dtype, device=device)
    target_t = torch.as_tensor(problem.target_points, dtype=dtype, device=device)
    source_mass_t = torch.as_tensor(problem.source_mass, dtype=dtype, device=device)
    target_mass_t = torch.as_tensor(problem.target_mass, dtype=dtype, device=device)

    # CN: 严格使用 float64 精确累计 squared-L2 cost 的 population standard deviation。
    # EN: Strictly use float64 to compute the population standard deviation of the squared-L2 cost.
    source_f64 = torch.as_tensor(problem.source_points, dtype=torch.float64, device=device)
    target_f64 = torch.as_tensor(problem.target_points, dtype=torch.float64, device=device)
    x_i_f64 = LazyTensor(source_f64[:, None, :])
    y_j_f64 = LazyTensor(target_f64[None, :, :])
    c_ij_f64 = ((x_i_f64 - y_j_f64) ** 2).sum(-1)

    sum_c = float(c_ij_f64.sum(dim=1).sum().item())
    sum_c2 = float((c_ij_f64 ** 2).sum(dim=1).sum().item())
    total_pairs = float(n * m)
    mean_c = sum_c / total_pairs
    mean_c2 = sum_c2 / total_pairs
    cost_std = float(math.sqrt(max(0.0, mean_c2 - mean_c * mean_c)))
    if not math.isfinite(cost_std) or cost_std <= 0.0:
        raise ValueError("Squared-L2 cost population standard deviation must be finite and positive.")

    actual_inner_regularization = float(inner_regularization) * cost_std

    # CN: 符号化 LazyTensor 代价核
    # EN: Symbolic LazyTensor cost kernel
    x_i = LazyTensor(source_t[:, None, :])
    y_j = LazyTensor(target_t[None, :, :])
    c_ij = ((x_i - y_j) ** 2).sum(-1)

    log_a = torch.log(source_mass_t)[:, None]  # (n, 1)
    log_b = torch.log(target_mass_t)[:, None]  # (m, 1)

    u_pot = torch.zeros(n, 1, dtype=dtype, device=device)
    v_pot = torch.zeros(m, 1, dtype=dtype, device=device)
    target_scaling = torch.zeros(m, 1, dtype=dtype, device=device)
    converged = False

    last_n_iter = 0
    for n_iters in range(int(max_iterations)):
        last_n_iter = n_iters
        gamma = float(n_iters + 1) / float(actual_inner_regularization)

        for _ in range(int(inner_iterations)):
            v_tilde = LazyTensor((v_pot + target_scaling)[None, :, :])
            lse_j = (-gamma * c_ij + v_tilde).logsumexp(dim=1)
            source_scaling = log_a - u_pot - lse_j

            u_tilde = LazyTensor((u_pot + source_scaling)[:, None, :])
            lse_i = (-gamma * c_ij + u_tilde).logsumexp(dim=0)
            target_scaling = log_b - v_pot - lse_i

        u_pot = u_pot + source_scaling
        v_pot = v_pot + target_scaling

        if n_iters % 10 == 0:
            u_lazy = LazyTensor(u_pot[:, None, :])
            v_lazy = LazyTensor(v_pot[None, :, :])
            kernel_lazy = (-gamma * c_ij + u_lazy + v_lazy).exp()
            row_sum = kernel_lazy.sum(dim=1).squeeze(-1)
            source_error = torch.max(torch.linalg.vector_norm(row_sum - source_mass_t, ord=2, dim=-1))
            if float(source_error.item()) < float(tolerance):
                converged = True
                break

    # CN: 最终在线计算评测指标与目标函数
    # EN: Final online evaluation of metrics and objective
    gamma = float(last_n_iter + 1) / float(actual_inner_regularization)
    u_lazy = LazyTensor(u_pot[:, None, :])
    v_lazy = LazyTensor(v_pot[None, :, :])
    kernel_lazy = (-gamma * c_ij + u_lazy + v_lazy).exp()

    row_sum_t = kernel_lazy.sum(dim=1).squeeze(-1)  # (n,)
    col_sum_t = kernel_lazy.sum(dim=0).squeeze(-1)  # (m,)
    objective_t = (c_ij * kernel_lazy).sum(dim=1).sum()

    source_marginal_l2 = float(torch.linalg.vector_norm(row_sum_t - source_mass_t, ord=2).item())
    source_marginal_l1 = float(torch.sum(torch.abs(row_sum_t - source_mass_t)).item())
    target_marginal_l1 = float(torch.sum(torch.abs(col_sum_t - target_mass_t)).item())

    row_marginal_np = row_sum_t.detach().cpu().numpy().astype(np.float64)
    col_marginal_np = col_sum_t.detach().cpu().numpy().astype(np.float64)
    row_residual = row_marginal_np - problem.source_mass
    col_residual = col_marginal_np - problem.target_mass
    row_l2 = float(np.linalg.norm(row_residual))
    col_l2 = float(np.linalg.norm(col_residual))
    abs_error = float(math.sqrt(row_l2 * row_l2 + col_l2 * col_l2))
    bound_norm = float(np.linalg.norm(np.concatenate([problem.source_mass, problem.target_mass])))
    primal_feasibility = float(abs_error / (1.0 + bound_norm))
    mass_error = float(abs(float(col_marginal_np.sum()) - float(problem.source_mass.sum())))

    # CN: 使用 KeOps 在线执行 Altschuler et al. primal feasible rounding
    # EN: Online Altschuler et al. primal feasible rounding via KeOps
    from paper_experiments.common.feasible_rounding import _rank_one_cost

    row_scale = torch.minimum(
        torch.ones_like(source_mass_t),
        source_mass_t / torch.clamp(row_sum_t, min=1.0e-12),
    )
    row_scale_i = LazyTensor(row_scale[:, None, None])
    kernel_row = row_scale_i * kernel_lazy
    col_after = kernel_row.sum(dim=0).squeeze(-1)

    col_scale = torch.minimum(
        torch.ones_like(target_mass_t),
        target_mass_t / torch.clamp(col_after, min=1.0e-12),
    )
    col_scale_j = LazyTensor(col_scale[None, :, None])
    kernel_rounded = col_scale_j * kernel_row

    rounded_row = kernel_rounded.sum(dim=1).squeeze(-1)
    rounded_col = kernel_rounded.sum(dim=0).squeeze(-1)
    base_obj = float((c_ij * kernel_rounded).sum(dim=1).sum().item())

    s_res = torch.clamp(source_mass_t - rounded_row, min=0.0).detach().cpu().numpy().astype(np.float64)
    t_res = torch.clamp(target_mass_t - rounded_col, min=0.0).detach().cpu().numpy().astype(np.float64)
    res_mass = float(s_res.sum())

    if res_mass > 1.0e-15:
        correction = _rank_one_cost(
            problem.source_points,
            problem.target_points,
            s_res,
            t_res,
            residual_mass=res_mass,
            cost_type=problem.cost_type,
            block_size=256,
        )
    else:
        correction = 0.0
    rounded_objective = float(base_obj + correction)
    objective = float(objective_t.item())

    torch.cuda.synchronize(device)
    runtime_sec = float(time.perf_counter() - solve_t0)

    evaluation = TransportEvaluation(
        objective=objective,
        primal_feasibility=primal_feasibility,
        primal_l2_abs_error=abs_error,
        row_marginal_l2_error=row_l2,
        col_marginal_l2_error=col_l2,
        transport_mass_error=mass_error,
        transport_kind="implicit",
        transport_shape=problem.shape,
        transport_nnz=n * m,
    )

    transport = ImplicitProximalTransport(
        source_potential=u_pot.squeeze(-1).detach().cpu().numpy().astype(np.float64),
        target_potential=v_pot.squeeze(-1).detach().cpu().numpy().astype(np.float64),
        regularization=actual_inner_regularization,
        batch_size=0,
        evaluation=evaluation,
        rounded_objective=rounded_objective,
    )

    completed_iterations = last_n_iter + 1
    return LinearOTResult(
        method="pot_proximal_point",
        solver_objective=objective,
        runtime_sec=runtime_sec,
        transport_kind="implicit",
        transport=transport,
        converged=converged,
        status="converged" if converged else "iteration_limit",
        diagnostics={
            "pot_version": str(getattr(ot, "__version__", "")),
            "torch_version": str(getattr(torch, "__version__", "")),
            "max_iterations": int(max_iterations),
            "completed_iterations": int(completed_iterations),
            "tolerance": float(tolerance),
            "source_marginal_l2_error": source_marginal_l2,
            "source_marginal_l1_error": source_marginal_l1,
            "target_marginal_l1_error": target_marginal_l1,
            "inner_iterations": int(inner_iterations),
            "inner_regularization": actual_inner_regularization,
            "inner_regularization_relative": float(inner_regularization),
            "inner_regularization_actual": actual_inner_regularization,
            "inner_regularization_scale": "population_std_squared_l2_cost",
            "cost_std": cost_std,
            "convergence_criterion": "pot_source_marginal_l2_checked_every_10_iterations",
            "dtype": str(dtype_name),
            "backend": "keops",
            "solver_backend": f"corrected_pot_style_log_domain_ipot.keops_{dtype_name}",
            "pot_0_9_7_sign_correction": "log_T_minus_C_over_beta",
            "rounded_objective": rounded_objective,
        },
    )


def _population_cost_std(cost: np.ndarray) -> float:
    """
    CN: 以 float64 累计 squared-L2 cost 的 population standard deviation。
    EN: Accumulate the population standard deviation of the squared-L2 cost in float64.
    """
    value = float(np.std(np.asarray(cost), dtype=np.float64, ddof=0))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Squared-L2 cost population standard deviation must be finite and positive.")
    return value


def _corrected_log_domain_ipot(
    cost: Any,
    source_mass: Any,
    target_mass: Any,
    *,
    max_iter: int,
    tol: float,
    inner_iter: int,
    inner_reg: float,
) -> dict[str, Any]:
    """
    CN: 逐行复现 POT 0.9.7.post1，仅将 affinity 修正为 log(Q)=log(T)-C/beta。
    EN: Reproduce POT 0.9.7.post1 line by line, correcting only the affinity to log(Q)=log(T)-C/beta.
    """
    import torch

    log_source = torch.log(source_mass)
    log_target = torch.log(target_mass)
    log_plan = torch.zeros_like(cost)
    source_scaling = torch.zeros_like(source_mass)
    target_scaling = torch.zeros_like(target_mass)
    converged = False

    for n_iters in range(int(max_iter)):
        # CN: POT 0.9.7.post1 此处错误地使用了 -log_plan-C/beta。
        # EN: POT 0.9.7.post1 incorrectly used -log_plan-C/beta here.
        projected_kernel = log_plan - cost / float(inner_reg)
        for _ in range(int(inner_iter)):
            source_scaling = log_source - torch.logsumexp(
                projected_kernel + target_scaling[:, None, :], dim=2
            )
            target_scaling = log_target - torch.logsumexp(
                projected_kernel + source_scaling[:, :, None], dim=1
            )
        log_plan = projected_kernel + source_scaling[:, :, None] + target_scaling[:, None, :]

        # CN: 严格保留 POT 每 10 步仅检查 source marginal L2 residual 的停止规则。
        # EN: Strictly retain POT's source-marginal-only stopping rule checked every ten steps.
        if n_iters % 10 == 0:
            plan = torch.exp(log_plan)
            source_error = torch.max(
                torch.linalg.vector_norm(torch.sum(plan, dim=2) - source_mass, ord=2, dim=1)
            )
            if float(source_error.item()) < float(tol):
                converged = True
                break

    plan = torch.exp(log_plan)
    return {
        "T": plan,
        "log_T": log_plan,
        "n_iters": int(n_iters),
        "converged": converged,
        "potentials": (source_scaling, target_scaling),
    }


def _squared_l2_cost(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float32, order="C")
    target_array = np.asarray(target, dtype=np.float32, order="C")
    source_norm = np.sum(source_array * source_array, axis=1)
    target_norm = np.sum(target_array * target_array, axis=1)
    cost = source_norm[:, None] + target_norm[None, :] - 2.0 * (source_array @ target_array.T)
    np.maximum(cost, 0.0, out=cost)
    return np.asarray(cost, dtype=np.float32, order="C")
