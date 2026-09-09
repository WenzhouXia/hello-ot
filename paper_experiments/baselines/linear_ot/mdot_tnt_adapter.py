"""
CN: 使用 KeOps 的 MDOT-TNT 线性 OT baseline adapter。
EN: MDOT-TNT linear-OT baseline adapter using KeOps.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch as th

from .mdot_tnt.mdot import adjust_schedule, smooth_marginals
from .mdot_tnt.truncated_newton import TruncatedNewtonProjector
from .ott_sinkhorn import (
    ImplicitSinkhornTransport,
    OttSinkhornDualSolution,
    evaluate_ott_sinkhorn_duals,
    negative_dot_cost_std_float64,
)
from .problem import LinearOTProblem
from .result import LinearOTResult


class TruncatedNewtonProjectorKeOps(TruncatedNewtonProjector):
    """
    CN: 基于 PyKeOps 的 O(n) 显存 Truncated Newton 投影器。
    EN: Truncated Newton projector with O(n) memory footprint using PyKeOps.
    """

    def project_keops(
        self,
        x_pts: th.Tensor,
        y_pts: th.Tensor,
        c_max_val: float,
        gamma_val: float,
        log_r: th.Tensor,
        log_c: th.Tensor,
        eps_d: Union[float, th.Tensor],
        u: th.Tensor,
        v: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Any], bool]:
        """
        CN: 使用 PyKeOps 符号化 LazyTensor 求解边缘对偶势，不物化 N x M 矩阵。
        EN: Solve marginal dual potentials using symbolic PyKeOps LazyTensors without materializing N x M matrices.
        """
        from pykeops.torch import LazyTensor

        logs: Dict[str, Any] = {
            "errs": [],
            "ls_func_cnt": 0,
            "chisinkhorn_steps": 0,
            "newtonsolve_steps": 0,
            "deltas": [],
            "all_newtonsolve_steps": [],
        }
        success_fn = lambda err_: (err_ < 10.0 * eps_d).all()

        r = log_r.exp()
        c = log_c.exp()

        x_i = LazyTensor(x_pts[:, None, :])
        y_j = LazyTensor(y_pts[None, :, :])
        d_ij = ((x_i - y_j) ** 2).sum(-1) / float(c_max_val)
        log_c_j = LazyTensor(th.log(c)[None, :, None])

        self.LSE_r = lambda v_: ((LazyTensor(v_[None, :, None]) - gamma_val * d_ij).logsumexp(dim=1)).squeeze(-1)
        self.LSE_c = lambda u_: ((LazyTensor(u_[:, None, None]) - gamma_val * d_ij).logsumexp(dim=0)).squeeze(-1)

        log_c_p = v + self.LSE_c(u)
        v = v + (log_c - log_c_p)
        log_r_p = u + self.LSE_r(v)
        k = 8

        u, v, log_r_p, err, k_ = self.chi_sinkhorn(u, v, log_r, log_c, log_r_p, eps_d ** (2.0 / 5.0))
        r_p = log_r_p.exp()
        logs["errs"].append(err)
        logs["chisinkhorn_steps"] = k_
        k += k_

        while err > eps_d:
            beta = 0.5
            eta_k = th.max(err, 0.9 * (eps_d / err))
            grad_k = r_p - r
            self.rho = th.max(th.zeros_like(self.rho), self.rho)

            u_i = LazyTensor(u[:, None, None])
            v_j = LazyTensor(v[None, :, None])
            p_ij = (u_i + v_j - gamma_val * d_ij).exp()

            diag_ppc = ((2.0 * v_j - log_c_j - 2.0 * gamma_val * d_ij).logsumexp(dim=1).squeeze(-1) + 2.0 * u).exp()
            k += 8

            delta_u, delta_v, matmul_cnt, rho, pcg_success = self.newton_solve_keops(
                p_ij, c, diag_ppc, grad_k, r_p, err, beta=beta, eta_k=eta_k, maxIter=5000
            )

            if not pcg_success:
                k += matmul_cnt
                logs["n_iter"] = k
                return u, v, logs, bool(success_fn(err).item())

            self.rho = th.max(th.zeros_like(self.rho), 1.0 - (1.0 - rho) * 4.0)
            k += matmul_cnt
            logs["newtonsolve_steps"] += matmul_cnt

            alpha = th.ones_like(self.rho)
            log_c_p = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
            k += 4
            linear_decr = -(grad_k * delta_u).sum(-1, keepdim=True)
            if not linear_decr > 0:
                logs["n_iter"] = k
                return u, v, logs, bool(success_fn(err).item())

            armijo = log_c_p.exp().sum(-1, keepdim=True) - 1.0 <= 0.99 * alpha * linear_decr
            while not armijo:
                alpha *= 0.5
                if alpha < 1.0e-9:
                    logs["n_iter"] = k
                    return u, v, logs, bool(success_fn(err).item())
                log_c_p = v + alpha * delta_v + self.LSE_c(u + alpha * delta_u)
                k += 4
                logs["ls_func_cnt"] += 4
                armijo = log_c_p.exp().sum(-1, keepdim=True) - 1.0 <= 0.99 * alpha * linear_decr

            u = u + alpha * delta_u
            v = v + alpha * delta_v

            err_before_sk = (c - log_c_p.exp()).abs().sum(-1)
            err_before_sk += (r - (u + self.LSE_r(v)).exp()).abs().sum(-1)

            v = v + (log_c - log_c_p)
            log_r_p = u + self.LSE_r(v)
            k += 4

            u, v, log_r_p, err, k_ = self.chi_sinkhorn(
                u, v, log_r, log_c, log_r_p, eps_d ** (2.0 / 5.0)
            )
            r_p = log_r_p.exp()
            logs["chisinkhorn_steps"] += k_
            k += k_

            logs["errs"].append(err)
            logs["deltas"].append(
                th.min((logs["errs"][-2] - err_before_sk) / ((1.0 - eta_k) * logs["errs"][-2])).item()
            )

        if u.isnan().any() or v.isnan().any():
            raise ValueError("NaNs encountered in u or v")

        logs["n_iter"] = k
        delta_u = log_r - log_r_p
        u = u + delta_u
        return u, v, logs, True

    def newton_solve_keops(
        self,
        p_ij: Any,
        c: th.Tensor,
        diag_ppc: th.Tensor,
        grad_k: th.Tensor,
        r_p: th.Tensor,
        err: th.Tensor,
        beta: float = 0.5,
        eta_k: Union[float, th.Tensor] = 0.5,
        maxIter: int = 5000,
    ) -> Tuple[th.Tensor, th.Tensor, int, th.Tensor, bool]:
        from pykeops.torch import LazyTensor

        rho = self.rho
        tol = err * eta_k

        def mml(x_: th.Tensor) -> th.Tensor:
            x_i_var = LazyTensor(x_[:, None, None])
            w = (x_i_var * p_ij).sum(dim=0).squeeze(-1) / c
            w_j = LazyTensor(w[None, :, None])
            return (w_j * p_ij).sum(dim=1).squeeze(-1)

        m_func = lambda rho_: r_p - rho_ * diag_ppc
        m_rho = m_func(th.ones_like(self.rho))
        m_rho[m_rho <= 0.0] = m_rho[m_rho > 0.0].min()

        x0 = -grad_k / m_rho
        ppc_x0 = mml(x0)
        matmul_cnt = 2
        r_p_x0 = r_p * x0

        x = x0.clone()
        ppc_x = ppc_x0.clone()
        r_p_x = r_p_x0.clone()

        res_true = r_p_x0 - ppc_x + grad_k
        linear_decr = (x * -grad_k).sum(-1)
        if (linear_decr <= 0.0).all():
            raise ValueError("Linear decrease condition not satisfied")

        r_true_norm = res_true.norm(p=1, dim=-1)
        best_sol = x.clone()
        best_r_true_norm = r_true_norm.clone()

        done = False
        success = True

        while (best_r_true_norm > tol).any():
            best_sol[r_true_norm < best_r_true_norm] = x[r_true_norm < best_r_true_norm]
            best_r_true_norm = th.min(r_true_norm, best_r_true_norm)

            rho[r_true_norm > tol] = 1.0 - (1.0 - rho[r_true_norm > tol]) * 0.25
            m_rho = m_func(rho)

            if matmul_cnt > 0:
                x = x0.clone()
                ppc_x = ppc_x0.clone()
                r_p_x = r_p_x0.clone()

            fr_x = r_p_x - rho * ppc_x
            res = fr_x + grad_k

            res_true = r_p_x - ppc_x + grad_k
            r_true_norm = res_true.norm(p=1, dim=-1)
            best_r_true_norm = th.min(r_true_norm, best_r_true_norm)
            linear_decr = (x * -grad_k).sum(-1)
            if (best_r_true_norm < tol).all() and (linear_decr > 0.0).all():
                break

            y = res / m_rho
            p = -y.clone()
            ry_old = (res * y).sum(-1, keepdim=True)

            r_norm = res.norm(p=1, dim=-1)
            while (r_norm > 0.5 * (1.0 - beta) * tol)[best_r_true_norm > tol].any():
                ppc_p = mml(p)
                matmul_cnt += 2
                fr_p = (r_p * p) - rho * ppc_p

                quad = (fr_p * p).sum(-1, keepdim=True)
                if (quad <= 0.0)[best_r_true_norm > tol].any():
                    x = best_sol.clone()
                    done = True
                    success = bool((best_r_true_norm < err).item())
                    rho = th.zeros_like(self.rho)
                    break

                alpha = ry_old / quad
                x += alpha * p
                res += alpha * fr_p
                r_norm = res.norm(p=1, dim=-1)

                ppc_x += alpha * ppc_p
                r_p_x = r_p * x
                res_true = r_p_x - ppc_x + grad_k
                r_true_norm = res_true.norm(p=1, dim=-1)
                best_sol[r_true_norm < best_r_true_norm] = x[r_true_norm < best_r_true_norm]
                best_r_true_norm = th.min(r_true_norm, best_r_true_norm)

                linear_decr = (x * -grad_k).sum(-1)
                if (best_r_true_norm <= tol).all() and (linear_decr > 0.0).all():
                    done = True
                    success = True
                    break

                if matmul_cnt > 2 * int(maxIter):
                    warnings.warn("PCG did not converge.")
                    done = True
                    success = False
                    break

                y = res / m_rho
                ry_new = (res * y).sum(-1, keepdim=True)
                p = -y + (ry_new / ry_old) * p
                ry_old = ry_new.clone()

            if done:
                break

        if (r_true_norm <= tol).all():
            success = True

        x = best_sol
        x_i_var = LazyTensor(x[:, None, None])
        pc_x = (x_i_var * p_ij).sum(dim=0).squeeze(-1) / c
        matmul_cnt += 1
        return x, -pc_x, matmul_cnt, rho, success


def mdot_keops(
    x: th.Tensor,
    y: th.Tensor,
    r: th.Tensor,
    c: th.Tensor,
    *,
    c_max: float,
    gamma_f: float,
    gamma_i: float = 16.0,
    p: float = 1.5,
    q: float = 2.0,
) -> Tuple[th.Tensor, th.Tensor, float, int, Dict[str, Any]]:
    """
    CN: 采用 PyKeOps O(n) 内存的 MDOT-TruncatedNewton 求解外循环。
    EN: Outer annealing loop for MDOT-TruncatedNewton with PyKeOps O(n) memory footprint.
    """
    projector = TruncatedNewtonProjectorKeOps(device=x.device, dtype=x.dtype)

    h_r = -(r * (r + 1.0e-30).log()).sum(-1)
    h_c = -(c * (c + 1.0e-30).log()).sum(-1)
    h_min = th.min(h_r, h_c)
    eps_fn = lambda g_: h_min / (g_**p)

    logs: Dict[str, Any] = {
        "proj_logs": [],
        "eps": [],
    }

    t = 1
    done = False
    gamma = min(float(gamma_i), float(gamma_f))
    gammas = [0.0, gamma]

    while not done:
        done = abs(gamma - float(gamma_f)) < 1.0e-5
        eps_d = eps_fn(gamma)

        r_hat, c_hat = smooth_marginals(r, c, eps_d / 2.0, w_r=0.9, w_c=0.1)

        if t == 1:
            u_init, v_init = r_hat.log(), c_hat.log()
            u_cur, v_cur = u_init.clone(), v_init.clone()

        u_prev, v_prev = u_cur.clone(), v_cur.clone()
        u_cur, v_cur, proj_log, success = projector.project_keops(
            x, y, float(c_max), float(gamma), r_hat.log(), c_hat.log(), eps_d / 2.0, u_init, v_init
        )

        logs["proj_logs"].append(proj_log)
        if not success:
            u_cur = u_prev.clone()
            v_cur = v_prev.clone()
            gammas = gammas[:-1]
            break

        q = adjust_schedule(q, proj_log["deltas"])
        gamma = min(gamma * q, float(gamma_f))

        if not done:
            u_init = u_cur + (u_cur - u_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])
            v_init = v_cur + (v_cur - v_prev) * (gamma - gammas[-1]) / (gammas[-1] - gammas[-2])

        gammas.append(gamma)
        t += 1

    k_total = sum(log["n_iter"] for log in logs["proj_logs"]) + (t - 1)
    logs["success"] = success
    logs["gammas"] = gammas

    return u_cur, v_cur, gammas[-1], k_total, logs


def solve_mdot_tnt(
    problem: LinearOTProblem,
    *,
    gamma_f: float = 1024.0,
    backend: str = "keops",
    gamma_i: float = 16.0,
    batch_size: int = 2048,
    device_str: str = "cuda",
) -> LinearOTResult:
    """
    CN: 通过统一接口调用 MDOT-TNT KeOps，并无损复用 OTT-JAX 进行 FP64 隐式评测与 Rounding。
    EN: Invoke MDOT-TNT through KeOps, losslessly reusing OTT-JAX for FP64 implicit evaluation and rounding.
    """
    if problem.cost_type != "l2^2":
        raise ValueError("MDOT-TNT currently supports only cost_type='l2^2'.")
    if not th.cuda.is_available():
        raise RuntimeError("MDOT-TNT requires a CUDA GPU.")

    backend = str(backend).strip().lower()
    if backend != "keops":
        raise ValueError(f"Unsupported backend: {backend}. Only 'keops' is supported.")

    device = th.device(device_str)
    gamma_f_val = float(gamma_f)
    gamma_i_val = float(gamma_i)

    # CN: 遵循作者原设计，若 gamma_f > 1024 则使用 float64，否则保持 float32
    # EN: Follow author design: use float64 if gamma_f > 1024, else keep float32
    target_dtype = th.float64 if gamma_f_val > 1024.0 else th.float32

    # CN: 清理 GPU 异步残余，从点云和代价归一化开始计时
    # EN: Drain GPU asynchronous residuals, timing from point cloud and cost normalization
    th.cuda.synchronize(device)
    solve_t0 = time.perf_counter()

    # CN: PyKeOps 后端：O(n) 内存模式，避免显式实例化 N x M 矩阵
    # EN: PyKeOps backend: O(n) memory mode, avoiding explicit N x M matrix instantiation
    from pykeops.torch import LazyTensor

    x_t = th.as_tensor(problem.source_points, device=device, dtype=target_dtype)
    y_t = th.as_tensor(problem.target_points, device=device, dtype=target_dtype)
    r_t = th.as_tensor(problem.source_mass, device=device, dtype=target_dtype)
    c_t = th.as_tensor(problem.target_mass, device=device, dtype=target_dtype)

    x_i = LazyTensor(x_t[:, None, :])
    y_j = LazyTensor(y_t[None, :, :])
    d_ij = ((x_i - y_j) ** 2).sum(-1)
    c_max = float(d_ij.max(dim=1).max().item())

    u, v, gamma_out, k_total, logs = mdot_keops(
        x_t, y_t, r_t, c_t, c_max=c_max, gamma_f=gamma_f_val, gamma_i=gamma_i_val
    )
    th.cuda.synchronize(device)
    runtime_sec = float(time.perf_counter() - solve_t0)

    # CN: 启用 JAX float64 模式以严格执行标准化双精度 Rounding 与评测
    # EN: Enable JAX float64 mode for canonical double-precision rounding and evaluation
    import jax
    jax.config.update("jax_enable_x64", True)

    # CN: 将 MDOT 对偶变量严格解析映射为 OTT 对偶势：f = eps_ott * u, g = eps_ott * v
    # EN: Mathematically map MDOT dual variables to OTT dual potentials: f = eps_ott * u, g = eps_ott * v
    eps_eff = c_max / max(float(gamma_out), 1.0e-30)
    eps_ott = 0.5 * eps_eff
    f = (u.double() * eps_ott).detach().cpu().numpy()
    g = (v.double() * eps_ott).detach().cpu().numpy()

    cost_std = negative_dot_cost_std_float64(problem.source_points, problem.target_points)
    solution = OttSinkhornDualSolution(
        problem=problem,
        source_potential=f,
        target_potential=g,
        errors=np.array([0.0], dtype=np.float64),
        converged=bool(logs.get("success", True)),
        n_iters=int(k_total),
        regularization=float(eps_ott),
        cost_std=float(cost_std),
        runtime_sec=float(runtime_sec),
        batch_size=int(batch_size),
        dtype_name="float64",
        jax_version=str(getattr(jax, "__version__", "")),
        ott_version="",
    )

    evaluation, rounded_objective, details, eval_time, round_time = evaluate_ott_sinkhorn_duals(
        solution, block_size=int(batch_size)
    )

    transport = ImplicitSinkhornTransport(
        source_potential=f,
        target_potential=g,
        regularization=float(eps_ott),
        batch_size=int(batch_size),
        evaluation=evaluation,
        rounded_objective=float(rounded_objective),
    )

    method_name = f"mdot_tnt_{backend}"
    return LinearOTResult(
        method=method_name,
        solver_objective=float(evaluation.objective),
        runtime_sec=float(runtime_sec),
        transport_kind="implicit",
        transport=transport,
        converged=bool(logs.get("success", True)),
        status="converged" if logs.get("success", True) else "failed",
        diagnostics={
            "backend": str(backend),
            "gamma_f": float(gamma_f_val),
            "gamma_out": float(gamma_out),
            "gamma_i": float(gamma_i_val),
            "c_max": float(c_max),
            "eps_eff": float(eps_eff),
            "eps_ott": float(eps_ott),
            "k_total": int(k_total),
            "evaluation_time_sec": float(eval_time),
            "rounding_time_sec": float(round_time),
            "rounded_objective": float(rounded_objective),
            "cost_std": float(cost_std),
            "solver_backend": f"mdot_tnt.{backend}",
            "evaluation_details": {
                key: float(val) if isinstance(val, (int, float, np.generic)) else str(val)
                for key, val in details.items()
                if not isinstance(val, np.ndarray) or val.size == 1
            },
        },
    )
