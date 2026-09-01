from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .wrapper import SolverResult


@dataclass(frozen=True)
class _Residuals:
    primal_objective: float
    dual_objective: float
    absolute_primal: float
    relative_primal: float
    absolute_dual: float
    relative_dual: float
    relative_gap: float


def _as_1d_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    CN: 将 restricted-OT 输入规范化为指定设备上的连续一维张量。
    EN: Normalize a restricted-OT input to a contiguous one-dimensional tensor on the requested device.
    """
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype).contiguous().view(-1)
    return torch.as_tensor(value, device=device, dtype=dtype).contiguous().view(-1)


def _scatter_marginals(
    values: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    n_source: int,
    n_target: int,
) -> torch.Tensor:
    """
    CN: 通过 active-support 端点累加计算运输计划的两侧边际。
    EN: Compute both transport marginals by accumulating values at active-support endpoints.
    """
    source = torch.zeros(n_source, dtype=values.dtype, device=values.device)
    target = torch.zeros(n_target, dtype=values.dtype, device=values.device)
    source.index_add_(0, rows, values)
    target.index_add_(0, cols, values)
    return torch.cat((source, target))


def _gather_transpose(
    dual: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    n_source: int,
) -> torch.Tensor:
    """
    CN: 利用二部图端点 gather 计算 A^T y，避免构造通用稀疏矩阵。
    EN: Compute A^T y with bipartite endpoint gathers without constructing a generic sparse matrix.
    """
    return dual[:n_source][rows] + dual[n_source:][cols]


def _residuals(
    *,
    primal: torch.Tensor,
    dual: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    costs: torch.Tensor,
    masses: torch.Tensor,
    n_source: int,
    n_target: int,
) -> _Residuals:
    """
    CN: 在未缩放的 restricted OT 问题上计算 L2 KKT 残差与相对对偶间隙。
    EN: Compute L2 KKT residuals and the relative duality gap on the unscaled restricted OT problem.
    """
    marginal_residual = _scatter_marginals(primal, rows, cols, n_source, n_target) - masses
    dual_scores = _gather_transpose(dual, rows, cols, n_source) - costs
    dual_residual = torch.clamp_min(dual_scores, 0.0)
    primal_objective_t = torch.dot(costs, primal)
    dual_objective_t = torch.dot(masses, dual)
    absolute_primal_t = torch.linalg.vector_norm(marginal_residual)
    absolute_dual_t = torch.linalg.vector_norm(dual_residual)
    relative_primal_t = absolute_primal_t / (1.0 + torch.linalg.vector_norm(masses))
    relative_dual_t = absolute_dual_t / (1.0 + torch.linalg.vector_norm(costs))
    relative_gap_t = torch.abs(primal_objective_t - dual_objective_t) / (
        1.0 + torch.abs(primal_objective_t) + torch.abs(dual_objective_t)
    )
    return _Residuals(
        primal_objective=float(primal_objective_t.item()),
        dual_objective=float(dual_objective_t.item()),
        absolute_primal=float(absolute_primal_t.item()),
        relative_primal=float(relative_primal_t.item()),
        absolute_dual=float(absolute_dual_t.item()),
        relative_dual=float(relative_dual_t.item()),
        relative_gap=float(relative_gap_t.item()),
    )


class TorchRestrictedOTPDLP:
    """
    CN: 面向 HELLO active support 的 FP64 Torch Halpern-PDHG restricted OT 求解器。
    EN: FP64 Torch Halpern-PDHG restricted OT solver specialized for HELLO active supports.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        evaluation_frequency: int = 200,
        time_limit_seconds: float = 3600.0,
        iteration_limit: int = np.iinfo(np.int32).max,
        record_state_trace: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.evaluation_frequency = int(evaluation_frequency)
        self.time_limit_seconds = float(time_limit_seconds)
        self.iteration_limit = int(iteration_limit)
        self.record_state_trace = bool(record_state_trace)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TorchRestrictedOTPDLP requested CUDA, but torch.cuda.is_available() is false.")
        if self.evaluation_frequency < 1:
            raise ValueError("evaluation_frequency must be >= 1")
        if self.time_limit_seconds <= 0.0 or self.iteration_limit < 1:
            raise ValueError("time_limit_seconds and iteration_limit must be positive")

    def solve_restricted(
        self,
        *,
        rows: Any,
        cols: Any,
        costs: Any,
        source_mass: Any,
        target_mass: Any,
        warm_start_primal: Optional[Any] = None,
        warm_start_dual: Optional[Any] = None,
        tolerance: float | Dict[str, float] = 1e-6,
        verbose: bool = False,
    ) -> SolverResult:
        """
        CN: 直接从 active-support 端点求解平衡 restricted OT，不物化 CSR/CSC 矩阵。
        EN: Solve balanced restricted OT directly from active-support endpoints without materializing CSR/CSC matrices.
        """
        if isinstance(tolerance, dict):
            objective_tolerance = float(tolerance.get("objective", 1e-6))
            primal_tolerance = float(tolerance.get("primal", objective_tolerance))
            dual_tolerance = float(tolerance.get("dual", objective_tolerance))
        else:
            objective_tolerance = primal_tolerance = dual_tolerance = float(tolerance)
        first = self._solve_once(
            rows=rows,
            cols=cols,
            costs=costs,
            source_mass=source_mass,
            target_mass=target_mass,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            objective_tolerance=objective_tolerance,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
            scalar_rescaling=False,
            verbose=bool(verbose),
        )
        if first.termination_reason != "NUMERICAL_DIVERGENCE":
            return first
        retry = self._solve_once(
            rows=rows,
            cols=cols,
            costs=costs,
            source_mass=source_mass,
            target_mass=target_mass,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            objective_tolerance=objective_tolerance,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
            scalar_rescaling=True,
            verbose=bool(verbose),
        )
        retry.solver_diag = dict(retry.solver_diag or {})
        retry.solver_diag.update(
            {
                "numerical_rescaling_retry": True,
                "numerical_rescaling_first_termination_reason": first.termination_reason,
                "numerical_rescaling_first_iterations": int(first.iterations),
            }
        )
        return retry

    def _solve_once(
        self,
        *,
        rows: Any,
        cols: Any,
        costs: Any,
        source_mass: Any,
        target_mass: Any,
        warm_start_primal: Optional[Any],
        warm_start_dual: Optional[Any],
        objective_tolerance: float,
        primal_tolerance: float,
        dual_tolerance: float,
        scalar_rescaling: bool,
        verbose: bool,
    ) -> SolverResult:
        started = time.perf_counter()
        dtype = torch.float64
        rows_t = _as_1d_tensor(rows, dtype=torch.int64, device=self.device)
        cols_t = _as_1d_tensor(cols, dtype=torch.int64, device=self.device)
        costs_t = _as_1d_tensor(costs, dtype=dtype, device=self.device)
        source_mass_t = _as_1d_tensor(source_mass, dtype=dtype, device=self.device)
        target_mass_t = _as_1d_tensor(target_mass, dtype=dtype, device=self.device)
        masses_t = torch.cat((source_mass_t, target_mass_t))
        n_source = int(source_mass_t.numel())
        n_target = int(target_mass_t.numel())
        n_variables = int(costs_t.numel())
        if rows_t.numel() != n_variables or cols_t.numel() != n_variables:
            raise ValueError("rows, cols, and costs must have equal lengths")
        if n_variables == 0:
            raise ValueError("restricted OT active support must not be empty")
        if torch.any(rows_t < 0) or torch.any(rows_t >= n_source):
            raise ValueError("source endpoint index is out of bounds")
        if torch.any(cols_t < 0) or torch.any(cols_t >= n_target):
            raise ValueError("target endpoint index is out of bounds")
        mass_scale = max(1.0, float(torch.max(torch.abs(masses_t)).item()))
        if abs(float(source_mass_t.sum().item()) - float(target_mass_t.sum().item())) > 1e-12 * mass_scale:
            raise ValueError("balanced restricted OT requires equal source and target total mass")

        source_degree = torch.bincount(rows_t, minlength=n_source).to(dtype=dtype)
        target_degree = torch.bincount(cols_t, minlength=n_target).to(dtype=dtype)
        constraint_scale = torch.sqrt(torch.cat((source_degree, target_degree)).clamp_min(1.0))
        variable_scale = torch.full((n_variables,), math.sqrt(2.0), dtype=dtype, device=self.device)
        bound_scalar = 1.0
        objective_scalar = 1.0
        if scalar_rescaling:
            bound_scalar = 1.0 / max(1.0, float(torch.linalg.vector_norm(masses_t).item()))
            objective_scalar = 1.0 / max(1.0, float(torch.linalg.vector_norm(costs_t).item()))

        scaled_costs = costs_t * objective_scalar / variable_scale
        scaled_masses = masses_t * bound_scalar / constraint_scale
        edge_scale = variable_scale * constraint_scale[:n_source][rows_t]
        target_edge_scale = variable_scale * constraint_scale[n_source:][cols_t]

        def apply_a(value: torch.Tensor) -> torch.Tensor:
            source = torch.zeros(n_source, dtype=dtype, device=self.device)
            target = torch.zeros(n_target, dtype=dtype, device=self.device)
            source.index_add_(0, rows_t, value / edge_scale)
            target.index_add_(0, cols_t, value / target_edge_scale)
            return torch.cat((source, target))

        def apply_at(value: torch.Tensor) -> torch.Tensor:
            return value[:n_source][rows_t] / edge_scale + value[n_source:][cols_t] / target_edge_scale

        if warm_start_primal is None:
            initial_primal = torch.zeros(n_variables, dtype=dtype, device=self.device)
        else:
            initial_primal = torch.clamp_min(
                _as_1d_tensor(warm_start_primal, dtype=dtype, device=self.device), 0.0
            ) * variable_scale * bound_scalar
        if warm_start_dual is None:
            initial_dual = torch.zeros(n_source + n_target, dtype=dtype, device=self.device)
        else:
            initial_dual = (
                _as_1d_tensor(warm_start_dual, dtype=dtype, device=self.device)
                * constraint_scale
                * objective_scalar
            )
        if initial_primal.numel() != n_variables or initial_dual.numel() != n_source + n_target:
            raise ValueError("warm-start dimensions do not match the restricted OT problem")

        current_primal = initial_primal.clone()
        current_dual = initial_dual.clone()
        pdhg_primal = current_primal.clone()
        pdhg_dual = current_dual.clone()
        step_size = 0.998
        if scalar_rescaling:
            primal_weight = 1.0
        else:
            primal_weight = float(torch.linalg.vector_norm(scaled_costs).item()) / max(
                float(torch.linalg.vector_norm(scaled_masses).item()), 1e-16
            )
            primal_weight = max(primal_weight, 1e-16)
        best_primal_weight = primal_weight
        error_sum = 0.0
        last_error = 0.0
        best_residual_balance = math.inf
        initial_fixed_point_error = math.inf
        last_trial_fixed_point_error = math.inf
        fixed_point_error = math.inf
        inner_count = 0
        total_count = 0
        restart_count = 0
        is_major_iteration = False
        trace: list[Dict[str, float]] = []
        termination_reason = "UNSPECIFIED"
        residual = _residuals(
            primal=pdhg_primal / (variable_scale * bound_scalar),
            dual=pdhg_dual / (constraint_scale * objective_scalar),
            rows=rows_t,
            cols=cols_t,
            costs=costs_t,
            masses=masses_t,
            n_source=n_source,
            n_target=n_target,
        )

        while termination_reason == "UNSPECIFIED":
            should_evaluate = is_major_iteration or total_count == 0
            if should_evaluate:
                primal_unscaled = pdhg_primal / (variable_scale * bound_scalar)
                dual_unscaled = pdhg_dual / (constraint_scale * objective_scalar)
                residual = _residuals(
                    primal=primal_unscaled,
                    dual=dual_unscaled,
                    rows=rows_t,
                    cols=cols_t,
                    costs=costs_t,
                    masses=masses_t,
                    n_source=n_source,
                    n_target=n_target,
                )
                if self.record_state_trace:
                    trace.append(
                        {
                            "iteration": int(total_count),
                            "primal_objective": residual.primal_objective,
                            "dual_objective": residual.dual_objective,
                            "relative_primal_residual": residual.relative_primal,
                            "relative_dual_residual": residual.relative_dual,
                            "relative_gap": residual.relative_gap,
                            "primal_weight": float(primal_weight),
                        }
                    )
                finite = all(math.isfinite(value) for value in residual.__dict__.values())
                if not finite or not math.isfinite(primal_weight):
                    termination_reason = "NUMERICAL_DIVERGENCE"
                elif (
                    residual.relative_primal < primal_tolerance
                    and residual.relative_dual < dual_tolerance
                    and residual.relative_gap < objective_tolerance
                ):
                    termination_reason = "OPTIMAL"
                elif total_count >= self.iteration_limit:
                    termination_reason = "ITERATION_LIMIT"
                elif time.perf_counter() - started >= self.time_limit_seconds:
                    termination_reason = "TIME_LIMIT"
                if termination_reason != "UNSPECIFIED":
                    break

            do_restart = False
            if is_major_iteration or total_count == 0:
                if total_count == self.evaluation_frequency:
                    do_restart = True
                elif total_count > self.evaluation_frequency:
                    do_restart = (
                        inner_count >= 0.36 * total_count
                        or fixed_point_error <= 0.2 * initial_fixed_point_error
                        or (
                            fixed_point_error <= 0.5 * initial_fixed_point_error
                            and fixed_point_error > last_trial_fixed_point_error
                        )
                    )
                last_trial_fixed_point_error = fixed_point_error
            if do_restart:
                delta_primal = pdhg_primal - initial_primal
                delta_dual = pdhg_dual - initial_dual
                primal_distance = float(torch.linalg.vector_norm(delta_primal).item())
                dual_distance = float(torch.linalg.vector_norm(delta_dual).item())
                ratio = residual.relative_dual / max(residual.relative_primal, 1e-300)
                if (
                    1e-16 < primal_distance < 1e12
                    and 1e-16 < dual_distance < 1e12
                    and 1e-8 < ratio < 1e8
                ):
                    error = math.log(dual_distance) - math.log(primal_distance) - math.log(primal_weight)
                    error_sum = 0.3 * error_sum + error
                    primal_weight *= math.exp(0.99 * error + 0.01 * error_sum)
                    last_error = error
                else:
                    primal_weight = best_primal_weight
                    error_sum = 0.0
                    last_error = 0.0
                if residual.relative_primal > 0.0 and residual.relative_dual > 0.0:
                    balance = abs(math.log10(residual.relative_dual / residual.relative_primal))
                    if balance < best_residual_balance:
                        best_residual_balance = balance
                        best_primal_weight = primal_weight
                initial_primal = pdhg_primal.clone()
                current_primal = pdhg_primal.clone()
                initial_dual = pdhg_dual.clone()
                current_dual = pdhg_dual.clone()
                inner_count = 0
                last_trial_fixed_point_error = math.inf
                restart_count += 1

            is_major_iteration = ((total_count + 1) % self.evaluation_frequency) == 0
            primal_step = step_size / primal_weight
            dual_product = apply_at(current_dual)
            pdhg_primal = torch.clamp_min(current_primal - primal_step * (scaled_costs - dual_product), 0.0)
            reflected_primal = 2.0 * pdhg_primal - current_primal
            dual_step = step_size * primal_weight
            pdhg_dual = current_dual - dual_step * (apply_a(reflected_primal) - scaled_masses)
            reflected_dual = 2.0 * pdhg_dual - current_dual

            if is_major_iteration or do_restart:
                delta_primal = reflected_primal - current_primal
                delta_dual = reflected_dual - current_dual
                movement = (
                    torch.dot(delta_primal, delta_primal) * primal_weight
                    + torch.dot(delta_dual, delta_dual) / primal_weight
                )
                interaction = 2.0 * step_size * torch.dot(apply_at(delta_dual), delta_primal)
                fixed_point_error = math.sqrt(max(0.0, float((movement + interaction).item())))
                if do_restart:
                    initial_fixed_point_error = fixed_point_error

            halpern_weight = float(inner_count + 1) / float(inner_count + 2)
            current_primal = halpern_weight * reflected_primal + (1.0 - halpern_weight) * initial_primal
            current_dual = halpern_weight * reflected_dual + (1.0 - halpern_weight) * initial_dual
            inner_count += 1
            total_count += 1

        primal_out_t = pdhg_primal / (variable_scale * bound_scalar)
        dual_out_t = pdhg_dual / (constraint_scale * objective_scalar)
        duration = float(time.perf_counter() - started)
        solver_diag: Dict[str, Any] = {
            "backend": "torch",
            "device": str(self.device),
            "dtype": "float64",
            "scaling_mode": "degree_sqrt" + ("_with_scalar_retry" if scalar_rescaling else ""),
            "step_size": float(step_size),
            "termination_norm": "l2",
            "termination_abs_primal_res": residual.absolute_primal,
            "termination_rel_primal_res": residual.relative_primal,
            "termination_abs_dual_res": residual.absolute_dual,
            "termination_rel_dual_res": residual.relative_dual,
            "termination_objective_vector_norm": float(torch.linalg.vector_norm(costs_t).item()),
            "termination_constraint_bound_norm": float(torch.linalg.vector_norm(masses_t).item()),
            "algorithm_abs_primal_res_l2": residual.absolute_primal,
            "algorithm_rel_primal_res_l2": residual.relative_primal,
            "algorithm_abs_dual_res_l2": residual.absolute_dual,
            "algorithm_rel_dual_res_l2": residual.relative_dual,
            "relative_primal_dual_gap": residual.relative_gap,
            "primal_weight": float(primal_weight),
            "restart_count": int(restart_count),
            "runtime_sec": duration,
            "termination_reason": termination_reason,
            "numerical_rescaling_retry": bool(scalar_rescaling),
        }
        if self.record_state_trace:
            solver_diag["state_trace"] = trace
        if verbose:
            print(
                f"[TorchRestrictedOTPDLP] termination={termination_reason} "
                f"iterations={total_count} rel_primal={residual.relative_primal:.3e} "
                f"rel_dual={residual.relative_dual:.3e} gap={residual.relative_gap:.3e}"
            )
        return SolverResult(
            success=termination_reason == "OPTIMAL",
            x=primal_out_t.detach(),
            y=dual_out_t.detach(),
            obj_val=residual.primal_objective,
            dual_obj_val=residual.dual_objective,
            duration=duration,
            iterations=int(total_count),
            peak_mem=0.0,
            termination_reason=termination_reason,
            primal_feas=residual.relative_primal,
            dual_feas=residual.relative_dual,
            gap=residual.relative_gap,
            solver_diag=solver_diag,
        )


__all__ = [
    "TorchRestrictedOTPDLP",
    "_gather_transpose",
    "_residuals",
    "_scatter_marginals",
]
