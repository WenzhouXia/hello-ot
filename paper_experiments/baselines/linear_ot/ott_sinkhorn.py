from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .problem import LinearOTProblem
from .result import LinearOTResult, TransportEvaluation


METHOD_NAME = "ott_jax_sinkhorn_l1_negdot_std"
ONLINE_EVALUATION_MEMORY_BUDGET_MIB = 2048
ONLINE_EVALUATION_MEMORY_SAFETY_FACTOR = 2.0
ROUNDING_VALIDATION_ATOL = 1.0e-10


@dataclass
class OttSinkhornDualSolution:
    """
    CN: 保存共享 OTT-JAX 求解器的紧凑 dual 输出，不实例化 transport matrix。
    EN: Store compact dual output from the shared OTT-JAX solver without materializing a transport matrix.
    """

    problem: LinearOTProblem
    source_potential: Any
    target_potential: Any
    errors: Any
    converged: Any
    n_iters: Any
    regularization: float
    cost_std: float
    runtime_sec: float
    batch_size: int | None
    dtype_name: str
    jax_version: str
    ott_version: str


@dataclass(frozen=True)
class ImplicitSinkhornTransport:
    """
    CN: 保存 online Sinkhorn 的紧凑表示及已分块计算的 scalar evaluation。
    EN: Store a compact online Sinkhorn representation and its blockwise scalar evaluation.
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


def solve_ott_jax_sinkhorn_l1_negdot_std(
    problem: LinearOTProblem,
    *,
    epsilon: float,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-3,
    dtype_name: str = "float64",
    batch_size: int | None = None,
) -> LinearOTResult:
    """
    CN: 运行共享 OTT-JAX Sinkhorn；batch_size 非空时返回 implicit transport。
    EN: Run shared OTT-JAX Sinkhorn, returning an implicit transport when batch_size is set.
    """
    _require_fp64_rounding(dtype_name)
    solution = solve_ott_jax_sinkhorn_l1_negdot_std_dual(
        problem,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
        dtype_name=dtype_name,
        batch_size=batch_size,
    )
    if batch_size is None:
        plan_t0 = time.perf_counter()
        plan = materialize_ott_sinkhorn_transport(solution)
        transport_recovery_time_sec = float(time.perf_counter() - plan_t0)
        evaluation_t0 = time.perf_counter()
        row = np.sum(plan, axis=1, dtype=np.float64)
        col = np.sum(plan, axis=0, dtype=np.float64)
        objective = _squared_l2_objective_from_dense(problem, plan, row=row, col=col)
        evaluation_time_sec = float(time.perf_counter() - evaluation_t0)
        runtime_sec = float(solution.runtime_sec)
        transport_kind = "dense"
        transport: Any = plan
        rounding_time_sec = 0.0
        rounded_objective = None
        evaluation_details: dict[str, Any] = {"evaluation_mode": "dense_materialized"}
    else:
        evaluation, rounded_objective, evaluation_details, evaluation_time_sec, rounding_time_sec = (
            evaluate_ott_sinkhorn_duals(solution, block_size=int(batch_size))
        )
        objective = float(evaluation.objective)
        runtime_sec = float(solution.runtime_sec)
        transport_recovery_time_sec = 0.0
        import jax

        transport_kind = "implicit"
        transport = ImplicitSinkhornTransport(
            source_potential=np.asarray(jax.device_get(solution.source_potential), dtype=np.float64),
            target_potential=np.asarray(jax.device_get(solution.target_potential), dtype=np.float64),
            regularization=float(solution.regularization),
            batch_size=int(batch_size),
            evaluation=evaluation,
            rounded_objective=float(rounded_objective),
        )

    import jax

    errors = np.asarray(jax.device_get(solution.errors), dtype=np.float64).reshape(-1)
    valid_errors = errors[np.isfinite(errors) & (errors >= 0.0)]
    error_last = float("nan") if valid_errors.size == 0 else float(valid_errors[-1])
    converged = bool(jax.device_get(solution.converged))
    diagnostics = {
        "jax_version": solution.jax_version,
        "ott_version": solution.ott_version,
        "epsilon": float(epsilon),
        "regularization": float(solution.regularization),
        "cost_std": float(solution.cost_std),
        "cost_std_dtype": "float64",
        "max_iterations": int(max_iterations),
        "completed_iterations": int(jax.device_get(solution.n_iters)),
        "tolerance": float(tolerance),
        "error_last": error_last,
        "dtype": str(dtype_name),
        "batch_size": None if batch_size is None else int(batch_size),
        "solver_backend": "ott.solvers.linear.Sinkhorn",
        "geometry_mode": "dense_negative_dot" if batch_size is None else "online_pointcloud",
        "runtime_scope": "solver_only_through_synchronized_dual_output",
        "solver_runtime_sec": float(solution.runtime_sec),
        "transport_recovery_time_sec": float(transport_recovery_time_sec),
        "evaluation_time_sec": float(evaluation_time_sec),
        "rounding_time_sec": float(rounding_time_sec),
        "postprocess_time_sec": float(
            transport_recovery_time_sec + evaluation_time_sec + rounding_time_sec
        ),
        "evaluation_mode": str(evaluation_details["evaluation_mode"]),
    }
    for key in (
        "rounding_residual_mass",
        "rounding_source_residual_mass",
        "rounding_target_residual_mass",
        "rounding_residual_mass_difference",
        "rounded_row_marginal_l1_error",
        "rounded_col_marginal_l1_error",
        "rounding_effective_dtype",
        "rounding_correction_objective",
        "evaluation_point_batch_size",
        "evaluation_rank",
        "evaluation_rank_chunk_size",
        "evaluation_rank_chunk_count",
        "evaluation_memory_budget_mib",
        "evaluation_estimated_peak_working_mib",
        "rounded_evaluation_rank_chunk_count",
    ):
        if key in evaluation_details:
            diagnostics[key] = evaluation_details[key]
    if rounded_objective is not None:
        diagnostics["rounded_objective"] = float(rounded_objective)
    return LinearOTResult(
        method=METHOD_NAME,
        solver_objective=float(objective),
        runtime_sec=runtime_sec,
        transport_kind=transport_kind,
        transport=transport,
        converged=converged,
        status="converged" if converged else "iteration_limit",
        diagnostics=diagnostics,
    )


def solve_ott_jax_sinkhorn_l1_negdot_std_dual(
    problem: LinearOTProblem,
    *,
    epsilon: float,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-3,
    dtype_name: str = "float32",
    batch_size: int | None = None,
) -> OttSinkhornDualSolution:
    """
    CN: 用统一 float64 population STD 运行 dense 或 online Sinkhorn，仅返回 dual。
    EN: Run dense or online Sinkhorn with unified float64 population STD and return only duals.
    """
    _validate_options(problem, epsilon, max_iterations, tolerance, dtype_name, batch_size)
    try:
        import jax
        import jax.numpy as jnp
        import ott
    except Exception as exc:
        raise ImportError(f"JAX and OTT-JAX are required for {METHOD_NAME}.") from exc
    if str(dtype_name) == "float64" and not bool(jax.config.jax_enable_x64):
        raise RuntimeError("dtype_name='float64' requires jax_enable_x64 before importing this adapter.")

    solve_t0 = time.perf_counter()
    cost_std = negative_dot_cost_std_float64(problem.source_points, problem.target_points)
    regularization = float(epsilon) * cost_std
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("Sinkhorn regularization must be finite and positive.")
    dtype = jnp.float32 if str(dtype_name) == "float32" else jnp.float64
    solve_fn = _compiled_dual_solver(
        int(max_iterations),
        float(tolerance),
        str(dtype_name),
        None if batch_size is None else int(batch_size),
    )
    source = jnp.asarray(problem.source_points, dtype=dtype)
    target = jnp.asarray(problem.target_points, dtype=dtype)
    source_mass = jnp.asarray(problem.source_mass, dtype=dtype)
    target_mass = jnp.asarray(problem.target_mass, dtype=dtype)
    f, g, errors, converged, n_iters = solve_fn(
        source,
        target,
        source_mass,
        target_mass,
        jnp.asarray(regularization, dtype=dtype),
    )
    f.block_until_ready()
    runtime_sec = float(time.perf_counter() - solve_t0)
    return OttSinkhornDualSolution(
        problem=problem,
        source_potential=f,
        target_potential=g,
        errors=errors,
        converged=converged,
        n_iters=n_iters,
        regularization=regularization,
        cost_std=cost_std,
        runtime_sec=runtime_sec,
        batch_size=None if batch_size is None else int(batch_size),
        dtype_name=str(dtype_name),
        jax_version=str(getattr(jax, "__version__", "")),
        ott_version=str(getattr(ott, "__version__", "")),
    )


def negative_dot_cost_std_float64(source: np.ndarray, target: np.ndarray) -> float:
    """
    CN: 用分块二阶矩计算所有 negative-dot pair costs 的 float64 population STD。
    EN: Compute the float64 population STD of all negative-dot pair costs from blockwise second moments.
    """
    x = np.asarray(source)
    y = np.asarray(target)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("source and target must be nonempty compatible point matrices.")
    mean_x, second_x = _float64_first_second_moments(x)
    mean_y, second_y = _float64_first_second_moments(y)
    mean_dot = float(np.dot(mean_x, mean_y))
    second_moment = float(np.einsum("ij,ij->", second_x, second_y, optimize=True))
    variance = max(second_moment - mean_dot * mean_dot, 0.0)
    value = float(math.sqrt(variance))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("negative-dot cost population STD must be finite and positive.")
    return value


def _float64_first_second_moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 分块累计一阶与二阶矩，避免创建完整 float64 点云副本。
    EN: Accumulate first and second moments in blocks without a full float64 point-cloud copy.
    """
    n_rows, dimension = points.shape
    block_rows = max(1, min(int(n_rows), 4_194_304 // int(dimension)))
    first = np.zeros((dimension,), dtype=np.float64)
    second = np.zeros((dimension, dimension), dtype=np.float64)
    for start in range(0, int(n_rows), block_rows):
        block = np.asarray(points[start : start + block_rows], dtype=np.float64, order="C")
        first += np.sum(block, axis=0, dtype=np.float64)
        second += block.T @ block
    return first / float(n_rows), second / float(n_rows)


def materialize_ott_sinkhorn_transport(solution: OttSinkhornDualSolution) -> np.ndarray:
    """
    CN: 仅为 legacy dense 模式物化完整 coupling。
    EN: Materialize the full coupling only for legacy dense mode.
    """
    if solution.batch_size is not None:
        raise ValueError("Online Sinkhorn transport must be evaluated blockwise, not materialized.")
    import jax
    import jax.numpy as jnp

    dtype = jnp.float32 if solution.dtype_name == "float32" else jnp.float64
    source = jnp.asarray(solution.problem.source_points, dtype=dtype)
    target = jnp.asarray(solution.problem.target_points, dtype=dtype)
    cost = -(source @ target.T)
    plan = jnp.exp(
        (solution.source_potential[:, None] + solution.target_potential[None, :] - cost)
        / jnp.asarray(solution.regularization, dtype=dtype)
    )
    plan.block_until_ready()
    return np.asarray(jax.device_get(plan))


def evaluate_ott_sinkhorn_duals(
    solution: OttSinkhornDualSolution,
    *,
    block_size: int,
    support_threshold: float = 0.0,
) -> tuple[TransportEvaluation, float, dict[str, Any], float, float]:
    """
    CN: 从 dual 分块计算 raw metrics 和隐式 rounded objective，不物化 O(mn) coupling。
    EN: Compute raw metrics and the implicit rounded objective from duals in blocks without an O(mn) coupling.
    """
    _require_fp64_rounding(solution.dtype_name)
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    if solution.batch_size is not None:
        # CN: 必须复用 solver 的 PointCloud batch，确保求解与评测使用相同的 cost kernel。
        # EN: Reuse the solver PointCloud batch so solving and evaluation share the same cost kernel.
        online_block_size = int(solution.batch_size)
        return _evaluate_online_ott_sinkhorn_duals_stable(
            solution,
            block_size=online_block_size,
            support_threshold=float(support_threshold),
        )
    import jax
    import jax.numpy as jnp

    problem = solution.problem
    dtype = jnp.float32 if solution.dtype_name == "float32" else jnp.float64
    source = jnp.asarray(problem.source_points, dtype=dtype)
    target = jnp.asarray(problem.target_points, dtype=dtype)
    source_mass = jnp.asarray(problem.source_mass, dtype=dtype)
    target_mass = jnp.asarray(problem.target_mass, dtype=dtype)
    target_norm = jnp.sum(target * target, axis=1)
    regularization = jnp.asarray(solution.regularization, dtype=dtype)
    threshold = jnp.asarray(float(support_threshold), dtype=dtype)
    raw_block = _compiled_raw_transport_block(solution.batch_size is not None)
    rounded_block = _compiled_rounded_transport_block(solution.batch_size is not None)

    raw_t0 = time.perf_counter()
    col_mass = jnp.zeros((problem.shape[1],), dtype=dtype)
    col_after_row_scale = jnp.zeros((problem.shape[1],), dtype=dtype)
    objective = jnp.asarray(0.0, dtype=dtype)
    internal_objective = jnp.asarray(0.0, dtype=dtype)
    matrix_sq_sum = jnp.asarray(0.0, dtype=dtype)
    nnz = 0
    support_size = 0
    row_mass_parts: list[Any] = []
    row_scale_parts: list[Any] = []
    row_argmax_parts: list[Any] = []
    for start in range(0, problem.shape[0], int(block_size)):
        stop = min(start + int(block_size), problem.shape[0])
        metrics = raw_block(
            source[start:stop],
            target,
            solution.source_potential[start:stop],
            solution.target_potential,
            source_mass[start:stop],
            target_norm,
            regularization,
            threshold,
        )
        (
            row_mass,
            row_scale,
            col_part,
            col_scaled_part,
            obj_part,
            internal_part,
            sq_part,
            nnz_part,
            support_part,
            argmax,
        ) = metrics
        row_mass_parts.append(row_mass)
        row_scale_parts.append(row_scale)
        row_argmax_parts.append(argmax)
        col_mass = col_mass + col_part
        col_after_row_scale = col_after_row_scale + col_scaled_part
        objective = objective + obj_part
        internal_objective = internal_objective + internal_part
        matrix_sq_sum = matrix_sq_sum + sq_part
        block_nnz, block_support_size = jax.device_get((nnz_part, support_part))
        nnz += int(block_nnz)
        support_size += int(block_support_size)
    row_mass = jnp.concatenate(row_mass_parts)
    row_scale = jnp.concatenate(row_scale_parts)
    row_argmax = jnp.concatenate(row_argmax_parts)
    objective.block_until_ready()
    raw_time_sec = float(time.perf_counter() - raw_t0)

    row_residual = row_mass - source_mass
    col_residual = col_mass - target_mass
    row_l2 = jnp.linalg.norm(row_residual)
    col_l2 = jnp.linalg.norm(col_residual)
    primal_l2 = jnp.sqrt(row_l2 * row_l2 + col_l2 * col_l2)
    marginal_norm = jnp.sqrt(jnp.sum(source_mass * source_mass) + jnp.sum(target_mass * target_mass))
    primal_feasibility = primal_l2 / (1.0 + marginal_norm)
    transport_mass_error = jnp.abs(jnp.sum(col_mass) - jnp.sum(source_mass))

    rounding_t0 = time.perf_counter()
    col_scale = jnp.minimum(
        jnp.where(col_after_row_scale > 0.0, target_mass / col_after_row_scale, 1.0),
        1.0,
    )
    rounded_row_parts: list[Any] = []
    rounded_base_objective = jnp.asarray(0.0, dtype=dtype)
    for start in range(0, problem.shape[0], int(block_size)):
        stop = min(start + int(block_size), problem.shape[0])
        rounded_row, rounded_obj = rounded_block(
            source[start:stop],
            target,
            solution.source_potential[start:stop],
            solution.target_potential,
            row_scale[start:stop],
            col_scale,
            target_norm,
            regularization,
        )
        rounded_row_parts.append(rounded_row)
        rounded_base_objective = rounded_base_objective + rounded_obj
    rounded_row = jnp.concatenate(rounded_row_parts)
    rounded_col = col_scale * col_after_row_scale
    source_residual = jnp.maximum(source_mass - rounded_row, 0.0)
    target_residual = jnp.maximum(target_mass - rounded_col, 0.0)
    source_residual_mass = jnp.sum(source_residual)
    target_residual_mass = jnp.sum(target_residual)
    source_norm = jnp.sum(source * source, axis=1)
    source_first = source_residual @ source
    target_first = target_residual @ target
    correction_numerator = (
        jnp.sum(source_residual * source_norm) * jnp.sum(target_residual)
        + jnp.sum(source_residual) * jnp.sum(target_residual * target_norm)
        - 2.0 * jnp.sum(source_first * target_first)
    )
    correction = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        correction_numerator / jnp.maximum(source_residual_mass, jnp.finfo(dtype).eps),
        0.0,
    )
    rounded_objective = rounded_base_objective + correction
    correction_row = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        source_residual * target_residual_mass / jnp.maximum(source_residual_mass, jnp.finfo(dtype).eps),
        jnp.zeros_like(source_residual),
    )
    correction_col = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        target_residual,
        jnp.zeros_like(target_residual),
    )
    rounded_row_l1 = jnp.sum(jnp.abs(rounded_row + correction_row - source_mass))
    rounded_col_l1 = jnp.sum(jnp.abs(rounded_col + correction_col - target_mass))
    rounded_objective.block_until_ready()
    rounding_time_sec = float(time.perf_counter() - rounding_t0)

    values = jax.device_get(
        (
            objective,
            primal_feasibility,
            primal_l2,
            row_l2,
            col_l2,
            transport_mass_error,
            internal_objective,
            matrix_sq_sum,
            row_argmax,
            rounded_objective,
            source_residual_mass,
            target_residual_mass,
            rounded_row_l1,
            rounded_col_l1,
        )
    )
    rounding_diagnostics = _validate_rounded_marginals(
        source_residual_mass=float(values[10]),
        target_residual_mass=float(values[11]),
        row_l1_error=float(values[12]),
        col_l1_error=float(values[13]),
    )
    evaluation = TransportEvaluation(
        objective=float(values[0]),
        primal_feasibility=float(values[1]),
        primal_l2_abs_error=float(values[2]),
        row_marginal_l2_error=float(values[3]),
        col_marginal_l2_error=float(values[4]),
        transport_mass_error=float(values[5]),
        transport_kind="implicit",
        transport_shape=problem.shape,
        transport_nnz=nnz,
    )
    details = {
        "algorithm_support_size": support_size,
        "negative_dot_objective": float(values[6]),
        "matrix_sq_sum": float(values[7]),
        "row_argmax_cols": np.asarray(values[8], dtype=np.int32),
        "evaluation_mode": "dense_dual_blockwise",
        **rounding_diagnostics,
    }
    return evaluation, float(values[9]), details, raw_time_sec, rounding_time_sec


def _evaluate_online_ott_sinkhorn_duals_stable(
    solution: OttSinkhornDualSolution,
    *,
    block_size: int,
    support_threshold: float,
) -> tuple[TransportEvaluation, float, dict[str, Any], float, float]:
    """
    CN: 用 OTT 官方 log-domain transport apply 双重分块评测 online coupling。
    EN: Evaluate an online coupling with doubly blocked official OTT log-domain transport applications.
    """
    import jax
    import jax.numpy as jnp
    from ott.geometry import costs, pointcloud

    problem = solution.problem
    dtype = jnp.float32 if solution.dtype_name == "float32" else jnp.float64
    source = jnp.asarray(problem.source_points, dtype=dtype)
    target = jnp.asarray(problem.target_points, dtype=dtype)
    source_mass = jnp.asarray(problem.source_mass, dtype=dtype)
    target_mass = jnp.asarray(problem.target_mass, dtype=dtype)
    regularization = jnp.asarray(solution.regularization, dtype=dtype)
    scale = jnp.asarray(math.sqrt(0.5), dtype=dtype)
    scaled_source = source * scale
    scaled_target = target * scale
    evaluation_point_batch_size = int(block_size)
    geom = pointcloud.PointCloud(
        scaled_source,
        scaled_target,
        cost_fn=costs.SqEuclidean(),
        batch_size=evaluation_point_batch_size,
        epsilon=regularization,
    )
    source_f = jnp.asarray(solution.source_potential, dtype=dtype)
    target_g = jnp.asarray(solution.target_potential, dtype=dtype)

    evaluation_t0 = time.perf_counter()
    row_mass = geom.marginal_from_potentials(source_f, target_g, axis=1)
    col_mass = geom.marginal_from_potentials(source_f, target_g, axis=0)
    low_rank_geom = _squared_euclidean_low_rank_geometry(geom)
    objective, objective_details = _online_sq_l2_objective_from_official_transport_apply(
        geom,
        low_rank_geom,
        source_f,
        target_g,
        memory_budget_mib=ONLINE_EVALUATION_MEMORY_BUDGET_MIB,
    )
    source_norm = jnp.sum(source * source, axis=1)
    target_norm = jnp.sum(target * target, axis=1)
    matrix_sq_sum = _online_matrix_sq_sum(
        geom,
        source_f,
        target_g,
        regularization,
    )
    row_residual = row_mass - source_mass
    col_residual = col_mass - target_mass
    row_l2 = jnp.linalg.norm(row_residual)
    col_l2 = jnp.linalg.norm(col_residual)
    primal_l2 = jnp.sqrt(row_l2 * row_l2 + col_l2 * col_l2)
    marginal_norm = jnp.sqrt(jnp.sum(source_mass * source_mass) + jnp.sum(target_mass * target_mass))
    primal_feasibility = primal_l2 / (1.0 + marginal_norm)
    transport_mass_error = jnp.abs(jnp.sum(col_mass) - jnp.sum(source_mass))
    support_size, row_argmax = _online_support_summary(
        source,
        target,
        source_f,
        target_g,
        regularization,
        block_size=int(block_size),
        support_threshold=float(support_threshold),
    )
    raw_values = jax.device_get(
        (
            primal_feasibility,
            primal_l2,
            row_l2,
            col_l2,
            transport_mass_error,
            jnp.dot(row_mass, source_norm),
            jnp.dot(col_mass, target_norm),
            matrix_sq_sum,
            row_argmax,
        )
    )
    negative_dot_objective = 0.5 * (
        float(objective) - float(raw_values[5]) - float(raw_values[6])
    )
    evaluation_time_sec = float(time.perf_counter() - evaluation_t0)

    rounding_t0 = time.perf_counter()
    row_scale = jnp.minimum(
        jnp.where(row_mass > 0.0, source_mass / row_mass, 1.0),
        1.0,
    )
    scaled_f = source_f + regularization * jnp.log(jnp.maximum(row_scale, jnp.finfo(dtype).tiny))
    col_after_row_scale = geom.marginal_from_potentials(scaled_f, target_g, axis=0)
    col_scale = jnp.minimum(
        jnp.where(col_after_row_scale > 0.0, target_mass / col_after_row_scale, 1.0),
        1.0,
    )
    scaled_g = target_g + regularization * jnp.log(jnp.maximum(col_scale, jnp.finfo(dtype).tiny))
    rounded_row = geom.marginal_from_potentials(scaled_f, scaled_g, axis=1)
    rounded_col = geom.marginal_from_potentials(scaled_f, scaled_g, axis=0)
    rounded_base_objective, rounded_objective_details = (
        _online_sq_l2_objective_from_official_transport_apply(
            geom,
            low_rank_geom,
            scaled_f,
            scaled_g,
            memory_budget_mib=ONLINE_EVALUATION_MEMORY_BUDGET_MIB,
        )
    )
    source_residual = jnp.maximum(source_mass - rounded_row, 0.0)
    target_residual = jnp.maximum(target_mass - rounded_col, 0.0)
    source_residual_mass = jnp.sum(source_residual)
    target_residual_mass = jnp.sum(target_residual)
    source_first = source_residual @ source
    target_first = target_residual @ target
    correction_numerator = (
        jnp.sum(source_residual * source_norm) * jnp.sum(target_residual)
        + jnp.sum(source_residual) * jnp.sum(target_residual * target_norm)
        - 2.0 * jnp.sum(source_first * target_first)
    )
    correction = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        correction_numerator / jnp.maximum(source_residual_mass, jnp.finfo(dtype).eps),
        0.0,
    )
    correction_row = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        source_residual * target_residual_mass / jnp.maximum(source_residual_mass, jnp.finfo(dtype).eps),
        jnp.zeros_like(source_residual),
    )
    correction_col = jnp.where(
        source_residual_mass > jnp.finfo(dtype).eps,
        target_residual,
        jnp.zeros_like(target_residual),
    )
    rounded_row_l1 = jnp.sum(jnp.abs(rounded_row + correction_row - source_mass))
    rounded_col_l1 = jnp.sum(jnp.abs(rounded_col + correction_col - target_mass))
    rounded_values = jax.device_get(
        (
            source_residual_mass,
            target_residual_mass,
            correction,
            rounded_row_l1,
            rounded_col_l1,
        )
    )
    rounding_diagnostics = _validate_rounded_marginals(
        source_residual_mass=float(rounded_values[0]),
        target_residual_mass=float(rounded_values[1]),
        row_l1_error=float(rounded_values[3]),
        col_l1_error=float(rounded_values[4]),
    )
    rounded_objective = float(rounded_base_objective) + float(rounded_values[2])
    rounding_time_sec = float(time.perf_counter() - rounding_t0)
    evaluation = TransportEvaluation(
        objective=float(objective),
        primal_feasibility=float(raw_values[0]),
        primal_l2_abs_error=float(raw_values[1]),
        row_marginal_l2_error=float(raw_values[2]),
        col_marginal_l2_error=float(raw_values[3]),
        transport_mass_error=float(raw_values[4]),
        transport_kind="implicit",
        transport_shape=problem.shape,
        transport_nnz=int(problem.shape[0]) * int(problem.shape[1]),
    )
    details = {
        "algorithm_support_size": int(support_size),
        "negative_dot_objective": float(negative_dot_objective),
        "matrix_sq_sum": float(raw_values[7]),
        "row_argmax_cols": np.asarray(raw_values[8], dtype=np.int32),
        "rounding_correction_objective": float(rounded_values[2]),
        "evaluation_mode": "online_official_transport_apply_rank_chunked",
        "evaluation_point_batch_size": int(evaluation_point_batch_size),
        "evaluation_rank": int(objective_details["rank"]),
        "evaluation_rank_chunk_size": int(objective_details["rank_chunk_size"]),
        "evaluation_rank_chunk_count": int(objective_details["rank_chunk_count"]),
        "evaluation_memory_budget_mib": int(ONLINE_EVALUATION_MEMORY_BUDGET_MIB),
        "evaluation_estimated_peak_working_mib": float(objective_details["estimated_peak_working_mib"]),
        "rounded_evaluation_rank_chunk_count": int(rounded_objective_details["rank_chunk_count"]),
        **rounding_diagnostics,
    }
    return evaluation, float(rounded_objective), details, evaluation_time_sec, rounding_time_sec


def _online_sq_l2_objective_from_official_transport_apply(
    geom: Any,
    low_rank_geom: Any,
    source_f: Any,
    target_g: Any,
    *,
    memory_budget_mib: int,
    rank_chunk_size: int | None = None,
) -> tuple[float, dict[str, float | int]]:
    """
    CN: 沿 low-rank factor 分块调用 OTT 官方 transport apply，并以 float64 累加标量。
    EN: Apply OTT's official transport in low-rank-factor chunks and accumulate scalars in float64.
    """
    import jax
    import jax.numpy as jnp

    cost_1 = low_rank_geom.cost_1
    cost_2 = low_rank_geom.cost_2
    rank = int(cost_1.shape[1])
    if rank <= 0 or int(cost_2.shape[1]) != rank:
        raise ValueError("OTT low-rank geometry must have a shared positive rank.")
    point_batch_size = int(geom.batch_size)
    dtype_bytes = int(np.dtype(str(cost_1.dtype)).itemsize)
    integrated_points = int(geom.shape[0])
    output_points = int(geom.shape[1])
    bytes_per_rank = dtype_bytes * (
        ONLINE_EVALUATION_MEMORY_SAFETY_FACTOR * point_batch_size * integrated_points
        + integrated_points
        + output_points
    )
    budget_bytes = int(memory_budget_mib) * 1024 * 1024
    automatic_chunk_size = max(1, min(rank, int(budget_bytes // max(bytes_per_rank, 1))))
    chunk_size = automatic_chunk_size if rank_chunk_size is None else int(rank_chunk_size)
    if chunk_size <= 0:
        raise ValueError("rank_chunk_size must be positive when provided.")
    chunk_size = min(rank, chunk_size)

    partial_values: list[float] = []
    for start in range(0, rank, chunk_size):
        stop = min(start + chunk_size, rank)
        transported = geom.apply_transport_from_potentials(
            source_f,
            target_g,
            cost_1[:, start:stop].T,
            axis=0,
        )
        partial = jnp.sum(transported * cost_2[:, start:stop].T)
        partial_values.append(float(jax.device_get(partial)))
        del partial, transported
    solver_cost = math.fsum(partial_values)
    estimated_peak_bytes = bytes_per_rank * chunk_size
    return 2.0 * solver_cost, {
        "rank": rank,
        "rank_chunk_size": chunk_size,
        "rank_chunk_count": int(math.ceil(rank / chunk_size)),
        "estimated_peak_working_mib": estimated_peak_bytes / float(1024 * 1024),
    }


def _squared_euclidean_low_rank_geometry(geom: Any) -> Any:
    """
    CN: 强制使用 OTT 自身的 squared-Euclidean 精确因子，即使小问题不满足其收益判据。
    EN: Force OTT's exact squared-Euclidean factors even when a small problem fails its profitability check.
    """
    low_rank_geom = geom.to_LRCGeometry()
    if hasattr(low_rank_geom, "cost_1") and hasattr(low_rank_geom, "cost_2"):
        return low_rank_geom
    converter = getattr(geom, "_sqeucl_to_lr", None)
    if converter is None:
        raise TypeError("OTT PointCloud does not expose squared-Euclidean low-rank conversion.")
    return converter()


def _online_matrix_sq_sum(geom: Any, source_f: Any, target_g: Any, regularization: Any) -> Any:
    """
    CN: 以 epsilon/2 的 log-domain 边缘计算 ||P||_F^2。
    EN: Compute ||P||_F^2 from log-domain marginals at epsilon/2.
    """
    import jax.numpy as jnp

    half_epsilon = 0.5 * regularization
    lse_value = geom.apply_lse_kernel(source_f, target_g, half_epsilon, axis=1)[0]
    squared_row_mass = jnp.exp((lse_value + source_f) / half_epsilon)
    return jnp.sum(squared_row_mass)


def _online_support_summary(
    source: Any,
    target: Any,
    source_f: Any,
    target_g: Any,
    regularization: Any,
    *,
    block_size: int,
    support_threshold: float,
) -> tuple[int, Any]:
    """
    CN: 仅分块计算 log-weight support 与逐行 argmax，不构造完整 coupling。
    EN: Compute log-weight support and row-wise argmax blockwise without a full coupling.
    """
    import jax
    import jax.numpy as jnp

    threshold_log = -jnp.inf
    if float(support_threshold) > 0.0:
        threshold_log = jnp.log(jnp.asarray(float(support_threshold), dtype=source.dtype))
    support_size = 0
    argmax_parts: list[Any] = []
    target_norm = jnp.sum(target * target, axis=1)
    for start in range(0, int(source.shape[0]), int(block_size)):
        stop = min(start + int(block_size), int(source.shape[0]))
        block = source[start:stop]
        sq_cost = jnp.maximum(
            jnp.sum(block * block, axis=1)[:, None]
            + target_norm[None, :]
            - 2.0 * (block @ target.T),
            0.0,
        )
        log_weight = (
            source_f[start:stop, None] + target_g[None, :] - 0.5 * sq_cost
        ) / regularization
        argmax_parts.append(jnp.argmax(log_weight, axis=1).astype(jnp.int32))
        if float(support_threshold) <= 0.0:
            support_size += int(block.shape[0]) * int(target.shape[0])
        else:
            support_size += int(jax.device_get(jnp.count_nonzero(log_weight > threshold_log)))
    return int(support_size), jnp.concatenate(argmax_parts)


def transport_values_at_indices(
    solution: OttSinkhornDualSolution,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """
    CN: 从 dual 仅恢复指定 transport entries。
    EN: Recover only selected transport entries from duals.
    """
    import jax

    f = np.asarray(jax.device_get(solution.source_potential), dtype=np.float64)
    g = np.asarray(jax.device_get(solution.target_potential), dtype=np.float64)
    row_idx = np.asarray(rows, dtype=np.int64)
    col_idx = np.asarray(cols, dtype=np.int64)
    source = np.asarray(solution.problem.source_points[row_idx], dtype=np.float64)
    target = np.asarray(solution.problem.target_points[col_idx], dtype=np.float64)
    if solution.batch_size is None:
        cost = -np.einsum("ij,ij->i", source, target)
    else:
        diff = source - target
        cost = 0.5 * np.einsum("ij,ij->i", diff, diff)
    return np.exp((f[row_idx] + g[col_idx] - cost) / float(solution.regularization))


@functools.lru_cache(maxsize=None)
def _compiled_dual_solver(
    max_iterations: int,
    tolerance: float,
    dtype_name: str,
    batch_size: int | None,
) -> Callable[..., tuple[Any, ...]]:
    import jax
    import jax.numpy as jnp
    from ott.geometry import costs, geometry, pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    solver = sinkhorn.Sinkhorn(
        lse_mode=True,
        threshold=float(tolerance),
        norm_error=1,
        inner_iterations=10,
        max_iterations=int(max_iterations),
    )

    def _solve(source: Any, target: Any, source_mass: Any, target_mass: Any, regularization: Any) -> tuple[Any, ...]:
        if batch_size is None:
            geom = geometry.Geometry(cost_matrix=-(source @ target.T), epsilon=regularization)
        else:
            scale = jnp.asarray(math.sqrt(0.5), dtype=source.dtype)
            geom = pointcloud.PointCloud(
                source * scale,
                target * scale,
                cost_fn=costs.SqEuclidean(),
                batch_size=int(batch_size),
                epsilon=regularization,
            )
        output = solver(linear_problem.LinearProblem(geom, a=source_mass, b=target_mass))
        return output.f, output.g, output.errors, output.converged, output.n_iters

    return jax.jit(_solve)


def _raw_transport_block(
    source: Any,
    target: Any,
    source_f: Any,
    target_g: Any,
    source_mass: Any,
    target_norm: Any,
    regularization: Any,
    support_threshold: Any,
    online_geometry: bool,
) -> tuple[Any, ...]:
    import jax.numpy as jnp

    source_norm = jnp.sum(source * source, axis=1)
    negative_dot = -(source @ target.T)
    sq_cost = jnp.maximum(source_norm[:, None] + target_norm[None, :] + 2.0 * negative_dot, 0.0)
    solver_cost = 0.5 * sq_cost if online_geometry else negative_dot
    plan = jnp.exp((source_f[:, None] + target_g[None, :] - solver_cost) / regularization)
    row_mass = jnp.sum(plan, axis=1)
    row_scale = jnp.minimum(jnp.where(row_mass > 0.0, source_mass / row_mass, 1.0), 1.0)
    return (
        row_mass,
        row_scale,
        jnp.sum(plan, axis=0),
        plan.T @ row_scale,
        jnp.sum(plan * sq_cost),
        jnp.sum(plan * negative_dot),
        jnp.sum(plan * plan),
        jnp.count_nonzero(plan),
        jnp.count_nonzero(plan > support_threshold),
        jnp.argmax(plan, axis=1).astype(jnp.int32),
    )


def _rounded_transport_block(
    source: Any,
    target: Any,
    source_f: Any,
    target_g: Any,
    row_scale: Any,
    col_scale: Any,
    target_norm: Any,
    regularization: Any,
    online_geometry: bool,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    source_norm = jnp.sum(source * source, axis=1)
    sq_cost = jnp.maximum(source_norm[:, None] + target_norm[None, :] - 2.0 * (source @ target.T), 0.0)
    solver_cost = 0.5 * sq_cost if online_geometry else -(source @ target.T)
    plan = jnp.exp((source_f[:, None] + target_g[None, :] - solver_cost) / regularization)
    scaled = row_scale[:, None] * plan * col_scale[None, :]
    return jnp.sum(scaled, axis=1), jnp.sum(scaled * sq_cost)


@functools.lru_cache(maxsize=None)
def _compiled_raw_transport_block(online_geometry: bool) -> Callable[..., tuple[Any, ...]]:
    import jax

    return jax.jit(
        functools.partial(_raw_transport_block, online_geometry=bool(online_geometry))
    )


@functools.lru_cache(maxsize=None)
def _compiled_rounded_transport_block(online_geometry: bool) -> Callable[..., tuple[Any, Any]]:
    import jax

    return jax.jit(
        functools.partial(_rounded_transport_block, online_geometry=bool(online_geometry))
    )


def _validate_options(
    problem: LinearOTProblem,
    epsilon: float,
    max_iterations: int,
    tolerance: float,
    dtype_name: str,
    batch_size: int | None,
) -> None:
    if problem.cost_type != "l2^2":
        raise ValueError(f"{METHOD_NAME} supports only cost_type='l2^2'.")
    if float(epsilon) <= 0.0 or float(tolerance) <= 0.0 or int(max_iterations) <= 0:
        raise ValueError("epsilon, tolerance, and max_iterations must be positive.")
    if str(dtype_name) not in {"float32", "float64"}:
        raise ValueError("dtype_name must be 'float32' or 'float64'.")
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("batch_size must be positive when provided.")


def _require_fp64_rounding(dtype_name: str) -> None:
    """
    CN: 拒绝用非 FP64 的 Sinkhorn dual 执行 primal 恢复与标准 rounding。
    EN: Reject primal recovery and standard rounding from non-FP64 Sinkhorn duals.
    """
    if str(dtype_name) != "float64":
        raise ValueError(
            "Sinkhorn primal recovery and rounding require end-to-end float64; "
            f"received dtype_name={dtype_name!r}."
        )


def _validate_rounded_marginals(
    *,
    source_residual_mass: float,
    target_residual_mass: float,
    row_l1_error: float,
    col_l1_error: float,
) -> dict[str, float | str]:
    """
    CN: 验证标准 rank-one correction 的质量守恒和最终边缘可行性。
    EN: Validate mass conservation and final marginal feasibility of the standard rank-one correction.
    """
    values = (source_residual_mass, target_residual_mass, row_l1_error, col_l1_error)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Sinkhorn rounding produced non-finite residual diagnostics.")
    mass_difference = abs(source_residual_mass - target_residual_mass)
    if mass_difference > ROUNDING_VALIDATION_ATOL:
        raise RuntimeError(
            "Sinkhorn rounding residual masses disagree in float64: "
            f"source={source_residual_mass:.16e}, target={target_residual_mass:.16e}."
        )
    if max(row_l1_error, col_l1_error) > ROUNDING_VALIDATION_ATOL:
        raise RuntimeError(
            "Sinkhorn rounded transport failed marginal validation in float64: "
            f"row_l1={row_l1_error:.16e}, col_l1={col_l1_error:.16e}."
        )
    return {
        "rounding_residual_mass": float(source_residual_mass),
        "rounding_source_residual_mass": float(source_residual_mass),
        "rounding_target_residual_mass": float(target_residual_mass),
        "rounding_residual_mass_difference": float(mass_difference),
        "rounded_row_marginal_l1_error": float(row_l1_error),
        "rounded_col_marginal_l1_error": float(col_l1_error),
        "rounding_effective_dtype": "float64",
    }


def _squared_l2_objective_from_dense(
    problem: LinearOTProblem,
    plan: np.ndarray,
    *,
    row: np.ndarray,
    col: np.ndarray,
) -> float:
    source = np.asarray(problem.source_points, dtype=np.float64)
    target = np.asarray(problem.target_points, dtype=np.float64)
    cross = np.sum((source.T @ np.asarray(plan, dtype=np.float64)) * target.T)
    return float(
        np.dot(row, np.sum(source * source, axis=1))
        + np.dot(col, np.sum(target * target, axis=1))
        - 2.0 * cross
    )
