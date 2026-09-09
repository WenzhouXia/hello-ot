from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hello_ot.config import _AlgorithmConfig
from hello_ot.types import DualPotentials
from hello_ot.variants._balanced import (
    BalancedOTResult,
    solve_bilinear_subproblem,
)

from .dual_atoms import DualAtomMixture, canonicalize_atom
from .objective import compute_translated_dual_state, evaluate_primal_certificate


def _run_ott_sinkhorn_marginals(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    *,
    epsilon: float = 1e-2,
    batch_size: int = 512,
    threshold: float = 1e-3,
    max_iterations: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CN: 调用 OTT-JAX 求解点云上的在线分块熵正则非平衡 OT，获取近似边际分布与势函数。
    EN: Run OTT-JAX online blockwise entropic UOT on point clouds to obtain marginals and potentials.
    """
    try:
        import jax
        import jax.numpy as jnp
        from ott.geometry import costs, pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError as err:
        raise ImportError(
            "ott-jax is required for initialization='ott_sinkhorn'. "
            "Please install it with `pip install 'hello-ot[ott]'` or `pip install ott-jax jax`, "
            "or use initialization=None for cold start."
        ) from err

    tau_a = float(rho_source / (rho_source + epsilon))
    tau_b = float(rho_target / (rho_target + epsilon))
    a_weights = source_mass ** (1.0 + epsilon / rho_source)
    b_weights = target_mass ** (1.0 + epsilon / rho_target)

    # CN: 借助 PointCloud 的 batch_size 避免显式物化 O(N*M) 代价矩阵
    # EN: Leverage PointCloud batch_size to avoid materializing the O(N*M) cost matrix
    geom = pointcloud.PointCloud(
        jnp.asarray(source_points, dtype=jnp.float32),
        jnp.asarray(target_points, dtype=jnp.float32),
        cost_fn=costs.SqEuclidean(),
        batch_size=int(batch_size),
        scale_cost=1.0,
        epsilon=float(epsilon),
    )
    prob = linear_problem.LinearProblem(
        geom,
        a=jnp.asarray(a_weights, dtype=jnp.float32),
        b=jnp.asarray(b_weights, dtype=jnp.float32),
        tau_a=tau_a,
        tau_b=tau_b,
    )
    solver = sinkhorn.Sinkhorn(
        lse_mode=True,
        threshold=float(threshold),
        max_iterations=int(max_iterations),
        recenter_potentials=True,
    )
    out = solver(prob)

    source_marginal = np.asarray(jax.device_get(out.marginal(1)), dtype=np.float64)
    target_marginal = np.asarray(jax.device_get(out.marginal(0)), dtype=np.float64)
    target_potential = np.asarray(jax.device_get(out.g), dtype=np.float64)
    return source_marginal, target_marginal, target_potential


def _run_pot_sinkhorn_marginals(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    *,
    reg: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    CN: 调用 POT 求解稠密代价上的非平衡 Sinkhorn，获取近似边际分布。
    EN: Run POT unbalanced Sinkhorn on dense cost to obtain approximate marginals.
    """
    try:
        import ot
    except ImportError as err:
        raise ImportError("POT is required for initialization='pot_sinkhorn'.") from err

    M = ot.dist(source_points, target_points, metric="sqeuclidean")
    gamma = ot.unbalanced.sinkhorn_unbalanced(
        source_mass,
        target_mass,
        M,
        reg=float(reg),
        reg_m=float(rho_source),
        method="sinkhorn",
        numItermax=2000,
    )
    source_marginal = np.asarray(gamma.sum(axis=1), dtype=np.float64)
    target_marginal = np.asarray(gamma.sum(axis=0), dtype=np.float64)
    return source_marginal, target_marginal, None


def initialize_unbalanced_fcfw(
    source_factors: np.ndarray,
    target_factors: np.ndarray,
    source_offset: np.ndarray,
    target_offset: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    config: _AlgorithmConfig,
    *,
    initialization: Optional[str] = "ott_sinkhorn",
) -> Tuple[DualAtomMixture, Dict[str, Any], BalancedOTResult]:
    """
    CN: 构建首个可行 HELLO dual atom，建立 DualAtomMixture 并评估初始 primal certificate。
    EN: Construct initial feasible HELLO dual atom, create DualAtomMixture, and evaluate initial primal certificate.
    """
    source_total = float(np.sum(source_mass))
    target_total = float(np.sum(target_mass))
    log_source = np.log(source_mass)
    log_target = np.log(target_mass)

    target_potential: Optional[np.ndarray] = None
    init_mode = str(initialization).strip().lower() if initialization is not None else "none"

    if init_mode == "ott_sinkhorn":
        src_marg, tgt_marg, target_potential = _run_ott_sinkhorn_marginals(
            source_points,
            target_points,
            source_mass,
            target_mass,
            rho_source=rho_source,
            rho_target=rho_target,
        )
        src_norm = src_marg / np.sum(src_marg)
        tgt_norm = tgt_marg / np.sum(tgt_marg)
    elif init_mode == "pot_sinkhorn":
        src_marg, tgt_marg, target_potential = _run_pot_sinkhorn_marginals(
            source_points,
            target_points,
            source_mass,
            target_mass,
            rho_source=rho_source,
            rho_target=rho_target,
        )
        src_norm = src_marg / np.sum(src_marg)
        tgt_norm = tgt_marg / np.sum(tgt_marg)
    elif init_mode in ("none", "cold"):
        n_source = int(source_mass.size)
        n_target = int(target_mass.size)
        mixture = DualAtomMixture.zeros(n_source, n_target)
        return mixture, None, None
    else:
        raise ValueError(
            f"Unknown initialization={initialization!r}; expected 'ott_sinkhorn', 'pot_sinkhorn', or None."
        )

    # CN: 求解首个平衡 OT 子问题以生成第一个 dual atom
    # EN: Solve initial balanced OT subproblem to yield the first dual atom
    inherited_dual = (
        DualPotentials(target=target_potential)
        if target_potential is not None
        else None
    )
    subproblem = solve_bilinear_subproblem(
        source_points=source_factors,
        target_points=target_factors,
        source_offset=source_offset,
        target_offset=target_offset,
        source_mass=src_norm,
        target_mass=tgt_norm,
        config=config,
        inherited_dual=inherited_dual,
        preparation="source_from_target" if inherited_dual is not None else "preserve",
    )

    atom_src, atom_tgt = canonicalize_atom(
        subproblem.hello_result.solution.source_dual,
        subproblem.hello_result.solution.target_dual,
    )
    mixture = DualAtomMixture.from_atom(atom_src, atom_tgt)

    # CN: 评估初始对偶状态与 primal certificate
    # EN: Evaluate initial dual state and primal certificate
    state0 = compute_translated_dual_state(
        *mixture.current(),
        log_source,
        log_target,
        rho_source,
        rho_target,
        source_total,
        target_total,
    )
    cert0 = evaluate_primal_certificate(
        subproblem.coupling,
        state0["transported_mass"],
        source_mass,
        target_mass,
        source_factors,
        target_factors,
        source_offset,
        target_offset,
        rho_source,
        rho_target,
    )
    cert0["dual_objective"] = state0["dual_objective"]

    return mixture, cert0, subproblem


__all__ = ["initialize_unbalanced_fcfw"]
