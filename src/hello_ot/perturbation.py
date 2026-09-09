from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from .cost import assign_from_complete_dual, begin_refinement_by_cost
from .hierarchy.construction import slice_node_arrays
from .refinement.iterations import check_optimality, finish_interrupted_iteration, solve_lp, update_support
from .refinement.reentry import ReentryDetector
from .state import WarmStartState


@dataclass
class CostPerturbationRun:
    """
    CN: 单次求解的成本阶段状态；保持原成本、全局索引及各阶段统计。
    EN: Per-solve cost-stage state retaining original costs, global identities, and stage statistics.
    """

    policy: str
    relative_scale: float
    random_seed: int
    source_index: np.ndarray
    target_index: np.ndarray
    activated: bool = False
    phase: str = "original"
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    original: Any = None
    stage_started: float = field(default_factory=time.perf_counter)

    def new_detector(self) -> ReentryDetector | None:
        """
        CN: 每层清空历史；仅首次原成本阶段允许检测。
        EN: Reset history at each level and detect only during the initial original-cost phase.
        """
        return ReentryDetector() if self.policy == "auto" and not self.activated else None

    def record_stage(self, levels: list[dict]) -> None:
        """
        CN: 保存完整阶段记录，阶段间不保留稀疏解的大数组。
        EN: Save complete stage records without retaining sparse solution arrays between stages.
        """
        now = time.perf_counter()
        self.stages.append({"name": self.phase, "levels": list(levels), "wall_time": now - self.stage_started})
        self.stage_started = now
        levels.clear()

    def activate(self, execution: Any) -> None:
        """
        CN: 按全尺寸原问题采样一次尺度，并激活全局一致的扰动。
        EN: Sample the full original problem once and activate a globally consistent perturbation.
        """
        if self.activated:
            raise RuntimeError("cost perturbation may be activated only once")
        seed = int(self.random_seed) & 0xFFFFFFFF
        for character in "full_cost_perturb":
            seed = (seed * 1664525 + ord(character) + 1013904223) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        n, m = len(execution.source_F), len(execution.target_G)
        rows = np.argsort(self.source_index)[rng.integers(n, size=65536)]
        cols = np.argsort(self.target_index)[rng.integers(m, size=65536)]
        sampled = np.empty(65536, dtype=np.float64)
        bilinear = execution.cost_context.score_family == "inner_product"
        for start in range(0, 65536, 1024):
            rr, cc = rows[start:start + 1024], cols[start:start + 1024]
            source, target = execution.source_F[rr], execution.target_G[cc]
            if bilinear:
                sampled[start:start + 1024] = (
                    execution.source_cost[rr] + execution.target_cost[cc]
                    - execution.cost_context.dot_scale * np.einsum("ij,ij->i", source, target)
                )
            else:
                difference = np.abs(source - target)
                cost = execution.cost_context.cost_type
                sampled[start:start + 1024] = (
                    difference.sum(axis=1) if cost == "l1" else
                    difference.max(axis=1) if cost == "linf" else
                    np.sqrt((difference * difference).sum(axis=1))
                )
        mean = float(sampled.mean())
        sigma = self.relative_scale * mean
        if not np.isfinite(mean) or mean <= 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("cost perturbation requires a finite positive sampled cost mean and scale")
        self.original = (execution.cost_context, execution.source_F, execution.target_G,
                         execution.source_cost, execution.target_cost)
        self.metadata.update(
            mode="rank2" if bilinear else "index_hash",
            relative_scale=self.relative_scale,
            sampled_mean=mean,
            sigma=sigma,
            sample_pairs=65536,
            seed=seed,
            transfer="dual_only",
            support_proposal_cost="perturbed",
        )
        if bilinear:
            scale = np.sqrt(0.5 * sigma / execution.cost_context.dot_scale)
            source_angle = self.source_index.astype(np.float64) * 12.9898 + seed * 0.001
            target_angle = self.target_index.astype(np.float64) * 78.233
            source_extra = scale * np.column_stack((np.sin(source_angle), np.cos(source_angle)))
            target_extra = -scale * np.column_stack((np.cos(target_angle), np.sin(target_angle)))
            execution.source_F = np.concatenate((execution.source_F, source_extra.astype(np.float32)), axis=1)
            execution.target_G = np.concatenate((execution.target_G, target_extra.astype(np.float32)), axis=1)
            execution.source_cost = execution.source_cost + 0.5 * sigma
        else:
            execution.cost_context = replace(execution.cost_context, perturbation={
                "mode": "full_cost_perturb", "noise": "index_hash", "sigma": sigma, "seed": seed,
                "source_global_index": self.source_index, "target_global_index": self.target_index,
            })
        self.activated = True
        self.phase = "perturbed"


def initialize_from_dual(node: Any, dual: Any, execution: Any, *, proposal: Any = None) -> Any:
    """
    CN: 只继承双侧对偶势，以候选成本重建支撑，再按求解成本初始化 LP。
    EN: Inherit only both dual potentials, rebuild support with proposal costs, then initialize the LP with solve costs.
    """
    from .algorithm import _DualAssignmentResult, _InitializedHierarchyLevel

    started = time.perf_counter()
    dual_np = dual.detach().cpu().numpy() if torch.is_tensor(dual) else np.asarray(dual)
    if dual_np.size != node.n_source + node.n_target or not np.isfinite(dual_np).all():
        raise ValueError("cost-stage transfer requires finite dual potentials of the expected size")
    state = WarmStartState(
        rows=np.empty(0, dtype=np.int32), cols=np.empty(0, dtype=np.int32),
        x_prev=np.empty(0, dtype=np.float64), dual_uv=dual_np.copy(),
        n_source=node.n_source, n_target=node.n_target,
    )
    arrays = slice_node_arrays(
        node=node, source_F_full=execution.source_F, target_G_full=execution.target_G,
        source_cost_vec_full=execution.source_cost, target_cost_vec_full=execution.target_cost,
        source_mass_raw=execution.source_mass, target_mass_raw=execution.target_mass,
    )
    context = execution.context_for(node)
    proposal_context, proposal_arrays = (context, arrays) if proposal is None else proposal
    counts, profiles, stats = [], [], []
    for side in ("target", "source"):
        state, stat, profile, _ = assign_from_complete_dual(
            cost_context=proposal_context, state=state,
            source_F=proposal_arrays["source_F"], target_G=proposal_arrays["target_G"],
            source_cost_vec=proposal_arrays["source_cost_vec"], target_cost_vec=proposal_arrays["target_cost_vec"],
            known_side=side, assignment_topk=execution.assignment_topk, tracer=execution.tracer,
            trace_args={"phase": execution.perturbation.phase, "depth": node.depth},
            backend=execution.config.backend, torch_device=execution.config.torch_device,
        )
        counts.append(int(state.rows.numel()) if torch.is_tensor(state.rows) else len(state.rows))
        profiles.append(profile)
        stats.append(stat)
    assignment = _DualAssignmentResult(
        state=state, topk_stats=stats[-1],
        augment_profile={"total_time": sum(p.get("total_time", 0.0) for p in profiles),
                         "forward": profiles[0], "reverse": profiles[1]},
        dual_completion_profile={},
        forward_topk_stats=stats[0], support_after_forward=counts[0],
        reverse_topk_stats=stats[1], support_after_reverse=counts[1],
    )
    level_stopping_norm = execution.stopping_norm_for(node)
    refinement = begin_refinement_by_cost(
        cost_context=context,
        source_F=arrays["source_F"], target_G=arrays["target_G"],
        source_cost_vec=arrays["source_cost_vec"], target_cost_vec=arrays["target_cost_vec"],
        source_mass=arrays["source_mass"], target_mass=arrays["target_mass"],
        warm_start=state, config=execution.config, skip_initial_pricing=context.score_family == "inner_product",
        dual_feasibility_tol=execution.dual_feasibility_tol,
        dual_feasibility_norm=level_stopping_norm,
        lp_termination_norm=level_stopping_norm,
        tracer=execution.tracer, trace_prefix="hello." + execution.perturbation.phase,
        pricing_index_pool=execution.pricing_index_pool, warm_start_profile_depth=node.depth,
    )
    initialization_elapsed = float(time.perf_counter() - started)
    return _InitializedHierarchyLevel(
        node=node, arrays=arrays, state=state, split_axis="both", trace_args={},
        started_at=started, initialization_elapsed=initialization_elapsed,
        support_before_solve=counts[-1], stitch_profile={},
        assignment=assignment, refinement=refinement, refine_span=None,
    )


def switch_level_to_perturbed_cost(initialized: Any, iteration: Any, certificate: Any,
                                  execution: Any, levels: list[dict], detector: ReentryDetector) -> Any:
    """
    CN: 完成触发轮记录并切换当前层；不在辅助函数内隐藏 refinement 循环。
    EN: Complete trigger-round records and switch this level without hiding a refinement loop.
    """
    from .algorithm import _finalize_hierarchy_level

    dual = iteration.dual
    finish_interrupted_iteration(initialized.refinement.runtime, iteration, certificate)
    from .progress import report_iteration, report_level_start, report_perturbation_activated

    report_iteration(execution, initialized.refinement.runtime)
    _finalize_hierarchy_level(initialized, levels=levels, execution=execution)
    run = execution.perturbation
    levels[-1]["stop_reason"] = "cost_perturbation_requested"
    run.metadata.update(trigger_level=initialized.node.depth, trigger_iteration=iteration.index + 1,
                        reentry_count=detector.reentry_count)
    run.record_stage(levels)
    run.activate(execution)
    report_perturbation_activated(execution)
    report_level_start(execution, initialized.node)
    restarted = initialize_from_dual(initialized.node, dual, execution)
    from .progress import report_level_initialized

    report_level_initialized(execution, restarted)
    return restarted


def finish_original_problem(root: Any, state: Any, execution: Any, levels: list[dict]) -> tuple[Any, Any]:
    """
    CN: 以扰动成本构建候选支撑，在全尺寸原问题上完成最后 refinement。
    EN: Propose support with perturbed costs and finish refinement on the full original problem.
    """
    from .algorithm import _finalize_hierarchy_level

    run = execution.perturbation
    run.record_stage(levels)
    from .progress import (
        report_iteration,
        report_level_initialized,
        report_level_start,
        report_original_transition,
    )

    report_original_transition(execution)
    proposal = (execution.context_for(root), {
        "source_F": execution.source_F, "target_G": execution.target_G,
        "source_cost_vec": execution.source_cost, "target_cost_vec": execution.target_cost,
    })
    (execution.cost_context, execution.source_F, execution.target_G,
     execution.source_cost, execution.target_cost) = run.original
    run.original = None
    run.phase = "original_final"
    report_level_start(execution, root)
    initialized = initialize_from_dual(root, state.dual_uv, execution, proposal=proposal)
    report_level_initialized(execution, initialized)
    del proposal
    refinement = initialized.refinement
    for index in range(refinement.max_iterations):
        iteration = solve_lp(refinement.runtime, index)
        certificate = check_optimality(refinement.runtime, iteration)
        if certificate.converged:
            report_iteration(execution, refinement.runtime)
            break
        update_support(refinement.runtime, iteration, certificate)
        report_iteration(execution, refinement.runtime)
    return _finalize_hierarchy_level(initialized, levels=levels, execution=execution)
