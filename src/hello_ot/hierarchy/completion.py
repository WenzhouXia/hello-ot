"""
CN: 在层级完成点将求解输出装配为强类型记录。
EN: Assemble typed records from solver outputs at hierarchy completion points.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from hello_ot.types import CoarsestSolveStats, InitializationStats, IterationStats, LevelResult, RefinementResult, RefinementSummary
from .records import HierarchyLevelRecord

if TYPE_CHECKING:
    from .execution import InitializedHierarchyLevel


def coarsest_level_record(*, path: str, depth: int, n_source: int, n_target: int,
                          support_size: int, elapsed: float,
                          solve_summary: Mapping[str, Any]) -> HierarchyLevelRecord:
    """
    CN: 在 leaf 输出边界要求必要字段存在，拒绝把遗漏变成 NaN 或零。
    EN: Require essential fields at the leaf output boundary instead of turning omissions into NaN or zero.
    """
    objective = float(solve_summary["distance"])
    lp_time = float(solve_summary["lp_solve_time_total"])
    return HierarchyLevelRecord(
        path=path, lp_solve_time_total=lp_time,
        result=LevelResult(level_index=depth, kind="coarsest", n_source=n_source, n_target=n_target,
                           initialization=None, wall_time=elapsed,
                           solve=CoarsestSolveStats(wall_time=elapsed, objective=objective, support_size=support_size)),
        solve_diagnostics={key: value for key, value in solve_summary.items()
                           if key not in {"distance", "lp_solve_time_total"}},
    )


def refined_level_record(*, initialized: InitializedHierarchyLevel, node_result: Mapping[str, Any],
                         assignment_topk: int, support_after: int, elapsed: float,
                         diagnostics: Mapping[str, Any], solve_diagnostics: Mapping[str, Any]) -> HierarchyLevelRecord:
    """
    CN: 在层完成点构建公开类型，后续传递相同的迭代记录引用。
    EN: Build public types at level completion and carry the same iteration-record references thereafter.
    """
    node = initialized.node
    assignment = initialized.assignment
    initialization = node_result["warm_start_init"]
    components = initialization["components"]
    iterations = tuple(node_result["warm_start_iteration_records"])
    if not all(isinstance(item, IterationStats) for item in iterations):
        raise TypeError("refinement records must contain IterationStats")
    converged = bool(iterations and iterations[-1].convergence.passed)
    before = initialized.support_before_solve
    initial = InitializationStats(
        wall_time=initialized.initialization_elapsed,
        dual_propagation_time=float(assignment.dual_completion_profile.get("total_time", 0.0) or 0.0),
        dual_assignment_time=float(assignment.augment_profile.get("total_time", 0.0) or 0.0),
        northwest_augmentation_time=float(components.get("bfs_skeleton", 0.0) or 0.0),
        inherited_dual_side=None if initialized.split_axis == "both" else (
            "target" if initialized.split_axis == "source" else "source"),
        assignment_topk=assignment_topk,
        support_after_forward=int(assignment.support_after_forward or before),
        support_after_reverse=assignment.support_after_reverse,
        northwest_edges_added=int(initialization["northwest_added"]),
        initial_active_support_size=int(initialization["post_pricing_support_size"] or before),
    )
    solve = RefinementResult(iterations=iterations, summary=RefinementSummary(
        wall_time=float(node_result["time"]), iterations=len(iterations),
        final_objective=float(node_result["distance"]), final_active_support_size=support_after,
        peak_active_support_size=max((item.support.peak for item in iterations), default=before),
        stop_reason="converged" if converged else "max_refinement_iterations", converged=converged,
    ))
    return HierarchyLevelRecord(
        path=node.path, lp_solve_time_total=float(node_result["lp_solve_time_total"]),
        support_before_solve=before,
        result=LevelResult(level_index=node.depth, kind="refined", n_source=node.n_source,
                           n_target=node.n_target, initialization=initial, solve=solve, wall_time=elapsed),
        diagnostics=diagnostics, solve_diagnostics=solve_diagnostics,
    )
