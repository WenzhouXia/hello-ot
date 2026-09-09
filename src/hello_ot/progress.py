from __future__ import annotations

import math
import sys
from typing import Any, Mapping


def _enabled(value: Any) -> bool:
    return str(getattr(value, "verbosity", value)) in {"compact", "detailed"}


def _number(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{parsed:.2e}" if math.isfinite(parsed) else "n/a"


def _objective(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{parsed:.8f}" if math.isfinite(parsed) else "n/a"


def _elapsed(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(parsed):
        return "n/a"
    if parsed < 0.01:
        return f"{parsed:.3f}s"
    return f"{parsed:.2f}s" if parsed < 10.0 else f"{parsed:.1f}s"


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def stage_name(execution: Any) -> str:
    phase = str(getattr(getattr(execution, "perturbation", None), "phase", "original"))
    return "original" if phase == "original_final" else phase


def level_kind(execution: Any, *, coarsest: bool = False) -> str:
    if coarsest:
        return "coarsest"
    phase = str(getattr(getattr(execution, "perturbation", None), "phase", "original"))
    return "final-refinement" if phase == "original_final" else "refined"


def report_header(
    verbosity: Any,
    *,
    backend: str,
    device: str,
    cost: str,
    n_source: int,
    n_target: int,
    dimension: int,
    perturbation: str,
) -> None:
    if not _enabled(verbosity):
        return
    _emit(
        "HELLO | "
        f"backend={backend} device={device} cost={cost} "
        f"shape={n_source:,}x{n_target:,} d={dimension:,} cost_perturbation={perturbation}"
    )


def report_level_start(execution: Any, node: Any, *, coarsest: bool = False) -> None:
    if not _enabled(execution.config):
        return
    _emit(
        f"[{stage_name(execution)}] level={int(node.depth)} "
        f"kind={level_kind(execution, coarsest=coarsest)} "
        f"shape={int(node.n_source):,}x{int(node.n_target):,}"
    )


def report_level_initialized(execution: Any, initialized: Any) -> None:
    """
    CN: 在首轮 LP 前报告完整的层间 initialization wall time。
    EN: Report the complete inter-level initialization wall time before the first LP.
    """
    if not _enabled(execution.config):
        return
    _emit(
        "  init done | "
        f"active={int(initialized.support_before_solve):,} "
        f"elapsed={_elapsed(initialized.initialization_elapsed)}"
    )


def report_iteration(execution: Any, runtime: Any) -> None:
    if not _enabled(execution.config) or not runtime.iteration_records:
        return
    record: Mapping[str, Any] = runtime.iteration_records[-1]
    lp = dict(record.get("lp_diag") or {})
    convergence = dict(record.get("convergence_info") or {})
    pricing = dict(record.get("pricing_info") or {})
    components = dict(record.get("finalize_components") or {})
    elapsed = record.get("wall_time")
    if elapsed is None:
        elapsed = float(record.get("lp_time", 0.0) or 0.0) + sum(
            float(value or 0.0) for value in components.values()
        )
    _emit(
        "  "
        f"iter={int(record.get('iter', len(runtime.iteration_records)))} "
        f"obj={_objective(record.get('objective'))} "
        f"pFeas={_number(lp.get('solver_algorithm_rel_primal_res_l2'))} "
        f"dFeas={_number(convergence.get('stopping_dual_feasibility', convergence.get('dual_feasibility')))} "
        f"gap={_number(lp.get('solver_relative_primal_dual_gap'))} "
        f"active={int(record.get('active_support_after_cleaning', 0) or 0):,} "
        f"added={int(pricing.get('added_count', pricing.get('added', 0)) or 0):,} "
        "| "
        f"lp={_elapsed(record.get('lp_time'))} "
        f"full_scan={_elapsed(record.get('full_scan_time'))} "
        f"elapsed={_elapsed(elapsed)}"
    )


def report_level_done(
    execution: Any,
    node: Any,
    result: Mapping[str, Any],
    *,
    active: int,
    initialization_elapsed: float,
    elapsed: float,
) -> None:
    if not _enabled(execution.config):
        return
    records = list(result.get("warm_start_iteration_records") or [])
    final_active = (
        int(records[-1].get("active_support_after_cleaning", active) or active)
        if records
        else int(active)
    )
    lp_time = sum(float(record.get("lp_time", 0.0) or 0.0) for record in records)
    full_scan_time = sum(
        float(record.get("full_scan_time", 0.0) or 0.0) for record in records
    )
    _emit(
        f"  level done | iterations={len(records)} "
        f"obj={_objective(result.get('distance'))} active={final_active:,} "
        "| "
        f"init={_elapsed(initialization_elapsed)} "
        f"lp={_elapsed(lp_time)} full_scan={_elapsed(full_scan_time)} "
        f"elapsed={_elapsed(elapsed)}"
    )
    _emit("")


def report_coarsest_done(execution: Any, node: Any, level: Mapping[str, Any]) -> None:
    if not _enabled(execution.config):
        return
    solve = dict(level.get("solve_summary") or {})
    _emit(
        "  coarsest done | "
        f"obj={_objective(solve.get('distance'))} "
        f"support={int(level.get('support_size', 0) or 0):,} "
        f"lp={_elapsed(level.get('lp_solve_time_total'))} "
        f"elapsed={_elapsed(level.get('time'))}"
    )
    _emit("")


def report_perturbation_activated(execution: Any) -> None:
    run = execution.perturbation
    if not _enabled(execution.config) or run.policy != "auto":
        return
    _emit(
        "cost perturbation activated | "
        f"level={int(run.metadata['trigger_level'])} "
        f"iter={int(run.metadata['trigger_iteration'])} "
        f"sigma={_number(run.metadata.get('sigma'))}"
    )


def report_original_transition(execution: Any) -> None:
    if _enabled(execution.config):
        _emit("cost perturbation | perturbed stage completed; switching back to original cost")


def final_metrics(result: Any) -> tuple[Any, Any, Any, Any] | None:
    from .types import RefinementResult

    refined = [level for level in result.solve_stage.levels if isinstance(level.solve, RefinementResult)]
    if not refined or not refined[-1].solve.iterations:
        return None
    final = refined[-1].solve.iterations[-1]
    pfeas = final.solve_lp.primal_feasibility
    dfeas = final.convergence.dual_feasibility
    gap = final.solve_lp.primal_dual_gap
    finite = [abs(float(value)) for value in (pfeas, dfeas, gap) if value is not None and math.isfinite(float(value))]
    kkt = max(finite) if len(finite) == 3 else None
    return pfeas, dfeas, gap, kkt


def report_final(verbosity: Any, result: Any) -> None:
    if not _enabled(verbosity):
        return
    metrics = final_metrics(result)
    if metrics is None:
        _emit(
            f"HELLO converged | obj={_objective(result.objective)} "
            f"elapsed={_elapsed(result.total_wall_time)}"
        )
        return
    pfeas, dfeas, gap, kkt = metrics
    final_level = next(
        level for level in reversed(result.solve_stage.levels) if hasattr(level.solve, "summary")
    )
    prefix = "HELLO converged" if final_level.solve.summary.converged else "HELLO not converged"
    reason = "" if final_level.solve.summary.converged else f" | reason={final_level.solve.summary.stop_reason}"
    _emit(
        f"{prefix}{reason} | obj={_objective(result.objective)} "
        f"pFeas={_number(pfeas)} dFeas={_number(dfeas)} gap={_number(gap)} "
        f"KKT={_number(kkt)} elapsed={_elapsed(result.total_wall_time)}"
    )


def report_failure(verbosity: Any, error: BaseException, elapsed: float) -> None:
    if _enabled(verbosity):
        _emit(f"HELLO failed | error={error} elapsed={_elapsed(elapsed)}")


__all__ = [
    "final_metrics",
    "report_coarsest_done",
    "report_failure",
    "report_final",
    "report_header",
    "report_iteration",
    "report_level_done",
    "report_level_initialized",
    "report_level_start",
    "report_original_transition",
    "report_perturbation_activated",
]
