"""
CN: 仅供 HELLO 之前的 cluster 实现使用的旧报告辅助函数。
EN: Legacy reporting helpers used only by the pre-HELLO cluster implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from typing import Any as ConfigType

from ..core.solver import HierarchicalOTSolver


def _runtime_log(
    solver: HierarchicalOTSolver,
    category: str,
    message: str,
    *,
    flush: bool = False,
) -> None:
    if hasattr(solver, "_runtime_log"):
        solver._runtime_log(category, message, flush=flush)
    else:
        print(message, flush=flush)


def _warm_start_log_enabled(solver: HierarchicalOTSolver) -> bool:
    if hasattr(solver, "_runtime_log_enabled"):
        return bool(solver._runtime_log_enabled("warm_start"))
    return True


def _config_runtime_log_enabled(config: ConfigType, category: str) -> bool:
    runtime_logging = config.normalized_runtime_logging()
    if not bool(runtime_logging.get("enabled", True)):
        return False
    return bool(runtime_logging.get(category, True))


def _format_profile_components(
    components: Dict[str, float],
    max_items: int = 6,
    base_total: Optional[float] = None,
) -> str:
    if not components:
        return "-"
    items = [(k, float(v)) for k, v in components.items() if float(v) > 0.0]
    if not items:
        return "-"
    items.sort(key=lambda kv: kv[1], reverse=True)
    if base_total is None:
        base_total = sum(v for _, v in items)
    base = max(float(base_total), 1e-12)
    return ", ".join(f"{k}={v:.2f}s({(v/base)*100.0:.1f}%)" for k, v in items[:max_items])


def _fmt_optional_sci(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}e}"


def _fmt_optional_fixed(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _fmt_count_ratio(count: Any, total: Any) -> str:
    try:
        c = int(count)
        t = int(total)
    except (TypeError, ValueError):
        return "n/a"
    ratio = 0.0 if t <= 0 else float(c) / float(t)
    return f"{c:,}({ratio:.1%})"


def _format_convergence_check_fields(convergence_info: Dict[str, Any]) -> str:
    criterion = str(convergence_info.get("criterion", "n/a"))
    fields = [
        f"rel_obj_change={_fmt_optional_sci(convergence_info.get('signed_rel_obj_change'))}",
    ]
    if criterion == "dual_feasibility":
        fields.extend(
            [
                f"dual_feas={_fmt_optional_sci(convergence_info.get('dual_feasibility'))}",
                f"dual_feas_tol={_fmt_optional_sci(convergence_info.get('dual_feasibility_tol'))}",
            ]
        )
    else:
        fields.append(f"tol={_fmt_optional_sci(convergence_info.get('objective_tol'))}")
        if criterion == "objective":
            fields.append(
                f"plateau={convergence_info.get('plateau_counter', 'n/a')}/"
                f"{convergence_info.get('required_plateau', 'n/a')}"
            )
        if bool(convergence_info.get("require_dual_feasibility_convergence", False)):
            fields.extend(
                [
                    f"dual_feas={_fmt_optional_sci(convergence_info.get('dual_feasibility'))}",
                    f"dual_feas_tol={_fmt_optional_sci(convergence_info.get('dual_feasibility_tol'))}",
                ]
            )
    if bool(convergence_info.get("require_added_convergence", False)):
        fields.extend(
            [
                f"prev_added={convergence_info.get('previous_pricing_added', 'n/a')}",
                f"added_thr={convergence_info.get('added_convergence_threshold', 'n/a')}",
            ]
        )
    return ", ".join(fields)


def _fmt_pricing_added(pricing_info: Dict[str, Any]) -> str:
    if bool(pricing_info.get("skipped", False)) and pricing_info.get("added") is None:
        return "n/a"
    try:
        return f"{int(pricing_info.get('added', 0)):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pricing_added_violation_stats(pricing_info: Dict[str, Any]) -> str:
    if "added_rel_violation_gt_threshold" not in pricing_info:
        return ""
    try:
        large_count = int(pricing_info.get("added_rel_violation_gt_threshold", 0))
        threshold = float(pricing_info.get("added_rel_violation_threshold", 0.0))
    except (TypeError, ValueError):
        return ""
    max_rel = _fmt_optional_sci(pricing_info.get("added_rel_violation_max"))
    mean_rel = _fmt_optional_sci(pricing_info.get("added_rel_violation_mean"))
    return (
        f", added_relvio_gt={large_count:,}, "
        f"relvio_thr={threshold:.1e}, "
        f"max_relvio={max_rel}, "
        f"mean_relvio={mean_rel}"
    )


def _fmt_lp_dimensions(n_constraints: Any, n_vars: Any) -> str:
    try:
        m = int(n_constraints)
        n = int(n_vars)
    except (TypeError, ValueError):
        return "vars=n/a, constraints=n/a, A=n/a"
    if m < 0 or n < 0:
        return "vars=n/a, constraints=n/a, A=n/a"
    return f"vars={n:,}, constraints={m:,}, A=({m:,}, {n:,})"


def _print_warm_start_init_profile(
    solver: HierarchicalOTSolver,
    *,
    components: Dict[str, float],
    primal_nnz: int,
    active_size: int,
    bfs_added: int,
    profile_depth: Optional[int] = None,
) -> None:
    if not _warm_start_log_enabled(solver):
        return
    profile_prefix = (
        f"[Profile][WarmStart][D{int(profile_depth)}]"
        if profile_depth is not None
        else "[Profile][WarmStart]"
    )
    initial_pricing = float(components.get("initial_pricing", 0.0))
    warm_start_fill = float(components.get("warm_start_fill", 0.0))
    else_t = max(
        sum(float(v) for v in components.values() if float(v) > 0.0) - initial_pricing - warm_start_fill,
        0.0,
    )
    summary_components = {
        "initial_pricing": initial_pricing,
        "warm_start_fill": warm_start_fill,
        "else": else_t,
    }
    total = sum(float(v) for v in summary_components.values() if float(v) > 0.0)
    base = max(total, 1e-12)
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[Init] total={total:.2f}s "
        f"primal_nnz={int(primal_nnz):,} active={int(active_size):,} bfs_added={int(bfs_added):,}",
    )
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[Init][components] "
        f"{_format_profile_components(summary_components, max_items=3, base_total=base)}",
    )
    _runtime_log(solver, "warm_start", "---")


def _print_warm_start_iter_profile(
    solver: HierarchicalOTSolver,
    *,
    inner_iter: int,
    lp_pack: Dict[str, Any],
    pricing_info: Dict[str, Any],
    convergence_info: Dict[str, Any],
    cleaning_info: Optional[Dict[str, Any]] = None,
    finalize_components: Dict[str, float],
    iter_wall: float,
    solve_t: float,
    finalize_t: float,
    profile_depth: Optional[int] = None,
    dual_stabilization_info: Optional[Dict[str, Any]] = None,
) -> None:
    if not _warm_start_log_enabled(solver):
        return
    profile_prefix = (
        f"[Profile][WarmStart][D{int(profile_depth)}]"
        if profile_depth is not None
        else "[Profile][WarmStart]"
    )
    res = lp_pack.get("res")
    diag = lp_pack.get("diag") or {}
    lp_dims = _fmt_lp_dimensions(
        diag.get("n_constraints", getattr(res, "y", None).size if getattr(res, "y", None) is not None else None),
        diag.get("n_vars", getattr(res, "x", None).size if getattr(res, "x", None) is not None else None),
    )
    base = max(float(iter_wall), 1e-12)
    else_t = max(float(iter_wall) - float(solve_t) - float(finalize_t), 0.0)
    finalize_base = max(sum(float(v) for v in finalize_components.values() if float(v) > 0.0), 1e-12)
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[I{inner_iter + 1}][lp] "
        f"time={_fmt_optional_fixed(lp_pack.get('lp_time'), digits=2)}s, "
        f"iter={getattr(res, 'iterations', 'n/a')}, "
        f"{lp_dims}, "
        f"obj={_fmt_optional_fixed(getattr(res, 'obj_val', None))} / {_fmt_optional_fixed(getattr(res, 'dual_obj_val', None))}, "
        f"RelRes={_fmt_optional_sci(getattr(res, 'primal_feas', None))} / {_fmt_optional_sci(getattr(res, 'dual_feas', None))}, "
        f"Gap={_fmt_optional_sci(getattr(res, 'gap', None), digits=4)}"
    )
    if diag.get("matrix_value_mode") is not None:
        _runtime_log(
            solver,
            "warm_start",
            f"{profile_prefix}[I{inner_iter + 1}][lp_implicit] "
            f"mode={diag.get('matrix_value_mode')} "
            f"implicit_ax={1 if bool(diag.get('implicit_ax_enabled', False)) else 0} "
            f"implicit_aty={1 if bool(diag.get('implicit_aty_enabled', False)) else 0} "
            f"ax_agg={diag.get('implicit_ax_agg', 'none')} "
            f"row0_p50={diag.get('implicit_ax_row0_unique_p50', 'n/a')} "
            f"row1_p50={diag.get('implicit_ax_row1_unique_p50', 'n/a')}",
        )
    if isinstance(dual_stabilization_info, dict) and str(dual_stabilization_info.get("mode", "off")) != "off":
        _runtime_log(
            solver,
            "warm_start",
            f"{profile_prefix}[I{inner_iter + 1}][dual_stabilization] "
            f"mode={dual_stabilization_info.get('mode')}, "
            f"alpha={float(dual_stabilization_info.get('alpha', 0.0)):.3g}, "
            f"applied={bool(dual_stabilization_info.get('applied', False))}, "
            f"reset={bool(dual_stabilization_info.get('reset', False))}, "
            f"delta_rel={_fmt_optional_sci(dual_stabilization_info.get('delta_rel'))}",
        )
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[I{inner_iter + 1}][pricing] "
        f"time={_fmt_optional_fixed(pricing_info.get('time'), digits=2)}s, "
        f"active={int(pricing_info.get('active_before', 0)):,}, "
        f"found={int(pricing_info.get('found', 0)):,}, "
        f"added={_fmt_pricing_added(pricing_info)}"
        f"{_fmt_pricing_added_violation_stats(pricing_info)}"
    )
    if cleaning_info is not None:
        _runtime_log(
            solver,
            "warm_start",
            f"{profile_prefix}[I{inner_iter + 1}][cleaning] "
            f"n_s={int(cleaning_info.get('n_s', 0)):,}, "
            f"n_t={int(cleaning_info.get('n_t', 0)):,}, "
            f"budget={int(cleaning_info.get('budget', 0)):,}, "
            f"before={int(cleaning_info.get('before', 0)):,}, "
            f"need={int(cleaning_info.get('need', 0)):,}, "
            f"removed={int(cleaning_info.get('removed', 0)):,}, "
            f"nnz={int(cleaning_info.get('nnz', 0)):,}, "
            f"old_zero={int(cleaning_info.get('old_zero', 0)):,}, "
            f"old_pool={int(cleaning_info.get('old_pool', 0)):,}, "
            f"remove_zero={_fmt_count_ratio(cleaning_info.get('removed_old_zero_flow', 0), cleaning_info.get('removed', 0))}, "
            f"remove_nnz={_fmt_count_ratio(cleaning_info.get('removed_old_nonzero_flow', 0), cleaning_info.get('removed', 0))}, "
            f"overflow={int(cleaning_info.get('budget_overflow', 0))}"
        )
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[I{inner_iter + 1}][convergence_check] "
        f"{_format_convergence_check_fields(convergence_info)}"
    )
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[I{inner_iter + 1}][finalize] "
        f"{_format_profile_components(finalize_components, max_items=4, base_total=finalize_base)}"
    )
    _runtime_log(
        solver,
        "warm_start",
        f"{profile_prefix}[I{inner_iter + 1}] "
        f"total={iter_wall:.2f}s "
        f"solve={solve_t:.2f}s({(solve_t/base)*100.0:.1f}%) "
        f"finalize={finalize_t:.2f}s({(finalize_t/base)*100.0:.1f}%) "
        f"else={else_t:.2f}s({(else_t/base)*100.0:.1f}%)"
    )
    _runtime_log(solver, "warm_start", "---")
