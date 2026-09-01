from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch

from hello_ot._internal.instrumentation.driver_memory import DriverMemoryTracker
from hello_ot.kernels.bilinear_topk import (
    raw_lowrank_bidir_feasibility_pricing_scan,
    validate_fused_lowrank_bidir_scan_config,
)
from hello_ot.refinement.stopping import lowrank_dual_feasibility_infeasibility
from hello_ot.config import SolverRuntimeConfig


DualFeasibilityNorm = Literal["l2", "linf"]


def normalize_dual_feasibility_norm(value: str) -> DualFeasibilityNorm:
    """
    CN: 规范化 HELLO 最精细层使用的 relative dual-feasibility 范数。
    EN: Normalize the relative dual-feasibility norm used at the HELLO finest level.
    """
    normalized = str(value).strip().lower()
    if normalized not in {"l2", "linf"}:
        raise ValueError("finest_dual_feasibility_norm must be one of: l2, linf")
    return normalized  # type: ignore[return-value]


def stopping_dual_feasibility(
    *,
    norm: str,
    l2_value: float,
    diagnostics: Dict[str, Any],
) -> float:
    """
    CN: 从同一次 full-problem scan 中选择实际用于判敛的 relative dual feasibility。
    EN: Select the relative dual feasibility used for stopping from one full-problem scan.
    """
    normalized = normalize_dual_feasibility_norm(norm)
    if normalized == "l2":
        return float(l2_value)
    value = diagnostics.get("relative_linf_dual_feasibility")
    if value is None:
        raise RuntimeError("L-infinity stopping requires relative_linf_dual_feasibility diagnostics.")
    return float(value)


@dataclass
class LowrankDualFeasibilityScan:
    """
    CN: lowrank dual-feasibility scan 结果，可选携带 fused pricing 候选边。
    EN: Lowrank dual-feasibility scan result, optionally carrying fused pricing candidates.
    """

    dual_feasibility: float
    dual_feasibility_source: str
    scan_time: float
    diagnostics: Dict[str, Any]
    rows: Optional[np.ndarray] = None
    cols: Optional[np.ndarray] = None
    peak_mem_mib: Optional[float] = None

    @property
    def has_candidate_pairs(self) -> bool:
        return self.rows is not None and self.cols is not None

    def pricing_extra_info(self) -> Dict[str, Any]:
        info = dict(self.diagnostics)
        info["dual_feasibility"] = float(self.dual_feasibility)
        info["dual_feasibility_source"] = str(self.dual_feasibility_source)
        info["dual_feasibility_scan_time"] = float(self.scan_time)
        if self.peak_mem_mib is not None:
            info["dual_feas_peak_mem_mib"] = float(self.peak_mem_mib)
            info["dual_feas_peak_mem_source"] = "driver_mem_get_info"
        return info


def run_lowrank_dual_feasibility_scan(
    *,
    config: SolverRuntimeConfig,
    lvl_s: Any,
    lvl_t: Any,
    dual_uv: np.ndarray,
    inner_iter: int,
    trace_collector: Optional[Any],
    trace_prefix: str,
    dual_feasibility_norm: DualFeasibilityNorm = "l2",
    fused_span_name: str = "early_fused_lowrank_bidir_scan",
    standalone_span_name: str = "early_dual_feasibility_scan",
) -> LowrankDualFeasibilityScan:
    """
    CN: 执行 dual-feasibility 判敛需要的 lowrank scan；fused 模式同时返回 pricing 候选边。
    EN: Run the lowrank scan required by dual-feasibility convergence; fused mode also returns pricing candidates.
    """
    use_fused = bool(getattr(config, "use_fused_lowrank_feasibility_pricing", True))
    scan_start = time.perf_counter()
    diagnostics: Dict[str, Any] = {
        "inner_iter": int(inner_iter + 1),
        "n_source": int(len(lvl_s.points)),
        "n_target": int(len(lvl_t.points)),
        "use_fused_lowrank_feasibility_pricing": bool(use_fused),
    }

    if str(getattr(config, "backend", "native")) == "torch":
        from hello_ot.kernels.torch_scan import bidirectional_violation_scan

        scan = bidirectional_violation_scan(
            source_points=lvl_s.points,
            target_points=lvl_t.points,
            source_offset=lvl_s.cost_vec,
            target_offset=lvl_t.cost_vec,
            source_dual=dual_uv[: len(lvl_s.points)],
            target_dual=dual_uv[len(lvl_s.points) :],
            score_family="inner_product",
            cost_type="lowrank",
            dot_scale=float(getattr(config, "dot_scale", 1.0)),
            topk=int(getattr(config, "pricing_topk", 2)),
            theta=0.0,
            device=str(getattr(config, "torch_device", "auto")),
            collect_linf_diagnostics=(
                normalize_dual_feasibility_norm(dual_feasibility_norm) == "linf"
                or bool(getattr(config, "record_dual_linf_diagnostics", False))
            ),
        )
        diagnostics.update(scan.diagnostics)
        diagnostics.setdefault("edge_selection_mode", "nodewise")
        return LowrankDualFeasibilityScan(
            dual_feasibility=float(scan.dual_feasibility),
            dual_feasibility_source="early_torch_blockwise_bidir_scan",
            scan_time=float(time.perf_counter() - scan_start),
            diagnostics=diagnostics,
            rows=scan.rows,
            cols=scan.cols,
            peak_mem_mib=None,
        )

    mem_tracker = DriverMemoryTracker(device=0)

    if bool(use_fused):
        fused_topk = validate_fused_lowrank_bidir_scan_config(
            config,
            cost_type=str(getattr(config, "cost_type", "lowrank")),
        )
        with (
            trace_collector.span(
                f"{trace_prefix}.{fused_span_name}",
                "solve_ot",
                args=diagnostics,
            )
            if trace_collector is not None
            else nullcontext()
        ):
            mem_tracker.tick()
            fused_scan = raw_lowrank_bidir_feasibility_pricing_scan(
                lvl_s.points,
                lvl_t.points,
                lvl_s.cost_vec,
                lvl_t.cost_vec,
                dual_uv[: len(lvl_s.points)],
                dual_uv[len(lvl_s.points) :],
                topk=int(fused_topk),
                theta=0.0,
                gpu_id=0,
                memory_floor_mib=float(getattr(config, "fused_lowrank_scan_memory_floor_mib", 768.0)),
                resident_multiplier=float(getattr(config, "fused_lowrank_scan_resident_multiplier", 1.10)),
                collect_linf_diagnostics=(
                    normalize_dual_feasibility_norm(dual_feasibility_norm) == "linf"
                    or bool(getattr(config, "record_dual_linf_diagnostics", False))
                ),
                dot_scale=float(getattr(config, "dot_scale", 1.0)),
            )
            mem_tracker.tick()
        rows = fused_scan.rows
        cols = fused_scan.cols
        diagnostics.update(dict(fused_scan.diagnostics))
        diagnostics.setdefault("edge_selection_mode", "nodewise")
        return LowrankDualFeasibilityScan(
            dual_feasibility=float(fused_scan.dual_feasibility),
            dual_feasibility_source="early_fused_lowrank_bidir_scan",
            scan_time=float(time.perf_counter() - scan_start),
            diagnostics=diagnostics,
            rows=rows,
            cols=cols,
            peak_mem_mib=None if mem_tracker.peak_mib is None else float(mem_tracker.peak_mib),
        )
    else:
        if bool(getattr(config, "record_dual_linf_diagnostics", False)):
            raise ValueError(
                "record_dual_linf_diagnostics requires the fused lowrank feasibility-pricing scan."
            )
        with (
            trace_collector.span(
                f"{trace_prefix}.{standalone_span_name}",
                "solve_ot",
                args=diagnostics,
            )
            if trace_collector is not None
            else nullcontext()
        ):
            mem_tracker.tick()
            dual_feasibility = lowrank_dual_feasibility_infeasibility(
                lvl_s.points,
                lvl_t.points,
                lvl_s.cost_vec,
                lvl_t.cost_vec,
                dual_uv,
                diagnostics=diagnostics,
                dot_scale=float(getattr(config, "dot_scale", 1.0)),
            )
            mem_tracker.tick()
        return LowrankDualFeasibilityScan(
            dual_feasibility=float(dual_feasibility),
            dual_feasibility_source="early_standalone_dual_feasibility_scan",
            scan_time=float(time.perf_counter() - scan_start),
            diagnostics=diagnostics,
            peak_mem_mib=None if mem_tracker.peak_mib is None else float(mem_tracker.peak_mib),
        )
