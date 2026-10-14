"""
CN: refinement 使用的对偶稳定化数值策略。
EN: Numerical dual-stabilization policies used by refinement.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

def _normalize_ot_dual_gauge(
    dual: np.ndarray,
    *,
    n_source: int,
) -> tuple[np.ndarray, float]:
    dual_out = np.asarray(dual, dtype=np.float64).reshape(-1).copy()
    source_count = int(n_source)
    if source_count <= 0 or source_count >= int(dual_out.size):
        raise ValueError("n_source must split a non-empty OT dual into source and target parts.")
    shift = float(np.mean(dual_out[:source_count].astype(np.float64, copy=False)))
    dual_out[:source_count] -= shift
    dual_out[source_count:] += shift
    return dual_out, shift


def _dual_stabilization_pricing_dual(
    dual_new: np.ndarray,
    previous_stable: Optional[np.ndarray],
    *,
    n_source: int,
    mode: str,
    alpha: float,
    gauge_normalization: bool = False,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    CN: 构造 pricing 使用的可选 EMA-stabilized dual，不修改 LP 返回的 raw dual。
    EN: Build the optional EMA-stabilized pricing dual without modifying the raw LP dual.
    """
    dual_raw = np.asarray(dual_new, dtype=np.float64).reshape(-1)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode != "ema":
        return dual_raw, dual_raw.copy(), {
            "mode": "off",
            "alpha": float(alpha),
            "enabled": False,
            "applied": False,
            "reset": False,
            "delta_rel": 0.0,
            "gauge_normalization": False,
            "gauge_shift": 0.0,
        }

    gauge_enabled = bool(gauge_normalization)
    gauge_shift = 0.0
    if gauge_enabled:
        dual_raw, gauge_shift = _normalize_ot_dual_gauge(
            dual_raw,
            n_source=int(n_source),
        )
    previous_raw = (
        None
        if previous_stable is None
        else np.asarray(previous_stable, dtype=np.float64).reshape(-1)
    )
    reset = bool(previous_raw is None or previous_raw.shape != dual_raw.shape)
    previous = (
        None
        if reset or previous_raw is None
        else (
            _normalize_ot_dual_gauge(previous_raw, n_source=int(n_source))[0]
            if gauge_enabled
            else previous_raw
        )
    )
    applied = bool(not reset and float(alpha) > 0.0)
    pricing_dual = (
        ((1.0 - float(alpha)) * dual_raw + float(alpha) * previous).astype(
            np.float64,
            copy=False,
        )
        if applied
        else dual_raw
    )
    delta = np.asarray(pricing_dual, dtype=np.float64).reshape(-1) - dual_raw
    denominator = max(
        float(np.linalg.norm(dual_raw.astype(np.float64, copy=False))),
        1e-12,
    )
    return (
        np.asarray(pricing_dual, dtype=np.float64).copy(),
        np.asarray(pricing_dual, dtype=np.float64).copy(),
        {
            "mode": "ema",
            "alpha": float(alpha),
            "enabled": True,
            "applied": applied,
            "reset": reset,
            "delta_rel": float(np.linalg.norm(delta) / denominator),
            "gauge_normalization": gauge_enabled,
            "gauge_shift": float(gauge_shift),
        },
    )
