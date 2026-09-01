from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional


def trace_span(
    trace_collector,
    trace_prefix: str,
    suffix: str,
    *,
    args: Optional[Dict[str, Any]] = None,
):
    if trace_collector is None:
        return nullcontext()
    prefix = str(trace_prefix).strip()
    name = str(suffix).strip()
    full_name = f"{prefix}.{name}" if prefix else name
    return trace_collector.span(full_name, "solve_ot", args=dict(args or {}))
