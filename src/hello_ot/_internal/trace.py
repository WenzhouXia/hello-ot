from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _ChromeTraceCollector:
    enabled: bool = True
    pid: int = 1
    tid: int = 1
    start_time: float = field(default_factory=time.perf_counter)
    events: List[Dict[str, Any]] = field(default_factory=list)
    name_prefix: str = ""

    def child(
        self,
        *,
        name_prefix: str = "",
        tid: Optional[int] = None,
    ) -> "_ChromeTraceCollector":
        child_prefix = str(name_prefix).strip()
        if self.name_prefix and child_prefix:
            combined_prefix = f"{self.name_prefix}.{child_prefix}"
        elif self.name_prefix:
            combined_prefix = str(self.name_prefix)
        else:
            combined_prefix = child_prefix
        return _ChromeTraceCollector(
            enabled=bool(self.enabled),
            pid=int(self.pid),
            tid=int(self.tid),
            start_time=float(self.start_time),
            events=self.events,
            name_prefix=combined_prefix,
        )

    @contextmanager
    def span(
        self,
        name: str,
        category: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        tid: Optional[int] = None,
    ) -> Any:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            event_name = str(name)
            if self.name_prefix:
                event_name = f"{self.name_prefix}.{event_name}"
            self.events.append(
                {
                    "name": event_name,
                    "cat": str(category),
                    "ph": "X",
                    "pid": int(self.pid),
                    "tid": int(self.tid),
                    "ts": float((t0 - self.start_time) * 1e6),
                    "dur": float(max(t1 - t0, 0.0) * 1e6),
                    "args": dict(args or {}),
                }
            )

    def export(self) -> Dict[str, Any]:
        return {
            "traceEvents": list(self.events),
            "displayTimeUnit": "ms",
        }


def _trace_device_synchronize(tracer: Optional[_ChromeTraceCollector]) -> None:
    if tracer is None or not bool(tracer.enabled):
        return
    try:
        import cupy as cp  # type: ignore

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
