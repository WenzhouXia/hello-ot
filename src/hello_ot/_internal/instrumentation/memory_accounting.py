from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


LOWRANK_MEMORY_ACCOUNTING_COVERAGE = [
    "lp_backend",
    "raw_lowrank_bidir_feasibility_pricing_scan",
]
METRIC_MEMORY_ACCOUNTING_COVERAGE = [
    "lp_backend",
    "raw_metric_bidir_feasibility_pricing_scan",
]

_CURRENT_MEMORY_RECORDER: ContextVar[Optional["MemoryAccountingRecorder"]] = ContextVar(
    "hello_ot_current_memory_accounting_recorder",
    default=None,
)

_CURRENT_SOLVE_MEMORY_TRACKER: ContextVar[Optional["SolveMemoryTracker"]] = ContextVar(
    "hello_ot_current_solve_memory_tracker",
    default=None,
)
_PROFILED_DEVICES_LOCK = threading.Lock()
_PROFILED_DEVICES: set[int] = set()


class SolveMemoryTracker:
    """
    CN: 整次 HELLO solve 的显存统计器；PyTorch peak 只重置一次，并与逐次 LP delta 组合。
    EN: Whole-solve HELLO memory tracker that resets the PyTorch peak once and combines it with per-LP deltas.
    """

    def __init__(self, device: Any) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("solve memory profiling requires PyTorch CUDA")
        parsed = torch.device(device)
        self.device_index = int(parsed.index if parsed.index is not None else torch.cuda.current_device())
        self.started = False
        self.entry_allocated_bytes: Optional[int] = None
        self.lp_candidates: List[Dict[str, Any]] = []
        self.planner_records: List[Dict[str, Any]] = []
        self._summary: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        if self.started:
            raise RuntimeError("solve memory tracker was already started")
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self.entry_allocated_bytes = int(torch.cuda.memory_allocated(self.device_index))
        self.started = True

    def begin_lp(self, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("solve memory tracker must be started before LP accounting")
        return {
            "torch_entry_bytes": int(torch.cuda.memory_allocated(self.device_index)),
            "metadata": dict(metadata or {}),
        }

    def finish_lp(
        self,
        entry: Dict[str, Any],
        *,
        delta_peak_mib: Any,
        source: str = "cupdlpx_loop_start_process_delta_approx",
    ) -> None:
        try:
            delta_bytes = max(0, int(float(delta_peak_mib) * float(1024 ** 2)))
        except (TypeError, ValueError):
            return
        torch_entry = int(entry.get("torch_entry_bytes", 0))
        self.lp_candidates.append(
            {
                "torch_entry_bytes": torch_entry,
                "lp_delta_bytes": delta_bytes,
                "combined_bytes": int(torch_entry + delta_bytes),
                "lp_delta_source": str(source),
                "metadata": dict(entry.get("metadata") or {}),
            }
        )

    def record_plan(self, values: Dict[str, Any]) -> None:
        self.planner_records.append(dict(values))

    def finish(self) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("solve memory tracker was not started")
        torch_peak = int(torch.cuda.max_memory_allocated(self.device_index))
        largest_lp = max(
            self.lp_candidates,
            key=lambda item: int(item["combined_bytes"]),
            default=None,
        )
        lp_peak = 0 if largest_lp is None else int(largest_lp["combined_bytes"])
        combined = max(int(torch_peak), int(lp_peak))
        source = "pytorch" if int(torch_peak) >= int(lp_peak) else "lp_combined"
        self._summary = {
            "peak_gpu_memory_mib": float(combined) / float(1024 ** 2),
            "peak_source": source,
            "torch_peak_allocated_mib": float(torch_peak) / float(1024 ** 2),
            "entry_torch_allocated_mib": (
                None
                if self.entry_allocated_bytes is None
                else float(self.entry_allocated_bytes) / float(1024 ** 2)
            ),
            "largest_lp_candidate": largest_lp,
            "lp_candidates": list(self.lp_candidates),
            "planner_records": list(self.planner_records),
        }
        self.started = False
        return dict(self._summary)


def current_solve_memory_tracker() -> Optional[SolveMemoryTracker]:
    return _CURRENT_SOLVE_MEMORY_TRACKER.get()


@contextmanager
def use_solve_memory_tracker(device: Any) -> Iterator[SolveMemoryTracker]:
    """
    CN: 安装单设备 solve tracker；拒绝同进程同设备的重叠 profiling。
    EN: Install a per-device solve tracker and reject overlapping profiling on the same device in one process.
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("solve memory profiling requires PyTorch CUDA")
    parsed = torch.device(device)
    device_index = int(parsed.index if parsed.index is not None else torch.cuda.current_device())
    with _PROFILED_DEVICES_LOCK:
        if device_index in _PROFILED_DEVICES:
            raise RuntimeError(
                f"another solve in this Python process is already profiling CUDA device {device_index}"
            )
        _PROFILED_DEVICES.add(device_index)
    tracker = SolveMemoryTracker(torch.device("cuda", device_index))
    token = _CURRENT_SOLVE_MEMORY_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_SOLVE_MEMORY_TRACKER.reset(token)
        with _PROFILED_DEVICES_LOCK:
            _PROFILED_DEVICES.discard(device_index)


def current_memory_recorder() -> Optional["MemoryAccountingRecorder"]:
    """
    CN: 返回当前上下文中的组件级显存 recorder；关闭时返回 None。
    EN: Return the component memory recorder in the current context; return None when disabled.
    """
    recorder = _CURRENT_MEMORY_RECORDER.get()
    if recorder is None or not recorder.enabled:
        return None
    return recorder


@contextmanager
def use_memory_recorder(recorder: Optional["MemoryAccountingRecorder"]) -> Iterator[None]:
    """
    CN: 在当前执行上下文中安装显存 recorder。
    EN: Install a memory recorder in the current execution context.
    """
    token: Optional[Token[Optional["MemoryAccountingRecorder"]]] = None
    if recorder is not None and recorder.enabled:
        token = _CURRENT_MEMORY_RECORDER.set(recorder)
    try:
        yield
    finally:
        if token is not None:
            _CURRENT_MEMORY_RECORDER.reset(token)


def make_memory_recorder_from_config(config: Dict[str, Any]) -> Optional["MemoryAccountingRecorder"]:
    """
    CN: 从 profiling.memory 配置构造 recorder；关闭时返回 None，避免运行期 CUDA/IO 开销。
    EN: Build a recorder from profiling.memory config; return None when disabled to avoid CUDA/IO overhead.
    """
    if not bool(config.get("enabled", False)):
        return None
    return MemoryAccountingRecorder(config)


class MemoryComponentSpan:
    """
    CN: 单个组件的显存统计上下文，支持组件主动上报内部 delta peak。
    EN: Memory accounting context for one component, with optional component-reported delta peak.
    """

    def __init__(
        self,
        recorder: "MemoryAccountingRecorder",
        *,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._recorder = recorder
        self.component = str(component)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.reported_delta_peak_mib: Optional[float] = None
        self.reported_delta_peak_source: Optional[str] = None
        self._entry: Optional[Dict[str, Optional[float]]] = None
        self._seq: Optional[int] = None
        self._start_time = 0.0
        self._start_elapsed_ms = 0.0

    def __enter__(self) -> "MemoryComponentSpan":
        self._seq = self._recorder.next_seq()
        self._start_time = time.perf_counter()
        self._start_elapsed_ms = float(self._start_time - self._recorder.start_time) * 1000.0
        self._entry = self._recorder.sample_cuda(reset_peak=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._recorder.finish_component(self, failed=exc_type is not None)

    def report_delta_peak_mib(self, value: Any, *, source: str) -> None:
        """
        CN: 上报组件内部更可靠的 delta peak，例如 LP backend 自己统计的峰值。
        EN: Report a more reliable component-internal delta peak, such as LP backend peak tracking.
        """
        try:
            parsed = None if value is None else float(value)
        except (TypeError, ValueError):
            parsed = None
        if parsed is None:
            return
        self.reported_delta_peak_mib = max(0.0, float(parsed))
        self.reported_delta_peak_source = str(source)

    def update_metadata(self, values: Optional[Dict[str, Any]]) -> None:
        """
        CN: 在组件执行后补充 metadata，例如返回 diagnostics 中的 tile 信息。
        EN: Add metadata after component execution, for example tile diagnostics from the result.
        """
        if values:
            self.metadata.update(dict(values))


class MemoryAccountingRecorder:
    """
    CN: 轻量组件级显存统计器，主口径使用 PyTorch allocated delta peak。
    EN: Lightweight component memory recorder using PyTorch allocated delta peak as the main metric.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.mode = str(config.get("mode", "light")).strip().lower()
        raw_jsonl_path = config.get("jsonl_path")
        self.jsonl_path = None if raw_jsonl_path is None else str(raw_jsonl_path)
        self.component_spans: List[Dict[str, Any]] = []
        self.coverage = list(LOWRANK_MEMORY_ACCOUNTING_COVERAGE)
        self.start_time = time.perf_counter()
        self._seq = 0
        self.algorithm_entry_sample = self.sample_cuda(reset_peak=False)
        self.algorithm_entry_torch_alloc_mib = self.algorithm_entry_sample.get("torch_alloc_mib")

    def next_seq(self) -> int:
        seq = int(self._seq)
        self._seq += 1
        return seq

    def sample_cuda(self, *, reset_peak: bool = False) -> Dict[str, Optional[float]]:
        torch_alloc_mib = None
        torch_peak_alloc_mib = None
        if torch is not None and torch.cuda.is_available():
            device = torch.cuda.current_device()
            if self.mode == "sync":
                torch.cuda.synchronize(device)
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
            torch_alloc_mib = float(torch.cuda.memory_allocated(device)) / float(1024 ** 2)
            torch_peak_alloc_mib = float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 2)
        return {
            "torch_alloc_mib": torch_alloc_mib,
            "torch_peak_alloc_mib": torch_peak_alloc_mib,
        }

    def component(self, component: str, *, metadata: Optional[Dict[str, Any]] = None) -> MemoryComponentSpan:
        return MemoryComponentSpan(self, component=component, metadata=metadata)

    def set_coverage(self, components: List[str]) -> None:
        """
        CN: 设置当前 cost backend 完整统计所需的组件集合。
        EN: Set the component set required for complete accounting of the current cost backend.
        """
        self.coverage = [str(component) for component in components]

    def _coverage_complete(self) -> bool:
        observed = {str(item.get("component", "")) for item in self.component_spans}
        return bool(self.coverage) and set(self.coverage).issubset(observed)

    def record(self, **_: Any) -> None:
        """
        CN: 兼容旧阶段事件调用；组件级系统不再记录 point event。
        EN: Compatibility shim for old phase events; component accounting no longer records point events.
        """
        return None

    def begin_span(self, **_: Any) -> None:
        """
        CN: 兼容旧粗粒度 span 调用；不 reset peak，避免污染组件级统计。
        EN: Compatibility shim for old coarse spans; do not reset peaks to avoid polluting component accounting.
        """
        return None

    def end_span(self, *_: Any, **__: Any) -> None:
        """
        CN: 兼容旧粗粒度 span 调用；不产生输出。
        EN: Compatibility shim for old coarse spans; do not emit output.
        """
        return None

    def finish_component(self, span: MemoryComponentSpan, *, failed: bool) -> None:
        entry = dict(span._entry or {})
        sample = self.sample_cuda(reset_peak=False)
        entry_alloc = entry.get("torch_alloc_mib")
        peak_alloc = sample.get("torch_peak_alloc_mib")
        exit_alloc = sample.get("torch_alloc_mib")
        algorithm_entry_alloc = self.algorithm_entry_torch_alloc_mib
        delta_peak_torch = None if entry_alloc is None or peak_alloc is None else max(0.0, float(peak_alloc) - float(entry_alloc))
        exit_delta_torch = None if entry_alloc is None or exit_alloc is None else float(exit_alloc) - float(entry_alloc)
        live_entry_torch = (
            None
            if entry_alloc is None or algorithm_entry_alloc is None
            else max(0.0, float(entry_alloc) - float(algorithm_entry_alloc))
        )
        effective_delta_peak = (
            span.reported_delta_peak_mib if span.reported_delta_peak_mib is not None else delta_peak_torch
        )
        effective_peak = (
            None
            if live_entry_torch is None or effective_delta_peak is None
            else float(live_entry_torch) + float(effective_delta_peak)
        )
        payload = {
            "seq": span._seq,
            "component": str(span.component),
            "metadata": dict(span.metadata),
            "algorithm_entry_torch_alloc_mib": algorithm_entry_alloc,
            "entry_torch_alloc_mib": entry_alloc,
            "peak_torch_alloc_mib": peak_alloc,
            "exit_torch_alloc_mib": exit_alloc,
            "live_entry_torch_mib": live_entry_torch,
            "delta_peak_torch_mib": delta_peak_torch,
            "exit_delta_torch_mib": exit_delta_torch,
            "reported_delta_peak_mib": span.reported_delta_peak_mib,
            "reported_delta_peak_source": span.reported_delta_peak_source,
            "effective_delta_peak_mib": effective_delta_peak,
            "effective_peak_mib": effective_peak,
            "duration_ms": float(time.perf_counter() - span._start_time) * 1000.0,
            "start_elapsed_ms": span._start_elapsed_ms,
            "end_elapsed_ms": float(time.perf_counter() - self.start_time) * 1000.0,
            "failed": bool(failed),
        }
        if str(span.metadata.get("memory_accounting_entry", "")).strip().lower() == "pre_native_solve":
            payload.update(
                {
                    "pre_native_entry_torch_alloc_mib": entry_alloc,
                    "pre_native_live_entry_torch_mib": live_entry_torch,
                    "pre_native_delta_peak_mib": effective_delta_peak,
                    "pre_native_effective_peak_mib": effective_peak,
                }
            )
        self.component_spans.append(payload)
        self._append_jsonl(payload)

    def summary(self) -> Dict[str, Any]:
        effective_candidates = [
            item for item in self.component_spans if _summary_effective_peak(item) is not None
        ]
        delta_candidates = [
            item for item in self.component_spans if item.get("effective_delta_peak_mib") is not None
        ]
        torch_delta_candidates = [
            float(item["delta_peak_torch_mib"])
            for item in self.component_spans
            if item.get("delta_peak_torch_mib") is not None
        ]
        pre_native_candidates = [
            item for item in self.component_spans if item.get("pre_native_effective_peak_mib") is not None
        ]
        largest_effective = (
            max(effective_candidates, key=lambda item: float(_summary_effective_peak(item) or 0.0))
            if effective_candidates
            else None
        )
        largest_delta = (
            max(delta_candidates, key=lambda item: float(item["effective_delta_peak_mib"]))
            if delta_candidates
            else None
        )
        largest_pre_native = (
            max(pre_native_candidates, key=lambda item: float(item["pre_native_effective_peak_mib"]))
            if pre_native_candidates
            else None
        )
        return {
            "algorithm_effective_peak_mib": (
                None
                if largest_effective is None
                else float(_summary_effective_peak(largest_effective) or 0.0)
            ),
            "algorithm_effective_peak_source": _summary_effective_peak_source(largest_effective),
            "algorithm_peak_torch_delta_mib": max(torch_delta_candidates) if torch_delta_candidates else None,
            "algorithm_pre_native_effective_peak_mib": (
                None
                if largest_pre_native is None
                else float(largest_pre_native["pre_native_effective_peak_mib"])
            ),
            "largest_effective_peak_component": _summary_component(largest_effective, peak_key="effective_peak_mib"),
            "largest_delta_peak_component": _summary_component(largest_delta, peak_key="effective_delta_peak_mib"),
            "largest_pre_native_effective_peak_component": _summary_component(
                largest_pre_native,
                peak_key="pre_native_effective_peak_mib",
            ),
            "component_span_count": int(len(self.component_spans)),
            "coverage": list(self.coverage),
            "coverage_complete": self._coverage_complete(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "component_spans": list(self.component_spans),
            "summary": self.summary(),
            "jsonl_path": self.jsonl_path,
            "coverage": list(self.coverage),
            "coverage_complete": self._coverage_complete(),
        }

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        parent = os.path.dirname(os.path.abspath(self.jsonl_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _summary_component(item: Optional[Dict[str, Any]], *, peak_key: str) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "component": item.get("component"),
        peak_key: item.get(peak_key),
        "summary_effective_peak_mib": _summary_effective_peak(item),
        "summary_effective_peak_source": _summary_effective_peak_source(item),
        "effective_peak_mib": item.get("effective_peak_mib"),
        "effective_delta_peak_mib": item.get("effective_delta_peak_mib"),
        "pre_native_effective_peak_mib": item.get("pre_native_effective_peak_mib"),
        "pre_native_delta_peak_mib": item.get("pre_native_delta_peak_mib"),
        "pre_native_entry_torch_alloc_mib": item.get("pre_native_entry_torch_alloc_mib"),
        "delta_peak_torch_mib": item.get("delta_peak_torch_mib"),
        "reported_delta_peak_mib": item.get("reported_delta_peak_mib"),
        "reported_delta_peak_source": item.get("reported_delta_peak_source"),
        "duration_ms": item.get("duration_ms"),
        "metadata": metadata,
    }


def _summary_effective_peak(item: Optional[Dict[str, Any]]) -> Optional[float]:
    if item is None:
        return None
    value = item.get("pre_native_effective_peak_mib")
    if value is None:
        value = item.get("effective_peak_mib")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _summary_effective_peak_source(item: Optional[Dict[str, Any]]) -> Optional[str]:
    if item is None:
        return None
    if item.get("pre_native_effective_peak_mib") is not None:
        return "pre_native_effective_peak_mib"
    if item.get("effective_peak_mib") is not None:
        return "effective_peak_mib"
    return None
