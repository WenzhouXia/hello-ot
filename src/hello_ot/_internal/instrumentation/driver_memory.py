from __future__ import annotations

from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class DriverMemoryTracker:
    """
    CN: 用 CUDA driver 口径采样整卡 used memory peak；默认仅在 memory diagnostics recorder 开启时采样。
    EN: Sample device-wide used memory peak from the CUDA driver; by default only sample when the memory diagnostics recorder is active.
    """

    def __init__(self, device: int = 0, *, enabled: Optional[bool] = None) -> None:
        self.device = int(device)
        if enabled is None:
            try:
                from .memory_accounting import current_memory_recorder

                enabled = current_memory_recorder() is not None
            except Exception:
                enabled = False
        self.enabled = bool(enabled and torch is not None and torch.cuda.is_available())
        self._last_used_bytes: Optional[int] = None
        self._peak_used_bytes: Optional[int] = None
        self.tick()

    def tick(self) -> Optional[float]:
        """
        CN: 采样当前整卡 used memory，并返回截至目前的 peak MiB。
        EN: Sample current device-wide used memory and return the peak MiB so far.
        """
        if not self.enabled or torch is None:
            return None
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(int(self.device))
        except Exception:
            self.enabled = False
            return None
        used = max(0, int(total_mem) - int(free_mem))
        self._last_used_bytes = int(used)
        if self._peak_used_bytes is None:
            self._peak_used_bytes = int(used)
        else:
            self._peak_used_bytes = max(int(self._peak_used_bytes), int(used))
        return self.peak_mib

    @property
    def current_mib(self) -> Optional[float]:
        """
        CN: 返回最近一次 tick 采样到的整卡 used memory，单位 MiB。
        EN: Return the device-wide used memory from the latest tick in MiB.
        """
        if self._last_used_bytes is None:
            return None
        return float(self._last_used_bytes) / float(1024 * 1024)

    @property
    def peak_mib(self) -> Optional[float]:
        """
        CN: 返回已采样到的整卡 used memory peak，单位 MiB。
        EN: Return the sampled device-wide used memory peak in MiB.
        """
        if self._peak_used_bytes is None:
            return None
        return float(self._peak_used_bytes) / float(1024 * 1024)
