from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def edge_keys(rows: Any, cols: Any, n_target: int) -> Any:
    """
    CN: 在数组当前设备上编码边，不复制整个 active support。
    EN: Encode edges on their current device without copying the whole active support.
    """
    if torch.is_tensor(rows):
        return rows.to(torch.int64) * int(n_target) + cols.to(torch.int64)
    return np.asarray(rows, dtype=np.int64) * int(n_target) + np.asarray(cols, dtype=np.int64)


@dataclass
class ReentryDetector:
    """
    CN: 严格上一轮删除边回流检测，给予一次 LP 观察期。
    EN: Detect re-entry of edges pruned in the previous round with one LP probation.
    """

    previous_pruned: Any = None
    reentry_count: int = 0

    def should_trigger(self) -> bool:
        """
        CN: 由主循环在下一次 LP 的收敛检查失败后调用。
        EN: Called by the main loop after the next LP fails its convergence check.
        """
        return self.reentry_count > 0

    def observe(self, update: Any) -> None:
        """
        CN: 先检查新增边，再替换上一轮删除记录，绝不累计历史。
        EN: Check added edges before replacing the previous prune record; never accumulate history.
        """
        added, previous = update.added_keys, self.previous_pruned
        self.reentry_count = 0
        if added is not None and previous is not None:
            if torch.is_tensor(added) or torch.is_tensor(previous):
                device = added.device if torch.is_tensor(added) else previous.device
                added = torch.as_tensor(added, device=device, dtype=torch.int64)
                previous = torch.as_tensor(previous, device=device, dtype=torch.int64)
                self.reentry_count = int(torch.isin(added, previous).sum().item())
            else:
                self.reentry_count = int(np.isin(added, previous).sum())
        self.previous_pruned = update.pruned_keys
