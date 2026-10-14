from __future__ import annotations

import numpy as np
import numba
from numba import types


@numba.jit(types.Tuple((types.int32[:], types.int32[:]))(types.float64[:], types.float64[:]), nopython=True, cache=True)
def _northwest_corner_numba(p_in, q_in):
    """
    CN: 使用 Numba 加速西北角法，返回 row 和 column 下标。
    EN: Run the Numba-accelerated northwest-corner method and return row and column indices.
    """
    # CN: 拷贝边际以避免修改输入。
    # EN: Copy the marginals to avoid mutating the inputs.
    p = p_in.copy()
    q = q_in.copy()

    m = p.shape[0]
    n = q.shape[0]

    # CN: 预分配最大可能的边空间；实际通常需要 m + n - 1 条边。
    # EN: Preallocate the maximum edge space; the usual count is m + n - 1.
    max_edges = m + n + 1
    rows = np.empty(max_edges, dtype=np.int32)
    cols = np.empty(max_edges, dtype=np.int32)

    i = 0
    j = 0
    count = 0

    # CN: 用于判定质量耗尽的容差。
    # EN: Tolerance for deciding whether mass is exhausted.
    tol = 1e-20

    while i < m and j < n:
        # CN: 记录当前边。
        # EN: Record the current edge.
        rows[count] = i
        cols[count] = j
        count += 1

        val = min(p[i], q[j])
        p[i] -= val
        q[j] -= val

        # CN: 判断西北角遍历的移动方向。
        # EN: Choose the movement direction in the northwest-corner traversal.
        p_empty = p[i] < tol
        q_empty = q[j] < tol

        if p_empty and q_empty:
            # CN: 两侧同时耗尽时只向下移动，避免跳过节点而破坏支撑连通性。
            # EN: When both sides are exhausted, move down only to avoid skipping a node and breaking support connectivity.
            # CN: 下一轮仍会看到耗尽的 q[j]，再向右移动。
            # EN: The next iteration still sees the exhausted q[j] and then moves right.
            i += 1
            # CN: 不要同时移动 j，否则可能在矩阵支撑中形成断点。
            # EN: Do not advance j simultaneously, or the matrix support may become disconnected.
        elif p_empty:
            i += 1
        else:
            j += 1

    # CN: 只返回已填充的边。
    # EN: Return only the populated edges.
    return rows[:count], cols[:count]
