"""
CN: 普通维度共用的扫描分块规划；只按工作量、布局和空间预算选择。
EN: Shared scan tiling for ordinary dimensions, based only on workload, layout and workspace.
"""
from __future__ import annotations
import math
import threading
import time
from typing import Any, Dict, List, Tuple


class WorkspaceBudgetCache:
    """
    CN: 短期复用设备预算，自有活动分配增长立即扣除，过期或OOM后重新采样。
    EN: Briefly reuse device budgets, deduct growth in owned active allocations immediately, and resample on expiry or OOM.
    """
    def __init__(self, refresh_seconds=0.1):
        """
        CN: 刷新周期是所有cost共用的设备采样策略，不是按实例调参。
        EN: The refresh period is shared device sampling policy, not per-instance tuning.
        """
        self.refresh_seconds=refresh_seconds
        self._entries={}
        self._lock=threading.RLock()

    def get(self, device, allocated, query, *, now=None):
        """
        CN: 分配减少时不提前上调预算，避免把未重新采样的显存当作新增余量。
        EN: Do not raise budgets early when allocations shrink, avoiding unsampled additional headroom.
        """
        now=time.monotonic() if now is None else now
        with self._lock:
            entry=self._entries.get(device)
            if entry is not None and now-entry[0]<self.refresh_seconds:
                reusable=entry[1]-max(0,allocated-entry[2])
                if reusable>0:return min(1024*2**20,reusable//4)
            reusable=int(query())
            self._entries[device]=(now,reusable,allocated)
            return min(1024*2**20,max(0,reusable)//4)

    def invalidate(self, device):
        """
        CN: OOM后丢弃旧快照，下一次必须查询当前设备显存。
        EN: Discard stale snapshots after OOM so the next request queries current device memory.
        """
        with self._lock:self._entries.pop(device,None)


def scan_workspace_bytes(q, n, k, *, pricing, matrix_bytes, copy_x_bytes=0, copy_y_bytes=0, target_blocks=256):
    """
    CN: 估算成本、块内候选、增量合并和必要坐标副本的同时存活空间。
    EN: Estimate simultaneously live costs, partial candidates, incremental merges and required feature copies.
    """
    matrix = matrix_bytes > 0
    rows = 16 if k <= 2 or (matrix and k <= 8) else (8 if matrix else 4)
    columns = 64 if matrix and 2 < k <= 8 else 128
    wanted = max(1, (target_blocks + (q + rows - 1)//rows - 1)//((q + rows - 1)//rows))
    db_tile = max(1024 if k <= 2 else columns, ((n+wanted-1)//wanted+columns-1)//columns*columns)
    splits = (n+db_tile-1)//db_tile
    per_pair = matrix_bytes + (16*min(k,rows)/rows if pricing and not (matrix and k<=8) else 0)
    return int((q+rows)*n*per_pair+(q+n)*(k*96+32)+(splits-1)*q*k*16+q*copy_x_bytes+n*copy_y_bytes+2**20)


def select_scan_tile(q, n, k, *, workspace_bytes, pricing, matrix_bytes, copy_x_bytes=0, copy_y_bytes=0,
                     target_blocks=256, tile_limits=None):
    """
    CN: 在同一空间模型下优先宽数据库块，再选查询块；不按普通维度或数据集调参。
    EN: Prefer wide database tiles then query tiles under one memory model, without ordinary-dimension or dataset tuning.
    """
    if min(q,n,workspace_bytes) <= 0:
        raise ValueError('scan sizes and workspace must be positive')
    qmax,nmax = tile_limits or (8192,65536)
    qmax,nmax = min(q,qmax),min(n,nmax)
    def candidates(limit):
        """
        CN: 纳入完整尾块与通用二进制档位，极小预算也有合法候选。
        EN: Include full tails and generic binary buckets, retaining candidates for tiny budgets.
        """
        return sorted({limit,*(1<<b for b in range(limit.bit_length()) if 1<<b < limit)},reverse=True)
    for nt in candidates(nmax):
        for qt in candidates(qmax):
            size = scan_workspace_bytes(qt,nt,k,pricing=pricing,matrix_bytes=matrix_bytes,
                copy_x_bytes=copy_x_bytes,copy_y_bytes=copy_y_bytes,target_blocks=target_blocks)
            if size <= workspace_bytes:
                return qt,nt
    raise RuntimeError('generic scan workspace cannot fit one tile')

def streamed_memory_estimate_mib(
    *,
    n_stream: int,
    n_resident: int,
    d: int,
    topk: int,
    stream_chunk: int,
    query_tile: int,
    db_tile: int,
    feature_itemsize: int = 4,
) -> Dict[str, float]:
    """
    CN: 估算流式上传、常驻数组、点积工作区及输出缓冲的共同预算。
    EN: Estimate a joint budget for streamed uploads, resident arrays, dot workspace and output buffers.
    """
    stream_chunk_eff = max(1, min(int(stream_chunk), int(n_stream)))
    query_tile_eff = max(1, min(int(query_tile), stream_chunk_eff))
    db_tile_eff = max(1, min(int(db_tile), int(n_resident)))
    mib = float(1024 * 1024)
    resident_mib = float(n_resident) * (float(d * feature_itemsize) + 2.0 * 8.0) / mib
    stream_chunk_mib = float(stream_chunk_eff) * (float(d * feature_itemsize) + 2.0 * 8.0) / mib
    score_tile_mib = float(query_tile_eff) * float(db_tile_eff) * float(feature_itemsize) / mib
    partial_resident_mib = float(math.ceil(float(stream_chunk_eff) / float(query_tile_eff))) * float(n_resident) * float(topk) * 8.0 / mib
    final_outputs_mib = float(n_resident + stream_chunk_eff) * float(topk) * 8.0 / mib
    est_peak_mib = resident_mib + stream_chunk_mib + score_tile_mib + partial_resident_mib + final_outputs_mib
    return {
        "resident_feature_mib": float(resident_mib),
        "stream_chunk_mib": float(stream_chunk_mib),
        "score_tile_mib": float(score_tile_mib),
        "partial_resident_mib": float(partial_resident_mib),
        "final_outputs_mib": float(final_outputs_mib),
        "est_peak_mib": float(est_peak_mib),
    }


def select_streamed_tiles(
    *,
    n_stream: int,
    n_resident: int,
    d: int,
    topk: int,
    memory_floor_mib: float,
    resident_multiplier: float,
    max_budget_mib: float,
    feature_itemsize: int = 4,
) -> Dict[str, Any]:
    """
    CN: 流式入口一次性选择上传批次及扫描块，沿用统一预算和排序规则。
    EN: Select upload batches and scan tiles once for streamed inputs, retaining shared budgets and ranking.
    """
    resident_mib = streamed_memory_estimate_mib(
        n_stream=int(n_stream),
        n_resident=int(n_resident),
        d=int(d),
        topk=int(topk),
        stream_chunk=1,
        query_tile=1,
        db_tile=1,
        feature_itemsize=int(feature_itemsize),
    )["resident_feature_mib"]
    target_budget_mib = min(
        float(max_budget_mib),
        max(float(memory_floor_mib), float(resident_multiplier) * float(resident_mib)),
    )
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()
    for query_candidate in (1024, 2048, 4096, 8192):
        for stream_mult in (1, 2):
            stream_chunk = min(int(n_stream), int(query_candidate) * int(stream_mult))
            query_tile = min(int(query_candidate), int(stream_chunk))
            if stream_chunk <= 0 or query_tile <= 0:
                continue
            for db_candidate in (8192, 16384, 32768, 65536):
                db_tile = min(int(db_candidate), int(n_resident))
                key = (int(stream_chunk), int(query_tile), int(db_tile))
                if key in seen:
                    continue
                seen.add(key)
                mem = streamed_memory_estimate_mib(
                    n_stream=int(n_stream),
                    n_resident=int(n_resident),
                    d=int(d),
                    topk=int(topk),
                    stream_chunk=int(stream_chunk),
                    query_tile=int(query_tile),
                    db_tile=int(db_tile),
                    feature_itemsize=int(feature_itemsize),
                )
                # CN: 新执行器的候选/合并空间在外层一起计入，避免执行层再次拆块。
                # EN: Budget new candidate and merge scratch jointly here, avoiding retiling in the executor.
                scratch=scan_workspace_bytes(query_tile,db_tile,topk,pricing=True,matrix_bytes=feature_itemsize)/2**20
                mem['scan_workspace_mib']=scratch
                mem['est_peak_mib']=mem['resident_feature_mib']+mem['stream_chunk_mib']+2*mem['final_outputs_mib']+scratch
                row: Dict[str, Any] = {
                    "source_chunk": int(stream_chunk),
                    "query_tile": int(query_tile),
                    "db_tile": int(db_tile),
                    "target_budget_mib": float(target_budget_mib),
                    "fits_soft_budget": bool(mem["est_peak_mib"] <= target_budget_mib),
                }
                row.update(mem)
                candidates.append(row)
    preferred = [row for row in candidates if bool(row["fits_soft_budget"])]
    if not preferred:
        minimum_peak = min(float(row["est_peak_mib"]) for row in candidates)
        raise RuntimeError(
            "fused inner-product refinement exceeds single-GPU scratch capacity: "
            f"minimum estimated peak is {minimum_peak:.1f} MiB, budget is {target_budget_mib:.1f} MiB."
        )
    pool = preferred

    def _rank(row: Dict[str, Any]) -> Tuple[float, int, int, int]:
        """
        CN: 优先避免跨查询块合并，再选较宽数据库块与较大查询块。
        EN: Prefer avoiding query-tile merges, then wider database tiles and larger query tiles.
        """
        source_bonus = 1 if int(row["source_chunk"]) == int(row["query_tile"]) else 0
        return (float(source_bonus), int(row["db_tile"]), int(row["query_tile"]), -int(row["source_chunk"]))

    if not pool:
        raise RuntimeError("no fused lowrank scan tile candidates were generated")
    selected = dict(max(pool, key=_rank))
    selected["candidate_count"] = int(len(candidates))
    selected["preferred_candidate_count"] = int(len(preferred))
    return selected
