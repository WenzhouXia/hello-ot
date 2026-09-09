"""CN: 从用户提供的官方 checkout 运行 HiRef。EN: Run HiRef from a user-provided official checkout."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = ROOT / "third_party" / "HiRef" / "src"
LEGACY_SRC = ROOT / "paper_experiments" / "baselines" / "linear_ot" / "hiref" / "src"


def run_hiref(
    source_x: np.ndarray,
    target_x: np.ndarray,
    *,
    hierarchy_depth: int,
    max_q: int,
    max_rank: int,
    base_rank: int,
    sq_euclidean: bool,
    seed: int,
    return_mapping: bool,
) -> dict[str, Any]:
    """
    CN: 调用官方 torch HiRef 源码并返回统一的 experiment payload。
    EN: Invoke the official torch HiRef sources and return a uniform experiment payload.
    """
    import torch

    source_directory = UPSTREAM_SRC if UPSTREAM_SRC.is_dir() else LEGACY_SRC
    if not source_directory.is_dir():
        raise ImportError("Clone https://github.com/raphael-group/HiRef into third_party/HiRef.")
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    import HR_OT
    import rank_annealing

    if not torch.cuda.is_available():
        raise RuntimeError("HiRef requires a CUDA GPU.")
    source = torch.from_numpy(np.asarray(source_x, dtype=np.float32, order="C")).cuda()
    target = torch.from_numpy(np.asarray(target_x, dtype=np.float32, order="C")).cuda()
    if source.shape[0] != target.shape[0]:
        raise ValueError("HiRef requires square balanced point clouds.")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    schedule = rank_annealing.optimal_rank_schedule(
        n=int(source.shape[0]),
        hierarchy_depth=int(hierarchy_depth),
        max_Q=int(max_q),
        max_rank=int(max_rank),
    )
    started = time.perf_counter()
    solver = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
        source,
        target,
        schedule,
        base_rank=int(base_rank),
        device=torch.device("cuda"),
        sq_Euclidean=bool(sq_euclidean),
    )
    assignments, _ = solver.run(return_as_coupling=False)
    distance = solver.compute_OT_cost()
    torch.cuda.synchronize()
    runtime = float(time.perf_counter() - started)
    rows = np.concatenate([_index_array(item[0]) for item in assignments])
    cols = np.concatenate([_index_array(item[1]) for item in assignments])
    mapping = np.column_stack([rows, cols]).astype(np.int64, copy=False)
    payload: dict[str, Any] = {
        "distance": float(distance.detach().cpu().item()),
        "runtime_sec": runtime,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
        "rank_schedule": [int(value) for value in schedule],
    }
    if return_mapping:
        payload["mapping"] = mapping
    return payload


def _index_array(value: Any) -> np.ndarray:
    """CN: 将 torch/numpy 索引转为 CPU int64。EN: Convert torch/numpy indices to CPU int64."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64).reshape(-1)
