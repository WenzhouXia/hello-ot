from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np
import torch
_INDEX_HASH_PRIMES = (2003, 2011)
_PERTURB_MODE_IDS = {"rank2": 0, "index_hash": 1}
MetricPerturbMode = Literal["rank2", "index_hash"]

def _full_metric_perturbation_enabled(perturbation: Optional[Dict[str, Any]]) -> bool:
    return (
        isinstance(perturbation, dict)
        and str(perturbation.get("mode", "")).strip().lower() == "full_cost_perturb"
        and float(perturbation.get("sigma", 0.0)) != 0.0
    )


@lru_cache(maxsize=32)
def _index_hash_parameters(seed: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    CN: 从种子生成两个模素数哈希的系数；取值范围保证 float32 的 KeOps 中间整数可精确表示。
    EN: Generate coefficients for two prime-modulus hashes while keeping KeOps float32 integer intermediates exact.
    """
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    return tuple(
        (
            int(rng.integers(1, prime)),
            int(rng.integers(1, prime)),
            int(rng.integers(0, prime)),
        )
        for prime in _INDEX_HASH_PRIMES
    )


def _metric_index_hash_noise_numpy(source_index: Any, target_index: Any, *, seed: int) -> np.ndarray:
    """
    CN: 使用与 Torch/CUDA 相同的 FP32 近似取模与运算顺序；不截断边界值。
    EN: Use the same FP32 approximate remainder and operation order as Torch/CUDA without clipping.
    """
    src = np.asarray(source_index, dtype=np.int64)
    tgt = np.asarray(target_index, dtype=np.int64)
    hashes = []
    for prime, (src_coeff, tgt_coeff, offset) in zip(_INDEX_HASH_PRIMES, _index_hash_parameters(seed)):
        src_residue = np.remainder(src, prime).astype(np.float32)
        tgt_residue = np.remainder(tgt, prime).astype(np.float32)
        value = ((np.float32(src_coeff) * src_residue + np.float32(tgt_coeff) * tgt_residue)
                 + src_residue * tgt_residue) + np.float32(offset)
        hashes.append(value - np.float32(prime) * np.floor(value * np.float32(1.0 / prime)))
    combined = hashes[0] + np.float32(_INDEX_HASH_PRIMES[0]) * hashes[1]
    modulus = _INDEX_HASH_PRIMES[0] * _INDEX_HASH_PRIMES[1]
    return (combined + np.float32(0.5)) * np.float32(1.0 / modulus)


def _metric_noise_numpy(source_index: Any, target_index: Any, *, seed: int) -> np.ndarray:
    src = np.asarray(source_index, dtype=np.float32)
    tgt = np.asarray(target_index, dtype=np.float32)
    z = src * np.float32(12.9898) + tgt * np.float32(78.233) + np.float32(int(seed) * 0.001)
    return (np.sin(z).astype(np.float32, copy=False) * np.float32(0.5) + np.float32(0.5)).astype(np.float32, copy=False)


def _metric_perturb_mode(perturbation: Dict[str, Any]) -> MetricPerturbMode:
    mode = str(perturbation.get("noise", "rank2")).strip().lower()
    if mode not in {"rank2", "index_hash"}:
        raise ValueError("metric perturbation noise must be one of: rank2, index_hash")
    return mode


def _apply_metric_pair_perturbation(
    costs: Any,
    rows: Any,
    cols: Any,
    perturbation: Optional[Dict[str, Any]],
) -> Any:
    if not _full_metric_perturbation_enabled(perturbation):
        return costs
    sigma = float(perturbation.get("sigma", 0.0))
    seed = int(perturbation.get("seed", 0))
    perturb_mode = _metric_perturb_mode(perturbation)
    source_index = perturbation.get("source_global_index")
    target_index = perturbation.get("target_global_index")
    if source_index is None or target_index is None:
        raise ValueError("full metric perturbation requires source_global_index and target_global_index.")
    if torch.is_tensor(costs):
        device = costs.device
        rows_t = rows.to(device=device, dtype=torch.long) if torch.is_tensor(rows) else torch.as_tensor(rows, dtype=torch.long, device=device)
        cols_t = cols.to(device=device, dtype=torch.long) if torch.is_tensor(cols) else torch.as_tensor(cols, dtype=torch.long, device=device)
        src_idx = torch.as_tensor(np.asarray(source_index, dtype=np.int64), dtype=torch.long, device=device).index_select(0, rows_t)
        tgt_idx = torch.as_tensor(np.asarray(target_index, dtype=np.int64), dtype=torch.long, device=device).index_select(0, cols_t)
        if perturb_mode == "rank2":
            noise = torch.sin(src_idx.float() * 12.9898 + tgt_idx.float() * 78.233 + float(seed) * 0.001) * 0.5 + 0.5
        else:
            hashes = []
            for prime, (src_coeff, tgt_coeff, offset) in zip(_INDEX_HASH_PRIMES, _index_hash_parameters(seed)):
                src_residue = torch.remainder(src_idx, prime).float()
                tgt_residue = torch.remainder(tgt_idx, prime).float()
                value = ((float(src_coeff) * src_residue + float(tgt_coeff) * tgt_residue)
                         + src_residue * tgt_residue) + float(offset)
                hashes.append(value - float(prime) * torch.floor(value * float(np.float32(1.0 / prime))))
            modulus = _INDEX_HASH_PRIMES[0] * _INDEX_HASH_PRIMES[1]
            noise = (hashes[0] + float(_INDEX_HASH_PRIMES[0]) * hashes[1]).add(0.5).mul(float(np.float32(1.0 / modulus)))
        return costs + noise.to(dtype=costs.dtype) * float(sigma)
    rows_np = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols_np = np.asarray(cols, dtype=np.int64).reshape(-1)
    src_idx = np.asarray(source_index, dtype=np.int64)[rows_np]
    tgt_idx = np.asarray(target_index, dtype=np.int64)[cols_np]
    noise = (
        _metric_noise_numpy(src_idx, tgt_idx, seed=int(seed))
        if perturb_mode == "rank2"
        else _metric_index_hash_noise_numpy(src_idx, tgt_idx, seed=int(seed))
    )
    return np.asarray(costs, dtype=np.float64) + noise.astype(np.float64, copy=False) * float(sigma)


def _fused_perturb_arguments(
    *,
    perturbation: Optional[Dict[str, Any]],
    query_global_index: Optional[np.ndarray],
    database_global_index: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[float, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CN: 把 metric 链的 perturbation dict 转成 fused kernel 的入参
        (sigma, seed, mode_id, query_global_index, database_global_index, coeffs)；
        无扰动时传 sigma=0 与空索引。
    EN: Convert a metric-chain perturbation dict into fused kernel arguments
        (sigma, seed, mode_id, query_global_index, database_global_index, coeffs);
        pass sigma=0 and empty indices when perturbation is disabled.
    """
    empty_idx = torch.empty(0, dtype=torch.int64, device=device)
    if not _full_metric_perturbation_enabled(perturbation):
        return 0.0, 0, 0, empty_idx, empty_idx, torch.zeros(6, dtype=torch.int64, device=device)
    sigma = float(perturbation.get("sigma", 0.0))
    seed = int(perturbation.get("seed", 0))
    mode = int(_PERTURB_MODE_IDS[_metric_perturb_mode(perturbation)])
    q_idx = None if query_global_index is None else np.asarray(query_global_index, dtype=np.int64)
    d_idx = None if database_global_index is None else np.asarray(database_global_index, dtype=np.int64)
    if q_idx is None or d_idx is None:
        raise ValueError("full metric perturbation requires row and column global indices for the fused kernel.")
    q_t = torch.as_tensor(q_idx, dtype=torch.int64, device=device).contiguous()
    d_t = torch.as_tensor(d_idx, dtype=torch.int64, device=device).contiguous()
    if mode == 1:
        coeffs = torch.as_tensor(
            np.asarray(sum(_index_hash_parameters(seed), ()), dtype=np.int64),
            dtype=torch.int64,
            device=device,
        ).contiguous()
    else:
        coeffs = torch.zeros(6, dtype=torch.int64, device=device)
    return sigma, seed, mode, q_t, d_t, coeffs
