from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import numpy as np


@dataclass
class Cost:
    """
    CN: GPU 点云及独立扰动列；精度转换和 pair gathers 始终分块。
    EN: GPU point clouds with separate perturbation columns and bounded casts and pair gathers.
    """

    source: torch.Tensor
    target: torch.Tensor
    kind: str
    source_offset: torch.Tensor
    target_offset: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor
    precision: str = "fp32"
    sigma: float = 0.0
    seed: int = 0
    source_extra: torch.Tensor | None = None
    target_extra: torch.Tensor | None = None
    workspace_bytes: int = 64 * 1024**2

    def sliced(self, n, m, precision):
        return replace(self, source=self.source[:n], target=self.target[:m],
                       source_offset=self.source_offset[:n], target_offset=self.target_offset[:m],
                       source_index=self.source_index[:n], target_index=self.target_index[:m],
                       source_extra=None if self.source_extra is None else self.source_extra[:n],
                       target_extra=None if self.target_extra is None else self.target_extra[:m],
                       precision=precision)

    @property
    def dtype(self):
        return torch.float64 if self.precision == "fp64" else torch.float32

    def noise(self, rows, cols):
        if self.sigma == 0:
            return 0.0
        if self.kind == "l2^2":
            return 0.5 * self.sigma - 2.0 * (self.source_extra[rows].to(self.dtype) * self.target_extra[cols].to(self.dtype)).sum(-1).double()
        # CN: 与 norm-cost 扫描共用 index-hash 定义。
        # EN: Share the index-hash definition with norm-cost scans.
        from hello_ot.costs.metric_perturbation import _INDEX_HASH_PRIMES, _index_hash_parameters
        hashes = []
        for prime, (a, b, offset) in zip(_INDEX_HASH_PRIMES, _index_hash_parameters(self.seed)):
            x = self.source_index[rows].remainder(prime).float()
            y = self.target_index[cols].remainder(prime).float()
            value = ((float(a) * x + float(b) * y) + x * y) + float(offset)
            hashes.append(value - float(prime) * torch.floor(value * float(np.float32(1.0 / prime))))
        modulus = _INDEX_HASH_PRIMES[0] * _INDEX_HASH_PRIMES[1]
        return ((hashes[0] + float(_INDEX_HASH_PRIMES[0]) * hashes[1] + 0.5) * float(np.float32(1.0 / modulus))).double() * self.sigma

    def pairs(self, rows, cols):
        output = torch.empty(rows.numel(), device=self.source.device, dtype=torch.float64)
        block = max(1, min(65536, self.workspace_bytes // max(1, self.source.shape[1] * 32)))
        for start in range(0, rows.numel(), block):
            rr, cc = rows[start:start + block].long(), cols[start:start + block].long()
            x, y = self.source[rr].to(self.dtype), self.target[cc].to(self.dtype)
            if self.kind == "l2^2":
                values = self.source_offset[rr] + self.target_offset[cc] - 2 * (x * y).sum(-1).double()
            else:
                diff = (x - y).abs()
                values = diff.sum(-1) if self.kind == "l1" else diff.amax(-1) if self.kind == "linf" else diff.square().sum(-1).sqrt()
            output[start:start + len(rr)] = values.double() + self.noise(rr, cc)
        return output

    def tile(self, i, j, q, b):
        x = self.source[i:i + q].to(self.dtype)
        y = self.target[j:j + b].to(self.dtype)
        if self.kind == "l2^2":
            values = self.source_offset[i:i + q, None] + self.target_offset[None, j:j + b] - 2 * (x @ y.T).double()
        else:
            diff = (x[:, None, :] - y[None, :, :]).abs()
            values = diff.sum(-1) if self.kind == "l1" else diff.amax(-1) if self.kind == "linf" else diff.square().sum(-1).sqrt()
        if self.sigma:
            if self.kind == "l2^2":
                values = values.double() + 0.5 * self.sigma - 2 * (self.source_extra[i:i + q].to(self.dtype) @ self.target_extra[j:j + b].to(self.dtype).T).double()
            else:
                rr = torch.arange(i, i + len(x), device=x.device)[:, None]
                cc = torch.arange(j, j + len(y), device=y.device)[None, :]
                values = values.double() + self.noise(rr, cc)
        return values.double()


def offsets(value):
    """
    CN: 分块 FP64 平方范数，避免完整 FP64 点云副本。
    EN: Compute FP64 squared norms in chunks without a full FP64 cloud copy.
    """
    out = torch.empty(len(value), device=value.device, dtype=torch.float64)
    block = max(1, 4 * 1024**2 // (8 * value.shape[1]))
    for start in range(0, len(value), block):
        part = value[start:start + block].double()
        out[start:start + len(part)] = (part * part).sum(-1)
    return out


def scan(cost, dual, k, *, certificate=False, direction=None):
    """
    CN: 统一扫描覆盖全部成本、精度和扰动，支持按方向初始化。
    EN: Cover all costs, precisions and perturbations with unified scans and directional initialization.
    """
    from hello_ot.kernels import generic_scan as unified
    from hello_ot.kernels.scan_contract import custom_topk_bucket

    if direction not in (None, "source", "target") or (certificate and direction is not None):
        raise ValueError("direction must be source/target or None; certificates require both directions")
    n, m = len(cost.source), len(cost.target)
    f, g = dual[:n], dual[n:]
    bucket = custom_topk_bucket(k, score_family="inner_product")
    precision = int(cost.precision == "fp64")
    kwargs = dict(workspace_bytes=cost.workspace_bytes, compute_precision=precision)
    if cost.kind == "l2^2":
        kwargs.update(raw_costs=(cost.source_offset, cost.target_offset),
                      raw_biases=(f-cost.source_offset, g-cost.target_offset))
        if cost.sigma:
            kwargs["raw_extras"] = (cost.source_extra, cost.target_extra, cost.sigma)
        kind = 3
    else:
        kind = {"l1": 0, "linf": 1, "l2": 2}[cost.kind]
        if cost.sigma:
            from hello_ot.costs.metric_perturbation import _index_hash_parameters
            coefficients = torch.tensor(_index_hash_parameters(cost.seed), dtype=torch.int64).reshape(-1)
            kwargs["perturbation"] = (cost.source_index, cost.target_index, cost.sigma,
                                       cost.seed, 1, True, coefficients)
    if direction is not None:
        reverse = direction == "target"
        x, y = (cost.target, cost.source) if reverse else (cost.source, cost.target)
        bias = f if reverse else g
        directional_kwargs = dict(workspace_bytes=cost.workspace_bytes, compute_precision=precision)
        if kind == 3:
            nx, ny = (cost.target_offset, cost.source_offset) if reverse else (cost.source_offset, cost.target_offset)
            directional_kwargs["query_bias"] = -nx
            bias = bias-ny
            if cost.sigma:
                ex, ey = (cost.target_extra, cost.source_extra) if reverse else (cost.source_extra, cost.target_extra)
                directional_kwargs["raw_extras"] = (ex, ey, cost.sigma)
        elif "perturbation" in kwargs:
            p = kwargs["perturbation"]
            directional_kwargs["perturbation"] = (p[1], p[0], *p[2:5], False, p[6]) if reverse else p
        values, indices = unified.directional(x, y, bias, bucket, kind, **directional_kwargs)
        values, indices = values[:, :min(k, len(y))], indices[:, :min(k, len(y))]
        unified._ROUTES["gpu_resident_directional:generic"] += 1
        return (None, None, values, indices, None) if reverse else (values, indices, None, None, None)
    result = unified.bidirectional(cost.source, cost.target, f, g, bucket, kind, **kwargs)
    unified._ROUTES["gpu_resident:generic"] += 1
    metrics = None
    if certificate:
        numerator, denominator, maximum, cmax, count = result[4:]
        metrics = {"l2": float(numerator.sqrt()/(1+denominator.sqrt())),
                   "linf": float(maximum/(1+cmax)), "max": float(maximum), "count": int(count)}
    # CN: pricing返回包含本行势的score，resident接口返回只含对侧势的候选值。
    # EN: Pricing includes the row potential; the resident interface returns opposite-side potential minus cost.
    return (result[0][:, :min(k, m)]-f[:, None], result[1][:, :min(k, m)],
            result[2][:, :min(k, n)]-g[:, None], result[3][:, :min(k, n)], metrics)
