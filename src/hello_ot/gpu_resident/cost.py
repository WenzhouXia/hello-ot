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


def _merge(values, indices, incoming, ids, k):
    merged = torch.cat((values, incoming), 1)
    keys = torch.cat((indices, ids), 1)
    best, order = merged.topk(k, dim=1, sorted=True)
    return best, keys.gather(1, order)


def scan(cost, dual, k, *, certificate=False):
    """
    CN: 双向 top-k 和完整对偶证书，只返回 GPU 数组与少量标量。
    EN: Bidirectional top-k and full dual certificates returning GPU arrays and scalar diagnostics only.
    """
    n, m = len(cost.source), len(cost.target)
    device = cost.source.device
    kx, ky = min(k, m), min(k, n)
    xv = torch.full((n, kx), -torch.inf, device=device, dtype=torch.float64)
    yv = torch.full((m, ky), -torch.inf, device=device, dtype=torch.float64)
    xi = torch.zeros((n, kx), device=device, dtype=torch.int64)
    yi = torch.zeros((m, ky), device=device, dtype=torch.int64)
    numerator = torch.zeros((), device=device, dtype=torch.float64)
    denominator = numerator.clone()
    vmax, cmax = numerator.clone(), numerator.clone()
    positive_count = torch.zeros((), device=device, dtype=torch.int64)
    # CN: 原生平方距离扫描直接借用连续输入，不经过 host staging。
    # EN: Native squared-distance scans borrow contiguous inputs without host staging.
    if cost.kind == "l2^2" and not cost.sigma and cost.precision == "fp32":
        from hello_ot.kernels.bilinear_topk import _load_extension
        from hello_ot.kernels.scan_contract import custom_topk_bucket
        ext = _load_extension()
        bucket = custom_topk_bucket(k, score_family="inner_product")
        chunk = max(1, min(n, 4096))
        db_tile = max(1, min(m, 2048, cost.workspace_bytes // (max(1, chunk) * 32)))
        u, v = dual[:n], dual[n:]
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            x, a = cost.source[start:stop], cost.source_offset[start:stop]
            if certificate and bucket <= 8:
                qv, qi, rv, ri, stats, _ = ext.fused_bidir_topk_stats_raw(
                    x, cost.target, a, cost.target_offset, u[start:stop] - a,
                    v - cost.target_offset, 128, db_tile, 256, bucket, True, True, False, 0, 2.)
                numerator += stats[0]
                denominator += stats[1]
                positive_count += stats[2].long()
                vmax = torch.maximum(vmax, qv.amax().clamp_min(0))
                cmax = torch.maximum(cmax, stats[3])
                xv[start:stop], xi[start:stop] = qv[:, :kx] - u[start:stop, None], qi[:, :kx]
                yv, yi = _merge(yv, yi, rv[:, :ky] - v[:, None], ri[:, :ky].long() + start, ky)
                continue
            qv, qi = ext.fused_directional_topk_raw(x, cost.target, -a, v - cost.target_offset, 128, db_tile, bucket, 2.0)
            rv, ri = ext.fused_directional_topk_raw(cost.target, x, -cost.target_offset, u[start:stop] - a, 128, min(chunk, 2048), bucket, 2.0)
            xv[start:stop], xi[start:stop] = qv[:, :kx], qi[:, :kx]
            yv, yi = _merge(yv, yi, rv[:, :ky], ri[:, :ky].long() + start, ky)
            if certificate:
                stats = ext.raw_lowrank_stats_only(x, cost.target, a, cost.target_offset, u[start:stop], v, 128, db_tile, 2.0)
                numerator += stats[0]
                denominator += stats[1]
                vmax = torch.maximum(vmax, stats[3])
                cmax = torch.maximum(cmax, stats[4])
                positive_count += stats[2].long()
    else:
        q = min(n, 128)
        bytes_per_pair = 32 if cost.kind == "l2^2" else max(32, cost.source.shape[1] * 24)
        b = max(1, min(m, 2048, cost.workspace_bytes // (q * bytes_per_pair)))
        for i in range(0, n, q):
            for j in range(0, m, b):
                c = cost.tile(i, j, q, b)
                qr, br = c.shape
                sx = dual[n + j:n + j + br][None, :] - c
                sy = (dual[i:i + qr, None] - c).T
                xv[i:i + qr], xi[i:i + qr] = _merge(xv[i:i + qr], xi[i:i + qr], sx,
                    torch.arange(j, j + br, device=device)[None, :].expand(qr, -1), kx)
                yv[j:j + br], yi[j:j + br] = _merge(yv[j:j + br], yi[j:j + br], sy,
                    torch.arange(i, i + qr, device=device)[None, :].expand(br, -1), ky)
                if certificate:
                    violation = (dual[i:i + qr, None] + sx).clamp_min(0)
                    numerator += violation.square().sum()
                    denominator += c.square().sum()
                    vmax = torch.maximum(vmax, violation.amax())
                    cmax = torch.maximum(cmax, c.abs().amax())
                    positive_count += torch.count_nonzero(violation)
    metrics = None
    if certificate:
        metrics = {"l2": float(numerator.sqrt() / (1 + denominator.sqrt())),
                   "linf": float(vmax / (1 + cmax)), "max": float(vmax), "count": int(positive_count)}
    return xv, xi, yv, yi, metrics
