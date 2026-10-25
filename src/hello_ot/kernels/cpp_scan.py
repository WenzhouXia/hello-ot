import math
import time

import torch
from hello_ot.kernels.torch_scan import bidirectional_violation_scan as _torch_violation_scan
from hello_ot.kernels.torch_scan import directional_assignment as _torch_directional_assignment


def fused_dot_tile(source, target, source_bias, target_bias, f, g, rv, ri, cv, ci,
                   dot_scale, source_start, target_start, mode):
    """
    CN: BLAS 计算点积，再融合偏置、score、证书和候选筛选。
    EN: Compute dot products with BLAS, then fuse biases, scores, certificates, and candidate selection.
    """
    from hello_ot._cpp_backend import load_cpp_extension

    extension = load_cpp_extension("scan")
    products = torch.mm(source, target.T)
    return extension.dot_tile(products.numpy(), source_bias.numpy(), target_bias.numpy(),
                              f.numpy(), g.numpy(), rv.numpy(), ri.numpy(), cv.numpy(), ci.numpy(),
                              dot_scale, source_start, target_start, mode, torch.get_num_threads())


def cpp_directional_assignment(*, state, source_points, target_points, source_offset,
                               target_offset, score_family, cost_type, dot_scale, known_side,
                               topk, device, max_tile_bytes=64 * 1024 * 1024,
                               perturbation=None, compute_precision="fp32"):
    """
    CN: 内积成本采用融合 CPU dual assignment，其他成本保留参考实现。
    EN: Use fused CPU dual assignment for inner-product costs and retain reference handling for other costs.
    """
    import hello_ot.kernels.torch_scan as scan

    if score_family != "inner_product" or perturbation is not None:
        return _torch_directional_assignment(
            state=state, source_points=source_points, target_points=target_points,
            source_offset=source_offset, target_offset=target_offset, score_family=score_family,
            cost_type=cost_type, dot_scale=dot_scale, known_side=known_side, topk=topk,
            device=device, max_tile_bytes=max_tile_bytes, perturbation=perturbation,
            compute_precision=compute_precision)
    started = time.perf_counter()
    dev = scan.resolve_torch_device(device)
    if dev.type != "cpu":
        raise ValueError("C++ scans require CPU execution")
    source, target, a, b = scan._problem_tensors(
        source_points=source_points, target_points=target_points, source_offset=source_offset,
        target_offset=target_offset, device=dev, compute_precision=compute_precision)
    _, _, _, dual = scan._state_tensors(state, device=dev)
    n, m = len(source), len(target)
    if dual.numel() != n + m:
        raise ValueError("dual length does not match the directional scan problem")
    if a is None or b is None:
        raise ValueError("inner-product score family requires source/target offsets")
    row_direction = str(known_side) == "target"
    k = min(int(topk), m if row_direction else n)
    rv = torch.full((n, k if row_direction else 1), -torch.inf, dtype=torch.float64)
    cv = torch.full((m, 1 if row_direction else k), -torch.inf, dtype=torch.float64)
    ri, ci = torch.full_like(rv, -1, dtype=torch.int64), torch.full_like(cv, -1, dtype=torch.int64)
    sb, tb = scan._tile_shape(n, m, max_tile_bytes=int(max_tile_bytes))
    blocks = ((s, t) for s in range(0, n, sb) for t in range(0, m, tb)) if row_direction else (
        (s, t) for t in range(0, m, tb) for s in range(0, n, sb))
    for s, t in blocks:
        e, u = min(n, s + sb), min(m, t + tb)
        fused_dot_tile(source[s:e], target[t:u], a[s:e], b[t:u], dual[s:e], dual[n+t:n+u],
                       rv[s:e], ri[s:e], cv[t:u], ci[t:u], dot_scale, s, t, 1 if row_direction else 2)
    best_values, best_indices = (rv, ri) if row_direction else (cv, ci)
    queries = torch.arange(n if row_direction else m)[:, None].expand(-1, k).reshape(-1)
    rows, cols = (queries, best_indices.reshape(-1)) if row_direction else (best_indices.reshape(-1), queries)
    augmented = scan._merge_state_candidates(state, candidate_rows=rows, candidate_cols=cols, device=dev)
    candidate = scan.TorchCompletionCandidate(known_side=known_side,
                                             values=best_values[:, 0].contiguous(),
                                             indices=best_indices[:, 0].contiguous())
    stats = dict(cross_transfer_added_raw=rows.numel(), cross_transfer_added_unique=augmented.rows.numel())
    profile = dict(backend="cpp_cpu_blockwise", device=str(dev), known_side=str(known_side),
                   query_count=n if row_direction else m, database_count=m if row_direction else n,
                   topk=k, source_block=sb, target_block=tb, max_tile_elements=sb*tb,
                   total_time=time.perf_counter()-started)
    return augmented, stats, profile, candidate


def cpp_merge_topk(current_values, current_indices, new_values, new_indices, *, k):
    """
    CN: 在 C++ 中直接合并 strided CPU tile 和小候选集。
    EN: Merge a strided CPU tile directly with small candidate sets in C++.
    """
    from hello_ot._cpp_backend import load_cpp_extension

    extension = load_cpp_extension("scan")
    values, indices = extension.merge_topk(
        current_values.numpy(), current_indices.numpy(), new_values.numpy(), new_indices.numpy(),
        k, torch.get_num_threads())
    return torch.from_numpy(values), torch.from_numpy(indices)


def cpp_violation_scan(*, source_points, target_points, source_offset, target_offset,
                       source_dual, target_dual, score_family, cost_type, dot_scale,
                       topk, theta, device, collect_linf_diagnostics=True,
                       max_tile_bytes=64 * 1024 * 1024, perturbation=None,
                       compute_precision="fp32"):
    """
    CN: 保留 Torch 成本计算，在 C++ 中融合 score 和证书归约并筛选双向候选。
    EN: Preserve Torch cost arithmetic and fuse score generation and certificate reductions in C++.
    """
    import hello_ot.kernels.torch_scan as scan
    from hello_ot._cpp_backend import load_cpp_extension

    started = time.perf_counter()
    dev = scan.resolve_torch_device(device)
    if dev.type != "cpu":
        raise ValueError("C++ scans require CPU execution")
    if score_family != "inner_product" and compute_precision == "fp32":
        # CN: norm FP32 的平方和也在 FP32 累积，沿用参考路径以保留其数值口径。
        # EN: FP32 norm costs also accumulate squared norms in FP32; retain reference arithmetic.
        return _torch_violation_scan(
            source_points=source_points, target_points=target_points, source_offset=source_offset,
            target_offset=target_offset, source_dual=source_dual, target_dual=target_dual,
            score_family=score_family, cost_type=cost_type, dot_scale=dot_scale,
            topk=topk, theta=theta, device=device, collect_linf_diagnostics=collect_linf_diagnostics,
            max_tile_bytes=max_tile_bytes, perturbation=perturbation, compute_precision=compute_precision)
    extension = load_cpp_extension("scan")
    source, target, source_bias, target_bias = scan._problem_tensors(
        source_points=source_points, target_points=target_points, source_offset=source_offset,
        target_offset=target_offset, device=dev, compute_precision=compute_precision)
    f = scan._tensor(source_dual, dtype=torch.float64, device=dev).view(-1)
    g = scan._tensor(target_dual, dtype=torch.float64, device=dev).view(-1)
    n, m = len(source), len(target)
    rk, ck = min(int(topk), m), min(int(topk), n)
    rv = torch.full((n, rk), -torch.inf, dtype=torch.float64)
    ri = torch.full((n, rk), -1, dtype=torch.int64)
    cv = torch.full((m, ck), -torch.inf, dtype=torch.float64)
    ci = torch.full((m, ck), -1, dtype=torch.int64)
    sb, tb = scan._tile_shape(n, m, max_tile_bytes=int(max_tile_bytes))
    numerator = denominator = maximum = cost_maximum = 0.0
    count = 0
    for a in range(0, n, sb):
        b = min(n, a + sb)
        for c in range(0, m, tb):
            d = min(m, c + tb)
            if score_family == "inner_product" and perturbation is None:
                if source_bias is None or target_bias is None:
                    raise ValueError("inner-product score family requires source/target offsets")
                ns, ds, ct, mx, cm = fused_dot_tile(
                    source[a:b], target[c:d], source_bias[a:b], target_bias[c:d], f[a:b], g[c:d],
                    rv[a:b], ri[a:b], cv[c:d], ci[c:d], dot_scale, a, c, 0)
                numerator += ns
                denominator += ds
                count += ct
                maximum = max(maximum, mx)
                cost_maximum = max(cost_maximum, cm)
                continue
            cost = scan._cost_tile(source[a:b], target[c:d], score_family=score_family,
                                   cost_type=cost_type, dot_scale=dot_scale,
                                   source_offset=None if source_bias is None else source_bias[a:b],
                                   target_offset=None if target_bias is None else target_bias[c:d])
            if perturbation is not None:
                cost = scan._perturb_tile(cost, perturbation, a, c)
            score, ns, ds, ct, mx, cm = extension.score_and_stats(
                cost.numpy(), f[a:b].numpy(), g[c:d].numpy(), torch.get_num_threads())
            score = torch.from_numpy(score)
            numerator += ns
            denominator += ds
            count += ct
            maximum = max(maximum, mx)
            cost_maximum = max(cost_maximum, cm)
            rv[a:b], ri[a:b] = cpp_merge_topk(
                rv[a:b], ri[a:b], score, torch.arange(c, d)[None, :].expand(b-a, -1), k=rk)
            cv[c:d], ci[c:d] = cpp_merge_topk(
                cv[c:d], ci[c:d], score.T, torch.arange(a, b)[None, :].expand(d-c, -1), k=ck)
    rm, cm = rv > float(theta), cv > float(theta)
    rows = torch.cat((torch.arange(n)[:, None].expand(-1, rk)[rm], ci[cm]))
    cols = torch.cat((ri[rm], torch.arange(m)[:, None].expand(-1, ck)[cm]))
    relative = math.sqrt(max(0.0, numerator)) / (1.0 + math.sqrt(max(0.0, denominator)))
    diagnostics = dict(
        scan_backend="cpp_cpu_blockwise", device=str(dev), dual_feasibility=relative,
        dual_feasibility_num_sq=numerator, dual_feasibility_den_sq=denominator,
        dual_feasibility_positive_count=count, dual_feasibility_max_violation=maximum,
        dual_feasibility_cost_linf=cost_maximum,
        relative_linf_dual_feasibility=maximum / (1.0 + cost_maximum) if collect_linf_diagnostics else None,
        fused_scan_candidate_count=rows.numel(), fused_scan_topk=int(topk), source_block=sb,
        target_block=tb, max_tile_elements=sb*tb, pairwise_passes=1,
        fused_scan_time=time.perf_counter()-started)
    return scan.TorchViolationScan(rows=rows, cols=cols, dual_feasibility=relative, diagnostics=diagnostics)
