"""
CN: 混合精度和FP64统一扫描适配器；稀疏点对成本保留独立实现。
EN: Unified mixed-precision and FP64 scan adapters with separate sparse pair costs.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache, wraps
import math
import threading

import torch

from hello_ot._source_build import load_native_module
from hello_ot.kernels.scan_planner import select_scan_tile, WorkspaceBudgetCache

_ROUTES = Counter()
_BUDGETS = WorkspaceBudgetCache()
_OOM_RETRY = threading.local()


def scan_routes(*, reset=False):
    """
    CN: 提供实际路由计数供回归报告使用，不改变求解输出。
    EN: Expose actual route counts for regression reports without changing solve outputs.
    """
    result = dict(_ROUTES)
    if reset:
        _ROUTES.clear()
    return result


@lru_cache(None)
def _module(component):
    """
    CN: 按实际使用加载发行扩展；不依赖实验目录或Triton。
    EN: Load distribution extensions on demand without experiment-directory or Triton dependencies.
    """
    return load_native_module(f"hello_ot_native.hierot_generic_{component}_ext")


def _check(x, y, k, *vectors):
    """
    CN: 在CUDA指针调用前校验共同的设备、类型和索引范围。
    EN: Validate devices, dtypes and index ranges before invoking CUDA pointer interfaces.
    """
    if k not in (1, 2, 4, 8, 16, 32):
        raise ValueError("generic scan requires a power-of-two bucket in [1,32]")
    if (not x.is_cuda or not y.is_cuda or x.device != y.device
            or x.dtype not in (torch.float32, torch.float64) or y.dtype != x.dtype
            or x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or min(*x.shape, *y.shape) < 1 or max(*x.shape, *y.shape) >= 2**31):
        raise ValueError("generic scan requires nonempty matching CUDA FP32/FP64 feature matrices with int32 dimensions")
    for v, count in vectors:
        if v.device != x.device or v.dtype != torch.float64 or v.numel() != count:
            raise ValueError("generic scan requires matching CUDA FP64 bias vectors")


def _workspace_budget(x):
    """
    CN: 所有设备扫描共用临时空间上限，并允许复用allocator缓存。
    EN: Share a device-scan scratch limit while allowing allocator-cache reuse.
    """
    allocated=torch.cuda.memory_allocated(x.device)
    def query():
        """
        CN: 只在快照过期时调用可能有较高延迟的驱动查询。
        EN: Call the potentially high-latency driver query only when the snapshot expires.
        """
        free,_=torch.cuda.mem_get_info(x.device)
        return free+torch.cuda.memory_reserved(x.device)-torch.cuda.memory_allocated(x.device)
    return _BUDGETS.get(x.device,allocated,query)


def _retry_workspace_oom(function):
    """
    CN: 外部占用突变导致OOM时刷新预算并完整重试一次；显式预算和已有计划不改写。
    EN: Refresh budgets and retry a whole scan once after abrupt external-pressure OOM; leave explicit budgets and existing plans unchanged.
    """
    @wraps(function)
    def call(*args,**kwargs):
        """
        CN: 失败尝试的局部输出全部丢弃，不重复累计到调用方结果。
        EN: Discard failed attempts' local outputs without double-counting caller results.
        """
        if getattr(_OOM_RETRY,'active',False):return function(*args,**kwargs)
        _OOM_RETRY.active=True
        try:
            try:return function(*args,**kwargs)
            except torch.cuda.OutOfMemoryError:
                if kwargs.get('workspace_bytes') is not None or kwargs.get('planned_tiles') is not None:raise
                _BUDGETS.invalidate(args[0].device)
            return function(*args,**kwargs)
        finally:_OOM_RETRY.active=False
    return call


@lru_cache(maxsize=256)
def _cached_scan_tile(q,n,k,workspace_bytes,pricing,matrix_bytes,x_copy_bytes,y_copy_bytes,target_blocks,tile_limits):
    """
    CN: 复用相同形状、布局和有效预算的确定性计划；预算变化形成不同键。
    EN: Reuse deterministic plans for identical shapes, layouts and valid budgets, with budget changes forming distinct keys.
    """
    return select_scan_tile(q,n,k,workspace_bytes=workspace_bytes,pricing=pricing,matrix_bytes=matrix_bytes,
        copy_x_bytes=x_copy_bytes,copy_y_bytes=y_copy_bytes,target_blocks=target_blocks,tile_limits=tile_limits)


def _tile_shape(x, y, k, *, pricing, matrix, workspace_bytes=None, matrix_bytes=4, cast_bytes=0, tile_limits=None):
    """
    CN: 同时约束成本矩阵、列候选、合并和坐标副本；两侧均可分块。
    EN: Bound cost matrices, column candidates, merges and coordinate copies, allowing tiling of either side.
    """
    automatic_budget=workspace_bytes is None
    if automatic_budget:
        workspace_bytes = _workspace_budget(x)
    target_blocks = 2 * torch.cuda.get_device_properties(x.device).multi_processor_count if x.is_cuda else 256
    d = x.shape[1]
    x_copy_bytes = d * (cast_bytes + (0 if x.is_contiguous() else x.element_size()))
    y_copy_bytes = d * (cast_bytes + (0 if y.is_contiguous() else y.element_size()))
    arguments=(len(x),len(y),k,workspace_bytes,pricing,matrix_bytes if matrix else 0,
        x_copy_bytes,y_copy_bytes,target_blocks,None if tile_limits is None else tuple(tile_limits))
    try:return _cached_scan_tile(*arguments)
    except RuntimeError as error:
        if not automatic_budget or str(error)!='generic scan workspace cannot fit one tile':raise
        # CN: 旧预算过低时也重新采样，避免外部释放显存后仍被旧快照拒绝。
        # EN: Resample overly restrictive snapshots too, avoiding rejection after external memory is released.
        _BUDGETS.invalidate(x.device)
        return _cached_scan_tile(*arguments[:3],_workspace_budget(x),*arguments[4:])


def _merge(old, incoming, k):
    """
    CN: 用同一精确队列合并两个候选集合，同分时优先较小索引。
    EN: Merge candidate sets with the same exact queue, preferring lower indices on ties.
    """
    if old is None:
        return incoming
    module = _module("pricing_small" if k <= 2 else "pricing")
    return module.merge(torch.stack((old[0], incoming[0])), torch.stack((old[1], incoming[1])))


def _shift(indices, offset):
    """
    CN: 平移有效索引，保留候选不足时的负哨兵。
    EN: Shift valid indices while retaining negative sentinels for missing candidates.
    """
    return indices if offset == 0 else torch.where(indices >= 0, indices + offset, indices)


def _matrix_workspace_bytes(x,precision,perturbation,raw_extras=None):
    """
    CN: 按同时存活的矩阵计数，扰动需包含类型转换、噪声和相加输出。
    EN: Count simultaneously live matrices, including casts, noise and sum outputs for perturbation.
    """
    if raw_extras is not None:return 32
    wide=precision==1 or x.dtype==torch.float64
    return (24 if wide else 28) if perturbation is not None else (8 if wide else 4)


def _matrix(x, y, kind, nx=None, ny=None, compute_precision=0):
    """
    CN: L1/Linf复用寄存器分块；L2展开出现严重相消时回退直接距离。
    EN: Reuse register tiles for L1/Linf and fall back to direct L2 distances on severe cancellation.
    """
    cost = _module("cost")
    if compute_precision == 1:
        return cost.direct_norm_fp64(x, y, kind, 2)
    if kind in (0, 1):
        return cost.direct_norm(x, y, kind, 8 if kind == 0 else 2)
    m = cost.dots(x, y, 1.)
    flag = cost.finish(m, nx, ny, x.shape[1])
    if bool(flag.item()):
        del m
        return torch.cdist(x, y, p=2., compute_mode="donot_use_mm_for_euclid_dist")
    return m


def _initial_policy(d, kind):
    """
    CN: 只按成本与维度范围选择有限几何，不使用维度白名单。
    EN: Select finite geometries by cost and dimension ranges, without ordinary-dimension whitelists.
    """
    if d in (2, 3):
        return False, 3 if kind == 2 and d == 3 else 1
    if kind == 2:
        return d > 4, 0 if d > 4 else 3
    return True, 0 if kind == 0 else 3


def _norm_perturbation(x, y, args):
    """
    CN: 保留旧扰动ABI，校验全局索引；系数及方向在所有分块中不变。
    EN: Preserve the legacy perturbation ABI and validate global indices, keeping coefficients and direction fixed across tiles.
    """
    if args is None or float(args[2]) == 0:
        return None
    qi, di, sigma, seed, mode, direction, coefficients = args
    if not math.isfinite(float(sigma)) or sigma < 0 or mode not in (0, 1):
        raise ValueError("invalid norm perturbation configuration")
    for index, count in ((qi, len(x)), (di, len(y))):
        if index.device != x.device or index.dtype != torch.int64 or index.numel() != count:
            raise ValueError("norm perturbation requires matching CUDA int64 global indices")
    if mode == 1 and (coefficients.dtype != torch.int64 or coefficients.numel() != 6):
        raise ValueError("index_hash requires six int64 coefficients")
    return qi.contiguous().view(-1), di.contiguous().view(-1), float(sigma), int(seed), int(mode), bool(direction), coefficients


def _add_perturbation(matrix, perturbation, row, col):
    """
    CN: 将基础成本与旧公式的噪声在FP64中相加，临时矩阵仅覆盖当前块。
    EN: Add legacy noise to base costs in FP64, restricting temporary matrices to the current tile.
    """
    qi, di, sigma, seed, mode, direction, coefficients = perturbation
    return _module("cost").perturb_cost(matrix, qi[row:row + len(matrix)], di[col:col + matrix.shape[1]],
                                      sigma, seed, mode, direction, coefficients)


@_retry_workspace_oom
def directional(x, y, db_bias, k, kind, *, query_bias=None, dot_scale=2., workspace_bytes=None,
                minimum=False, perturbation=None, collect_statistics=False, compute_precision=0, raw_extras=None):
    """
    CN: 返回降序分数；raw严格使用传入偏置，norm分数为dual减cost。
    EN: Return descending scores; raw uses supplied biases and norm scores equal dual minus cost.
    """
    raw = query_bias is not None
    if kind not in (0, 1, 2, 3) or (kind == 3 and not raw):
        raise ValueError("generic directional cost must be l1, linf, l2 or raw")
    vectors = [(db_bias, len(y))]
    if raw:
        vectors.append((query_bias, len(x)))
    _check(x, y, k, *vectors)
    if compute_precision not in (0, 1) or (not raw and x.dtype != torch.float32):
        raise ValueError("norm scans require FP32 coordinate storage and precision 0 or 1")
    p = _norm_perturbation(x, y, perturbation)
    if raw and (p is not None or collect_statistics):
        raise ValueError("norm extensions do not apply to raw initial")
    if raw and dot_scale not in (1., 2.):
        raise ValueError("dot_scale must be exactly 1 or 2")
    matrix, config = (True, 2) if raw else _initial_policy(x.shape[1], kind)
    matrix |= p is not None or compute_precision == 1
    module = _module("initial_small" if k == 1 else "initial")
    empty = db_bias.new_empty(0)
    flag = torch.empty(0, dtype=torch.int32, device=x.device)
    if not raw and k==1 and x.shape[1]<=32 and p is None and workspace_bytes is None and all(value.is_contiguous() for value in (x,y,db_bias)) and y.numel()*y.element_size()+2*2**20<=_workspace_budget(x):
        result=module.direct_top1(x,y,db_bias,kind,compute_precision,collect_statistics)
        values,indices=result[:2]
        return ((values if minimum else values.neg_()),indices,*([result[2]] if collect_statistics else []))
    # CN: 直接距离核只分配最终候选，无需矩阵规划或分块包装。
    # EN: Direct-distance kernels allocate only final candidates and need no matrix planning or tile wrappers.
    if not matrix and workspace_bytes is None and all(value.is_contiguous() for value in (x,y,db_bias)):
        result = module.topk(x,y,db_bias,kind,config,x.new_empty(0),0,empty,empty,flag,k,
                             *([True] if collect_statistics else []))
        values, indices = result[:2]
        return ((values if minimum else values.neg_()), indices, *result[2:])
    qt, nt = _tile_shape(x, y, k, pricing=False, matrix=matrix, workspace_bytes=workspace_bytes,
                         cast_bytes=8 if raw and compute_precision == 1 and x.dtype!=torch.float64 else 0,
                         matrix_bytes=_matrix_workspace_bytes(x,compute_precision,p,raw_extras))
    one_query_tile = qt >= len(x)
    out = None if one_query_tile else torch.empty((len(x), k), dtype=torch.float64, device=x.device)
    ids = None if one_query_tile else torch.empty((len(x), k), dtype=torch.int64, device=x.device)
    fixed_statistics = None
    for i in range(0, len(x), qt):
        xx = x[i:i + qt].contiguous()
        nx = _module("cost").norms(xx) if matrix and kind == 2 and not raw and compute_precision == 0 else empty
        best = None
        for j in range(0, len(y), nt):
            yy = y[j:j + nt].contiguous()
            g = db_bias[j:j + nt].contiguous()
            if raw:
                extras = None if raw_extras is None else (raw_extras[0][i:i+qt], raw_extras[1][j:j+nt], raw_extras[2])
                m = _raw_dots(xx, yy, dot_scale, compute_precision, extras)
                nx_arg, ny, mode = query_bias[i:i + qt].contiguous(), empty, 7 if m.dtype == torch.float64 else 6
            elif matrix:
                ny = _module("cost").norms(yy) if kind == 2 and compute_precision == 0 else empty
                m = _matrix(xx, yy, kind, nx, ny, compute_precision)
                nx_arg, mode = empty, 5 if compute_precision == 1 else 1
                if p is not None:
                    m = _add_perturbation(m, p, i, j)
                    mode = 5
            else:
                m, nx_arg, ny, mode = x.new_empty(0), empty, empty, 0
            if matrix and not raw and k==1:
                result=_module('pricing_small').matrix_initial(m,g)
            else:
                result = module.topk(xx, yy, g, 3 if raw else kind, config, m, mode, nx_arg, ny, flag, k,
                                     *([True] if collect_statistics else []))
            values, indices = result[:2]
            if collect_statistics:
                if fixed_statistics is None:
                    fixed_statistics = result[2]
                else:
                    fixed_statistics[0].add_(result[2][0])
                    fixed_statistics[1].copy_(torch.maximum(fixed_statistics[1], result[2][1]))
            del m
            # CN: 一个数据库块时无需合并或索引平移，直接保留调用方需要的符号。
            # EN: A single database tile needs no merge or index shift; retain the caller's desired sign.
            if nt >= len(y):
                best = (values if minimum else values.neg_(), indices)
            else:
                best = _merge(best, (values.neg_(), _shift(indices, j)), k)
        if minimum and nt < len(y):
            best[0].neg_()
        if one_query_tile:
            return (*best, fixed_statistics) if collect_statistics else best
        out[i:i + len(xx)], ids[i:i + len(xx)] = best
    return (out, ids, fixed_statistics) if collect_statistics else (out, ids)


def _raw_dots(x, y, dot_scale, compute_precision=0, extras=None):
    """
    CN: 仅转换当前坐标块；平方成本扰动保留独立低秩列的点积精度。
    EN: Cast only current feature tiles and preserve separate low-rank perturbation dot precision.
    """
    if compute_precision == 1:
        x, y = x.double(), y.double()
    cost = _module("cost")
    dots = getattr(cost, "dots_fp64" if x.dtype == torch.float64 else "dots")(x, y, dot_scale)
    if extras is not None:
        ex, ey, sigma = extras
        if compute_precision == 1:
            ex, ey = ex.double(), ey.double()
        extra = getattr(cost, "dots_fp64" if ex.dtype == torch.float64 else "dots")(ex.contiguous(), ey.contiguous(), 2.)
        dots = dots.double() + extra.double() - 0.5 * sigma
    return dots


@_retry_workspace_oom
def bidirectional(x, y, f, g, k, kind, *, raw_costs=None, raw_biases=None,
                  dot_scale=2., workspace_bytes=None, perturbation=None, compute_precision=0, raw_extras=None, tile_limits=None,
                  planned_tiles=None):
    """
    CN: 双向候选跨两侧分块合并；完整统计每个点对累计一次。
    EN: Merge bidirectional candidates across both tile axes; count each pair once in full statistics.
    """
    _check(x, y, k, (f, len(x)), (g, len(y)))
    p = _norm_perturbation(x, y, perturbation)
    raw = raw_costs is not None
    if compute_precision not in (0, 1) or (not raw and x.dtype != torch.float32):
        raise ValueError("norm scans require FP32 coordinate storage and precision 0 or 1")
    if raw and p is not None:
        raise ValueError("norm perturbation does not apply to raw pricing")
    if kind not in (0, 1, 2, 3) or (kind == 3 and not raw):
        raise ValueError("generic bidirectional cost must be l1, linf, l2 or raw")
    if raw:
        nx, ny = raw_costs
        qb, dbb = raw_biases
        _check(x, y, k, (nx, len(x)), (ny, len(y)), (qb, len(x)), (dbb, len(y)))
        if dot_scale not in (1., 2.):
            raise ValueError("dot_scale must be exactly 1 or 2")
    d = x.shape[1]
    matrix = raw or d > (64 if kind == 2 else 4)
    matrix |= k == 4 and kind != 3 and 1 <= d <= (64 if kind == 2 else 4)
    matrix |= p is not None or compute_precision == 1
    # CN: raw FP64 的块内转换及扰动点积也计入临时空间预算。
    # EN: Include tiled raw FP64 casts and perturbation dot products in the scratch budget.
    if planned_tiles is None:
        qt, nt = _tile_shape(x, y, k, pricing=True, matrix=matrix, workspace_bytes=workspace_bytes,
                             tile_limits=tile_limits,
                             cast_bytes=8 if raw and compute_precision == 1 and x.dtype!=torch.float64 else 0,
                             matrix_bytes=_matrix_workspace_bytes(x,compute_precision,p,raw_extras))
    else:
        # CN: 流式调用方已把上传、常驻及扫描工作区联合纳入预算，不在执行层再次规划。
        # EN: Streamed callers already budget uploads, residency and scan scratch jointly; execution does not replan.
        qt,nt=min(len(x),planned_tiles[0]),min(len(y),planned_tiles[1])
        if min(qt,nt)<=0:raise ValueError('planned scan tiles must be positive')
    if raw and k <= 2 and raw_extras is None and (compute_precision == 0 or x.dtype == torch.float64) and all(value.is_contiguous() for value in (x,y,nx,ny,qb,dbb)):
        # CN: 常驻与流式入口共用原生分块执行器，各入口只规划一次。
        # EN: Resident and streamed entry points share native tiled execution, planning once per entry point.
        a=_module('pricing_small').raw_scan_tiled(x,y,nx,ny,qb,dbb,k,dot_scale,True,True,qt,nt)
        return a[0],a[1].long(),a[2],a[3].long(),a[4][0],a[4][1],a[0].amax().clamp_min(0),a[4][3],a[4][2].long()
    module = _module("pricing_small" if k <= 2 else "pricing")
    empty = f.new_empty(0)
    one_tile = qt >= len(x) and nt >= len(y)
    rv = None if one_tile else torch.empty((len(x), k), dtype=torch.float64, device=x.device)
    ri = None if one_tile else torch.empty((len(x), k), dtype=torch.int64, device=x.device)
    cv = None if one_tile else torch.empty((len(y), k), dtype=torch.float64, device=x.device)
    ci = None if one_tile else torch.empty((len(y), k), dtype=torch.int64, device=x.device)
    total = None if one_tile else torch.zeros(5, dtype=torch.float64, device=x.device)
    for i in range(0, len(x), qt):
        xx, ff = x[i:i + qt].contiguous(), f[i:i + qt].contiguous()
        nxx = nx[i:i + qt].contiguous() if raw else (_module("cost").norms(xx) if matrix and kind == 2 and compute_precision == 0 else empty)
        best = None
        for j in range(0, len(y), nt):
            yy, gg = y[j:j + nt].contiguous(), g[j:j + nt].contiguous()
            nyy = ny[j:j + nt].contiguous() if raw else (_module("cost").norms(yy) if matrix and kind == 2 and compute_precision == 0 else empty)
            if raw:
                extras = None if raw_extras is None else (raw_extras[0][i:i+qt], raw_extras[1][j:j+nt], raw_extras[2])
                m = _raw_dots(xx, yy, dot_scale, compute_precision, extras)
            else:
                m = _matrix(xx, yy, kind, nxx, nyy, compute_precision) if matrix else x.new_empty(0)
                if p is not None:
                    m = _add_perturbation(m, p, i, j)
            a = module.scan(xx, yy, ff, gg, kind, k, m, raw, nxx if raw else empty, nyy if raw else empty,
                            qb[i:i + qt].contiguous() if raw else empty, dbb[j:j + nt].contiguous() if raw else empty)
            del m
            if one_tile:
                return *a[:4], a[4][0], a[4][1], a[4][2], a[4][3], a[4][4].long()
            best = _merge(best, (a[0], _shift(a[1], j)), k)
            current = (cv[j:j + len(yy)].contiguous(), ci[j:j + len(yy)].contiguous()) if i else None
            cv[j:j + len(yy)], ci[j:j + len(yy)] = _merge(current, (a[2], _shift(a[3], i)), k)
            total[:2].add_(a[4][:2])
            total[2:4].copy_(torch.maximum(total[2:4], a[4][2:4]))
            total[4].add_(a[4][4])
        rv[i:i + len(xx)], ri[i:i + len(xx)] = best
    return rv, ri, cv, ci, total[0], total[1], total[2], total[3], total[4].long()


@_retry_workspace_oom
def _raw_certificate_stats(x,y,nx,ny,f,g,*,dot_scale=2.,compute_denominator=True,compute_precision=0,workspace_bytes=None,caller_layout=False):
    """
    CN: 一个原生执行器复用点积缓冲、累计所有统计；两种返回ABI共用同一实现。
    EN: Reuse dot scratch and accumulate all statistics in one native executor, shared by both return ABIs.
    """
    _check(x,y,1,(nx,len(x)),(ny,len(y)),(f,len(x)),(g,len(y)))
    if dot_scale not in (1.,2.) or compute_precision not in (0,1):raise ValueError('invalid raw certificate scale or precision')
    qt,nt=_tile_shape(x,y,1,pricing=False,matrix=True,workspace_bytes=workspace_bytes,
        cast_bytes=8 if compute_precision==1 and x.dtype!=torch.float64 else 0,
        matrix_bytes=_matrix_workspace_bytes(x,compute_precision,None))
    return _module('cost').raw_certificate_tiled(x,y,nx,ny,f,g,qt,nt,dot_scale,
        compute_denominator,compute_precision==1,caller_layout)


@_retry_workspace_oom
def certificate_only(x, y, f, g, kind, *, perturbation=None, compute_denominator=True, workspace_bytes=None, compute_precision=0,
                     raw_costs=None, dot_scale=2.):
    """
    CN: 两侧分块生成成本并归约完整证书，不运行top-k或分配候选缓冲。
    EN: Generate tiled costs and reduce the full certificate without top-k work or candidate buffers.
    """
    _check(x, y, 1, (f, len(x)), (g, len(y)))
    raw = raw_costs is not None
    if compute_precision not in (0, 1) or (not raw and x.dtype != torch.float32):
        raise ValueError("norm scans require FP32 coordinate storage and precision 0 or 1")
    if kind not in (0, 1, 2, 3) or (kind == 3 and not raw):
        raise ValueError("certificate-only requires a norm or raw cost")
    if raw:
        _check(x, y, 1, (raw_costs[0], len(x)), (raw_costs[1], len(y)))
        if dot_scale not in (1., 2.) or perturbation is not None:
            raise ValueError("raw certificates require dot_scale 1 or 2 and encoded perturbation")
        total=_raw_certificate_stats(x,y,*raw_costs,f,g,dot_scale=dot_scale,compute_denominator=compute_denominator,
            compute_precision=compute_precision,workspace_bytes=workspace_bytes)
        return total[0],total[1],total[2],total[3],total[4].long()
    p = _norm_perturbation(x, y, perturbation)
    qt, nt = _tile_shape(x, y, 1, pricing=False, matrix=True, workspace_bytes=workspace_bytes,
                         matrix_bytes=_matrix_workspace_bytes(x,compute_precision,p))
    cost = _module("cost")
    total = None
    for i in range(0, len(x), qt):
        xx, ff = x[i:i+qt].contiguous(), f[i:i+qt].contiguous()
        nx = cost.norms(xx) if kind == 2 and compute_precision == 0 else None
        for j in range(0, len(y), nt):
            yy, gg = y[j:j+nt].contiguous(), g[j:j+nt].contiguous()
            ny = cost.norms(yy) if kind == 2 and compute_precision == 0 else None
            matrix = _matrix(xx, yy, kind, nx, ny, compute_precision)
            if p is not None:
                matrix = _add_perturbation(matrix, p, i, j)
            a = cost.certificate(matrix, ff, gg, compute_denominator)
            del matrix
            if total is None:
                total = a
            else:
                total[:2].add_(a[:2])
                total[2:4].copy_(torch.maximum(total[2:4], a[2:4]))
                total[4].add_(a[4])
    return total[0], total[1], total[2], total[3], total[4].long()


class ScanExtension:
    """
    CN: 统一扫描入口，保留调用方ABI与路由诊断；稀疏点对成本独立加载。
    EN: Unified scan entry points retaining caller ABIs and route diagnostics, with sparse pair costs loaded separately.
    """
    supports_assignment_statistics = True
    supports_cost_perturbation = True
    supports_fp64_accumulation = True
    index_hash_arithmetic = "fp32_remainder_v1"

    def __init__(self):
        """
        CN: 初始化诊断，不加载已删除的旧扫描扩展。
        EN: Initialize diagnostics without loading retired scan extensions.
        """
        self.routes = Counter()
        self.last_backend = "generic"

    def _record(self, name):
        """
        CN: 记录实际入口，不参与实现选择。
        EN: Record the actual entry point without selecting an implementation.
        """
        key = name + ":generic"
        self.routes[key] += 1
        _ROUTES[key] += 1

    def fused_gcost_pairs(self, *args):
        """
        CN: 稀疏点对保留独立CUDA核，不为点对计算执行完整扫描。
        EN: Retain the separate sparse-pair CUDA kernel without running a full scan for selected pairs.
        """
        return load_native_module("hello_ot_native.hierot_norm_cost_scan_ext").fused_gcost_pairs(*args)

    def raw_lowrank_stats_only(self, x, y, nx, ny, f, g, query_tile, db_tile, dot_scale):
        """
        CN: 只归约完整证书，不生成候选；返回顺序与调用方一致。
        EN: Reduce the complete certificate without candidates, preserving caller statistic ordering.
        """
        if query_tile <= 0 or db_tile <= 0:
            raise ValueError("tile sizes must be positive")
        self._record("raw_certificate")
        return _raw_certificate_stats(x,y,nx,ny,f,g,dot_scale=dot_scale,caller_layout=True)

    def fused_directional_topk_raw(self, query_feat, db_feat, query_bias, db_bias, query_tile, db_tile, k, dot_scale):
        """
        CN: 单向兼容名称直接复用统一initial选择。
        EN: Forward the directional compatibility name to unified initial selection.
        """
        self._record("raw_directional")
        return self.initialization_directional_topk_raw(query_feat,db_feat,query_bias,db_bias,query_tile,db_tile,k,dot_scale)

    def fused_gcost_topk(self, query, db, dual_db, k, cost_type,
                         query_global_index=None, database_global_index=None, sigma=0., seed=0,
                         perturb_mode=1, row_is_source=True, index_hash_coeffs=None,
                         compute_precision=0, collect_statistics=False):
        """
        CN: 返回升序cost减dual，支持两种精度、扰动和完整固定统计。
        EN: Return ascending cost minus dual with both precisions, perturbations and complete fixed statistics.
        """
        self._record("norm_initial")
        p=(query_global_index,database_global_index,sigma,seed,perturb_mode,row_is_source,index_hash_coeffs)
        return list(directional(query,db,dual_db,k,cost_type,minimum=True,perturbation=p,
                                collect_statistics=collect_statistics,compute_precision=compute_precision))

    def fused_gcost_bidir_certificate(self, query, db, query_dual, db_dual, k, cost_type,
                                      query_global_index=None, database_global_index=None, sigma=0., seed=0,
                                      perturb_mode=1, row_is_source=True, index_hash_coeffs=None,
                                      compute_precision=0, rows_per_block_hint=8):
        """
        CN: 一次扫描返回双向候选与完整统计，原几何提示不覆盖统一规划。
        EN: Return bidirectional candidates and complete statistics in one scan; the former geometry hint does not override shared planning.
        """
        self._record("norm_pricing")
        p=(query_global_index,database_global_index,sigma,seed,perturb_mode,row_is_source,index_hash_coeffs)
        return list(bidirectional(query,db,query_dual,db_dual,k,cost_type,perturbation=p,compute_precision=compute_precision))

    def fused_gcost_certificate(self, query, db, query_dual, db_dual, cost_type,
                                query_global_index=None, database_global_index=None, sigma=0., seed=0,
                                perturb_mode=1, row_is_source=True, index_hash_coeffs=None,
                                compute_denominator=1, compute_precision=0):
        """
        CN: 只计算完整证书，保留分母开关和六参数便利入口。
        EN: Compute only the complete certificate, retaining the denominator switch and six-argument convenience entry.
        """
        if isinstance(query_global_index,(int,bool)) and database_global_index is None:
            compute_denominator,query_global_index=query_global_index,None
        self._record("norm_certificate")
        p=(query_global_index,database_global_index,sigma,seed,perturb_mode,row_is_source,index_hash_coeffs)
        return list(certificate_only(query,db,query_dual,db_dual,cost_type,perturbation=p,
                                     compute_denominator=bool(compute_denominator),compute_precision=compute_precision))

    def initialization_directional_topk_raw(self, query_feat, db_feat, query_bias, db_bias, query_tile, db_tile, k, dot_scale):
        """
        CN: 两种精度共用调用方分块，保留一般偏置与缩放。
        EN: Share caller tiles across both precisions, retaining general biases and scaling.
        """
        x,y,qb,dbb=query_feat,db_feat,query_bias,db_bias
        if x.dtype != y.dtype:
            raise RuntimeError("query/db feature dtypes must match")
        _check(x,y,k,(qb,len(x)),(dbb,len(y)))
        if query_tile<=0 or db_tile<=0 or dot_scale not in (1.,2.):
            raise ValueError("invalid raw directional tile or scale")
        self._record("raw_initial")
        if all(value.is_contiguous() for value in (x,y,qb,dbb)):
            return _module("cost").raw_initial(x,y,qb,dbb,k,query_tile,db_tile,dot_scale)
        values,ids=directional(x,y,dbb,k,3,query_bias=qb,dot_scale=dot_scale)
        return [values,ids.to(torch.int32)]

    def fused_bidir_topk_stats_raw(self, query_feat, db_feat, query_cost, db_cost, query_bias, db_bias,
                                  query_tile, db_tile, col_reduce_tile, k, compute_stats,
                                  collect_linf_stats, collect_timing, col_kernel, dot_scale):
        """
        CN: 保留原FP64成本和score结合顺序，直接执行调用方计划；退役计时槽返回零。
        EN: Preserve FP64 cost/score association and consume caller plans directly, returning zeros in the retired timing slot.
        """
        x,y,nx,ny,qb,dbb=query_feat,db_feat,query_cost,db_cost,query_bias,db_bias
        if x.dtype != y.dtype:
            raise RuntimeError("query/db feature dtypes must match")
        _check(x,y,k,(nx,len(x)),(ny,len(y)),(qb,len(x)),(dbb,len(y)))
        if (query_tile<=0 or db_tile<=0 or not 0<col_reduce_tile<=256 or dot_scale not in (1.,2.)
                or col_kernel not in (0,1,2) or (col_kernel==1 and k!=4) or (col_kernel==2 and k!=2)):
            raise ValueError("invalid raw pricing tile, scale or column mode")
        self._record("raw_pricing")
        if k<=2 and all(value.is_contiguous() for value in (x,y,nx,ny,qb,dbb)):
            return _module("pricing_small").raw_scan_tiled(x,y,nx,ny,qb,dbb,k,dot_scale,compute_stats,collect_linf_stats,query_tile,db_tile)
        r=bidirectional(x,y,qb,dbb,k,3,raw_costs=(nx,ny),raw_biases=(qb,dbb),dot_scale=dot_scale,planned_tiles=(query_tile,db_tile))
        stats=torch.stack((r[4],r[5],r[8].double(),r[7] if collect_linf_stats else r[7].new_zeros(())))
        if not compute_stats:stats.zero_()
        return [r[0],r[1].to(torch.int32),r[2],r[3].to(torch.int32),stats,torch.zeros(4,dtype=torch.float64,device="cpu")]
