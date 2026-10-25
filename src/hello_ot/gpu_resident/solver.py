from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import math
import time

import numpy as np
import torch

from hello_ot.config import level_cost_precision, level_stopping_norm
from hello_ot.hierarchy.plan import _make_shuffle_once_permutation
from hello_ot.perturbation import dual_feasibility_jump_diagnostics
from hello_ot.refinement.support_budget import active_support_budget
from hello_ot.types import (
    ActiveSupportStats, BudgetedPruningStats, CoarsestSolveStats, ConvergenceStats,
    CostStageResult, DualViolationStats, HelloSolveError, InitializationStats,
    IterationStats, LevelResult, RefinementResult, RefinementSummary, Result,
    SolveLPStats, SolveStageResult, SparseOTSolution, TraceResult,
)

from .cost import Cost, offsets, scan
from .storage import Transfers, reordered
from .support import Support, normalize_masses


def _candidates(xv, xi, yv, yi, dual, n, m, positive=False):
    device = dual.device
    rows = torch.arange(n, device=device)[:, None].expand_as(xi)
    cols = torch.arange(m, device=device)[:, None].expand_as(yi)
    xmask, ymask = xi >= 0, yi >= 0
    if positive:
        xmask &= xv + dual[:n, None] > 0
        ymask &= yv + dual[n:, None] > 0
    return torch.unique(torch.cat(((rows * m + xi)[xmask], (yi * m + cols)[ymask])), sorted=True)


def _initialize(cost, a, b, previous, known_side, config):
    started = time.perf_counter()
    n, m, device = len(a), len(b), a.device
    dual = torch.zeros(n + m, device=device, dtype=torch.float64)
    if known_side == "target":
        dual[n:] = previous[-m:]
    elif known_side == "source":
        dual[:n] = previous[:n]
    else:
        dual = previous.clone()
    xv, xi, yv, yi, _ = scan(cost, dual, config.assignment_topk)
    propagation = time.perf_counter() - started
    if known_side == "target":
        dual[:n] = -xv[:, 0]
        _, _, yv, yi, _ = scan(cost, dual, config.assignment_topk)
    elif known_side == "source":
        dual[n:] = -yv[:, 0]
        xv, xi, _, _, _ = scan(cost, dual, config.assignment_topk)
    candidates = _candidates(xv, xi, yv, yi, dual, n, m)
    assignment_time = time.perf_counter() - started
    support = Support.initial(candidates, a, b)
    stats = InitializationStats(time.perf_counter() - started, propagation if known_side else 0.0,
        assignment_time, time.perf_counter() - started - assignment_time, known_side,
        config.assignment_topk, n * min(m, config.assignment_topk), len(candidates),
        len(support.keys) - len(candidates), len(support.keys))
    return support, dual, stats


def _refine(
    cost,
    a,
    b,
    previous,
    known_side,
    config,
    level_index,
    transfers,
    detect,
    proposal=None,
    jump_baseline=None,
):
    """
    CN: 完成单层 refinement；可在成本切换首轮检测对偶可行性跳变。
    EN: Refine one level, optionally probing the first post-switch dual-feasibility jump.
    """
    started = time.perf_counter()
    support, dual, initialization = _initialize(proposal or cost, a, b, previous, known_side, config)
    norm = level_stopping_norm(config.stopping_norm, level_index=level_index)
    budget = active_support_budget(config.support_budget_factor, len(a), len(b))
    peak = len(support.keys)
    previous_pruned, reentered = None, False
    history = []
    jump_diagnostics = None
    for index in range(config.max_iterations):
        iteration_started = time.perf_counter()
        before = len(support.keys)
        lp_started = time.perf_counter()
        dual, lp = support.solve_lp(cost, a, b, dual, config, norm, transfers)
        lp_seconds = time.perf_counter() - lp_started
        scan_started = time.perf_counter()
        xv, xi, yv, yi, metrics = scan(cost, dual, config.pricing_topk, certificate=True)
        scan_seconds = time.perf_counter() - scan_started
        converged = metrics[norm] <= config.dual_feasibility_tolerance
        trigger = not converged and detect and reentered
        if jump_baseline is not None and index == 0:
            jump_diagnostics = dual_feasibility_jump_diagnostics(
                jump_baseline,
                metrics[norm],
            )
        restart = bool(
            jump_diagnostics
            and jump_diagnostics["restarted_from_perturbed_coarsest"]
        )
        after, added_count, prune_seconds = before, 0, 0.0
        if not converged and not trigger and not restart:
            candidates = _candidates(xv, xi, yv, yi, dual, len(a), len(b), True)
            fresh = support.extend(candidates)
            added = support.keys[fresh]
            added_count, after = len(added), len(support.keys)
            peak = max(peak, after)
            reentered = previous_pruned is not None and bool(torch.isin(added, previous_pruned).any())
            prune_started = time.perf_counter()
            previous_pruned = support.prune(cost, dual, fresh, budget)
            prune_seconds = time.perf_counter() - prune_started
            if not added_count:
                raise HelloSolveError("GPU full optimality check failed without new candidate edges")
        history.append(IterationStats(index, float(lp.obj_val), time.perf_counter() - iteration_started, 0.0,
            scan_seconds, SolveLPStats(lp_seconds, lp.iterations, lp.primal_feas, lp.gap, norm,
                lp.termination_reason, lp.primal_feas, lp.dual_feas, lp.dual_feas),
            DualViolationStats(scan_seconds, metrics["count"], added_count),
            BudgetedPruningStats(prune_seconds, after, len(support.keys), budget, after > budget),
            ActiveSupportStats(before, after, len(support.keys), peak),
            ConvergenceStats(metrics[norm], config.dual_feasibility_tolerance, converged,
                metrics["l2"], metrics["linf"], metrics["max"], norm)))
        if config.verbose == "detailed":
            import sys
            print(f"[GPU HELLO] level={level_index} iteration={index} support={before} dual_{norm}={metrics[norm]:.3e}", file=sys.stderr)
        if converged or trigger or restart:
            stop_reason = (
                "converged" if converged else
                "dual_feasibility_jump_restart" if restart else
                "cost_perturbation_activated"
            )
            summary = RefinementSummary(time.perf_counter() - started, len(history), float(lp.obj_val),
                len(support.keys), peak, stop_reason, converged)
            level = LevelResult(level_index, "refined", len(a), len(b), initialization,
                RefinementResult(tuple(history), summary), time.perf_counter() - started)
            return support, dual, level, trigger, jump_diagnostics, restart
    summary = RefinementSummary(
        time.perf_counter() - started,
        len(history),
        float(lp.obj_val),
        len(support.keys),
        peak,
        "max_refinement_iterations",
        False,
    )
    level = LevelResult(
        level_index,
        "refined",
        len(a),
        len(b),
        initialization,
        RefinementResult(tuple(history), summary),
        time.perf_counter() - started,
    )
    return support, dual, level, False, jump_diagnostics, False


def _leaf(cost, a, b, level_index, transfers):
    from hello_ot.hierarchy.coarsest import _get_pot_module

    started = time.perf_counter()
    n, m = len(a), len(b)
    matrix = np.empty((n, m), dtype=np.float64)
    q = min(128, n)
    block = max(1, min(m, cost.workspace_bytes // max(1, q * cost.source.shape[1] * 24)))
    for i in range(0, n, q):
        for j in range(0, m, block):
            tile = cost.tile(i, j, q, block)
            matrix[i:i + tile.shape[0], j:j + tile.shape[1]] = transfers.host(tile, "leaf_pot")
    plan, log = _get_pot_module().emd(transfers.host(a, "leaf_pot"), transfers.host(b, "leaf_pot"), matrix,
                                    log=True, numItermax=10_000_000)
    if log.get("warning"):
        raise HelloSolveError(f"POT leaf failed: {log['warning']}")
    rr, cc = np.nonzero(plan > 1e-12)
    keys = transfers.move(rr * m + cc, a.device, "leaf_pot")
    values = transfers.move(np.asarray(plan[rr, cc], dtype=np.float64), a.device, "leaf_pot")
    dual = transfers.move(np.concatenate((log["u"], log["v"])), a.device, "leaf_pot")
    support = Support(keys, values, torch.zeros(len(keys), dtype=torch.bool, device=a.device), n, m)
    elapsed = time.perf_counter() - started
    return support, dual, LevelResult(level_index, "coarsest", n, m, None,
        CoarsestSolveStats(elapsed, float(log["cost"]), len(keys)), elapsed)


def _perturb(cost, config, source_perm, target_perm, transfers):
    seed = config.random_seed & 0xFFFFFFFF
    for char in "full_cost_perturb":
        seed = (seed * 1664525 + ord(char) + 1013904223) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    rr = np.argsort(source_perm)[rng.integers(len(source_perm), size=65536)]
    cc = np.argsort(target_perm)[rng.integers(len(target_perm), size=65536)]
    sample_cost = replace(cost, precision="fp64" if config.cost_precision in {"fp64", "finest_fp64"} else "fp32")
    mean = float(sample_cost.pairs(transfers.move(rr, cost.source.device, "initialization"),
                                  transfers.move(cc, cost.source.device, "initialization")).mean())
    sigma = config.cost_perturbation_relative_scale * mean
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("cost perturbation requires a positive finite sampled cost mean")
    result = replace(cost, sigma=sigma, seed=seed)
    if cost.kind == "l2^2":
        scale = math.sqrt(0.25 * sigma)
        sa = cost.source_index.double() * 12.9898 + seed * 0.001
        ta = cost.target_index.double() * 78.233
        result.source_extra = (scale * torch.stack((sa.sin(), sa.cos()), 1)).float()
        result.target_extra = (-scale * torch.stack((ta.cos(), ta.sin()), 1)).float()
    return result, {"mode": "rank2" if cost.kind == "l2^2" else "index_hash", "sigma": sigma,
        "sampled_mean": mean, "sample_pairs": 65536, "seed": seed,
        "relative_scale": config.cost_perturbation_relative_scale}


@torch.no_grad()
def solve(problem, config):
    """
    CN: 保持 GPU 数组贯穿 HELLO；只有 POT leaf 和 support sparsification 可以导出 CPU 数组。
    EN: Retain arrays through HELLO on GPU; only POT leaves and support sparsification may export CPU arrays.
    """
    from hello_ot._native_compat import require_native_runtime_compatibility
    from hello_ot._internal.instrumentation.memory_accounting import use_solve_memory_tracker
    from hello_ot._internal.trace import _ChromeTraceCollector
    from hello_ot.progress import report_header, report_final

    x, y = problem.source_points, problem.target_points
    if not all(torch.is_tensor(v) and v.is_cuda for v in (x, y)) or x.device != y.device:
        raise ValueError("gpu_resident requires source and target CUDA tensors on the same device")
    if any(v.dtype != torch.float32 or not v.is_contiguous() for v in (x, y)):
        raise ValueError("gpu_resident requires contiguous FP32 input features")
    if config.stopping_norm == "finest_linf" and problem.cost_type != "l2^2":
        raise ValueError("finest_linf currently requires l2^2")
    require_native_runtime_compatibility()
    from hello_ot_native import pycupdlpx
    if getattr(pycupdlpx, "GPU_ARRAY_IO_VERSION", 0) < 1:
        raise RuntimeError("GPU pipeline requires rebuilt CuPDLPx extensions with GPU array I/O")
    device = x.device
    n, m = len(x), len(y)
    source_perm, target_perm = _make_shuffle_once_permutation(n_source=n, n_target=m,
        node_seed=config.random_seed, split_count=config.split_count)
    tracer = _ChromeTraceCollector(enabled=True) if config.record_trace else None
    transfers = Transfers(device)
    memory_context = use_solve_memory_tracker(device) if config.profile_memory else nullcontext(None)
    with torch.cuda.device(device), memory_context as memory_tracker:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if memory_tracker is not None:
            memory_tracker.start()
        report_header(config.verbose, backend="native", device=str(device), cost=problem.cost_type,
            n_source=n, n_target=m, dimension=x.shape[1], perturbation=config.cost_perturbation)
        si = transfers.move(source_perm, device, "initialization")
        ti = transfers.move(target_perm, device, "initialization")
        a = transfers.move(problem.source_mass, device, "initialization").double()[si]
        b = transfers.move(problem.target_mass, device, "initialization").double()[ti]
        sizes = [(n, m)]
        while max(sizes[-1]) > config.coarsest_size_threshold:
            nn, mm = sizes[-1]
            sizes.append(((nn + config.split_count - 1) // config.split_count, mm) if nn >= mm
                         else (nn, (mm + config.split_count - 1) // config.split_count))
        stages, levels, perturbation_metadata = [], [], {}
        activated = config.cost_perturbation == "on"
        phase = "perturbed" if activated else "original"
        with reordered(x, y, source_perm, target_perm, transfers, config.consume_input_features) as (sx, ty):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            workspace = int(min(config.fused_scan_memory_floor_mib * 1024**2, free_bytes // 8, 128 * 1024**2))
            if workspace < 1024**2:
                raise torch.cuda.OutOfMemoryError("Insufficient memory for GPU HELLO scan workspace")
            original = Cost(sx, ty, problem.cost_type, offsets(sx), offsets(ty), si, ti, workspace_bytes=workspace)
            current = original
            if activated:
                current, perturbation_metadata = _perturb(original, config, source_perm, target_perm, transfers)
            phase_started = time.perf_counter()
            previous = None
            for level_index in reversed(range(len(sizes))):
                nn, mm = sizes[level_index]
                cost = current.sliced(nn, mm, level_cost_precision(config.cost_precision, level_index=level_index))
                aa, bb = normalize_masses(a[:nn], b[:mm])
                span = tracer.span("gpu_resident.level", "hierarchy", args={"level": level_index}) if tracer else nullcontext()
                with span:
                    if previous is None:
                        support, dual, level = _leaf(cost, aa, bb, level_index, transfers)
                    else:
                        known = "target" if nn >= mm else "source"
                        support, dual, level, trigger, _, _ = _refine(cost, aa, bb, previous, known, config,
                            level_index, transfers, config.cost_perturbation == "auto" and not activated)
                        if trigger:
                            before_switch = level.solve.iterations[-1].convergence.dual_feasibility
                            levels.append(level)
                            stages.append(CostStageResult(phase, SolveStageResult(tuple(levels), time.perf_counter() - phase_started)))
                            levels, phase_started, phase = [], time.perf_counter(), "perturbed"
                            current, perturbation_metadata = _perturb(original, config, source_perm, target_perm, transfers)
                            activated = True
                            cost = current.sliced(nn, mm, level_cost_precision(config.cost_precision, level_index=level_index))
                            support, dual, level, _, jump, restart = _refine(
                                cost,
                                aa,
                                bb,
                                dual,
                                None,
                                config,
                                level_index,
                                transfers,
                                False,
                                jump_baseline=before_switch,
                            )
                            perturbation_metadata.update(
                                trigger_level=level_index,
                                trigger_iteration=len(stages[-1].solve.levels[-1].solve.iterations),
                                reentry_count=1,
                                **jump,
                                restart_trigger_level=level_index if restart else None,
                            )
                            if restart:
                                levels.append(level)
                                previous = None
                                for restart_level_index in reversed(range(len(sizes))):
                                    restart_n, restart_m = sizes[restart_level_index]
                                    restart_cost = current.sliced(
                                        restart_n,
                                        restart_m,
                                        level_cost_precision(
                                            config.cost_precision,
                                            level_index=restart_level_index,
                                        ),
                                    )
                                    restart_a, restart_b = normalize_masses(
                                        a[:restart_n],
                                        b[:restart_m],
                                    )
                                    if previous is None:
                                        support, dual, restart_level = _leaf(
                                            restart_cost,
                                            restart_a,
                                            restart_b,
                                            restart_level_index,
                                            transfers,
                                        )
                                    else:
                                        restart_known = (
                                            "target" if restart_n >= restart_m else "source"
                                        )
                                        support, dual, restart_level, _, _, _ = _refine(
                                            restart_cost,
                                            restart_a,
                                            restart_b,
                                            previous,
                                            restart_known,
                                            config,
                                            restart_level_index,
                                            transfers,
                                            False,
                                        )
                                    levels.append(restart_level)
                                    previous = dual
                                break
                    levels.append(level)
                    previous = dual
            if activated:
                stages.append(CostStageResult(phase, SolveStageResult(tuple(levels), time.perf_counter() - phase_started)))
                phase_started = time.perf_counter()
                aa, bb = normalize_masses(a, b)
                support, dual, level, _, _, _ = _refine(original.sliced(n, m, level_cost_precision(config.cost_precision, level_index=0)),
                    aa, bb, dual, None, config, 0, transfers, False,
                    proposal=current.sliced(n, m, level_cost_precision(config.cost_precision, level_index=0)))
                levels, phase = [level], "original_final"
            stages.append(CostStageResult(phase, SolveStageResult(tuple(levels), time.perf_counter() - phase_started)))
            rows, cols = support.rows_cols()
            mapped_keys = si[rows] * m + ti[cols]
            order = torch.argsort(mapped_keys, stable=True)
            f, g = torch.empty_like(dual[:n]), torch.empty_like(dual[n:])
            f[si], g[ti] = dual[:n], dual[n:]
            solution = SparseOTSolution((n, m), si[rows][order], ti[cols][order], support.values[order], f, g)
            objective = float(original.sliced(n, m, level_cost_precision(config.cost_precision, level_index=0)).pairs(rows, cols) @ support.values)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        memory = memory_tracker.finish() if memory_tracker else None
    lp_seconds = sum(i.solve_lp.wall_time for s in stages for l in s.solve.levels
                     if isinstance(l.solve, RefinementResult) for i in l.solve.iterations)
    metadata = {"gpu_resident": True, "backend": "native", "device": str(device), "cost_type": problem.cost_type,
        "cost_precision": config.cost_precision, "consume_input_features": config.consume_input_features,
        "stopping_norm": config.stopping_norm, "assignment_mode": "nodewise_bidirectional",
        "assignment_topk": config.assignment_topk, "coarsest_solver": "pot", "transfers": transfers.records,
        "transfer_wall_time": sum(r["seconds"] for r in transfers.records.values()),
        "total_lp_solve_time": lp_seconds, "cost_perturbation": {"policy": config.cost_perturbation, "activated": activated, **perturbation_metadata}}
    metadata["total_refinement_iterations"] = sum(len(l.solve.iterations) for s in stages
        for l in s.solve.levels if isinstance(l.solve, RefinementResult))
    if memory is not None:
        metadata["memory_profile"] = memory
    result = Result(objective, solution, stages[-1].solve, wall,
        None if memory is None else memory["peak_gpu_memory_mib"],
        None if tracer is None else TraceResult(tracer.export()), metadata, tuple(stages))
    report_final(config.verbose, result)
    return result
