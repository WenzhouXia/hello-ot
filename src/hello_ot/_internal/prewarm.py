from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

def prewarm_pot_backend() -> dict[str, Any]:
    """
    CN: 在正式计时前加载 coarsest solve 使用的 POT backend。
    EN: Load the POT backend used by the coarsest solve before measured execution.
    """
    start = time.perf_counter()
    info: dict[str, Any] = {"enabled": True, "success": False}
    try:
        from hello_ot.hierarchy.coarsest import _get_pot_module

        module = _get_pot_module()
        info.update({"success": True, "error": None, "module": str(module.__name__)})
        return info
    except Exception as exc:  # noqa: BLE001
        info.update({"success": False, "error": repr(exc)})
        print(f"[Profile][POTBackendPrewarm] warning: failed: {exc}", flush=True)
        return info
    finally:
        info["time"] = float(time.perf_counter() - start)
        if bool(info.get("success", False)):
            print(f"[Profile][POTBackendPrewarm] time={float(info['time']):.2f}s", flush=True)


def prewarm_hierarchy_cold_paths(
    *,
    source_f: np.ndarray,
    solver_engine: str = "cupdlpx",
    dual_assignment_pipeline: str = "gpu",
    gpu_id: int = 0,
    variable_bound_mode: str = "explicit",
    matrix_value_mode: str = "explicit",
    vector_sum_mode: str = "resident_ones",
) -> dict[str, Any]:
    """CN: 在正式 hierarchy solve 计时前触发常见的首次运行冷启动路径。
    EN: Trigger common first-use cold paths before measured hierarchy-solve timing.
    """
    start_total = time.perf_counter()
    info: dict[str, Any] = {"enabled": True, "success": True, "stages": {}}

    def _run_stage(name: str, fn: Any) -> None:
        stage_start = time.perf_counter()
        stage_info: dict[str, Any] = {"success": False}
        try:
            extra = fn()
            if isinstance(extra, dict):
                stage_info.update(extra)
            stage_info.update({"success": True, "error": None})
        except Exception as exc:  # noqa: BLE001
            stage_info.update({"success": False, "error": repr(exc)})
            info["success"] = False
            print(f"[Profile][HierarchyColdPathPrewarm] warning: {name} failed: {exc}", flush=True)
        finally:
            stage_info["time"] = float(time.perf_counter() - stage_start)
            info["stages"][name] = stage_info

    def _prewarm_numba_support() -> dict[str, Any]:
        from hello_ot._internal.core.solver import _northwest_corner_numba
        nw_rows, nw_cols = _northwest_corner_numba(
            np.asarray([0.5, 0.5], dtype=np.float64),
            np.asarray([0.25, 0.75], dtype=np.float64),
        )
        return {
            "northwest_edges": int(np.asarray(nw_rows).size),
            "northwest_cols": int(np.asarray(nw_cols).size),
        }

    def _prewarm_torch_cuda_ops() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA is not available"}
        from hello_ot.initialization.state import _unique_merge_rows_cols_vals

        device = torch.device(f"cuda:{int(gpu_id)}")
        n_rows = 1024
        n_target = 4096
        n_keys = 65_536
        src_idx = torch.arange(0, n_rows, device=device, dtype=torch.int64)
        tgt_idx = torch.arange(0, n_target, device=device, dtype=torch.int64)
        rows_local = torch.arange(0, n_keys, device=device, dtype=torch.int64) % n_rows
        cols_local = torch.arange(0, n_keys, device=device, dtype=torch.int64) % n_target
        vals_local = torch.ones(n_keys, device=device, dtype=torch.float32)
        mapped_rows = src_idx.index_select(0, rows_local)
        mapped_cols = tgt_idx.index_select(0, cols_local)
        keys = mapped_rows * n_target + mapped_cols
        uniq_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        merged = torch.zeros(int(uniq_keys.numel()), dtype=torch.float32, device=device)
        merged.scatter_add_(0, inverse, vals_local)
        _ = torch.isin(uniq_keys, keys)
        active_keys = torch.arange(0, 51_200, device=device, dtype=torch.int64)
        northwest_keys = torch.arange(0, 5_120, device=device, dtype=torch.int64) * 8
        _ = torch.isin(active_keys, northwest_keys)
        _ = torch.isin(northwest_keys, active_keys)
        merged_rows, _merged_cols, _merged_vals, merge_backend = _unique_merge_rows_cols_vals(
            mapped_rows.to(dtype=torch.int32),
            mapped_cols.to(dtype=torch.int32),
            vals_local,
            n_target=n_target,
            tracer=None,
            trace_args=None,
            trace_prefix="prewarm.unique_merge",
        )
        torch.cuda.synchronize(device)
        return {
            "device": str(device),
            "unique_keys": int(uniq_keys.numel()),
            "merged_rows": int(np.asarray(merged_rows).size),
            "merge_backend": str(merge_backend),
        }

    def _prewarm_sddmm_sampled_addmm() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA is not available"}

        device = torch.device(f"cuda:{int(gpu_id)}")
        dim = int(np.asarray(source_f).shape[1]) if np.asarray(source_f).ndim == 2 else 4
        block_rows = 1024
        n_target = 4096
        nnz = 65_536
        counts = torch.full((block_rows,), nnz // block_rows, device=device, dtype=torch.int64)
        counts[: int(nnz % block_rows)] += 1
        crow = torch.empty(block_rows + 1, device=device, dtype=torch.int64)
        crow[0] = 0
        torch.cumsum(counts, dim=0, out=crow[1:])
        col = torch.arange(nnz, device=device, dtype=torch.int64) % n_target
        f_block = torch.zeros((block_rows, dim), device=device, dtype=torch.float32)
        g_full = torch.zeros((n_target, dim), device=device, dtype=torch.float32)
        a_block = torch.zeros(block_rows, device=device, dtype=torch.float32)
        b_full = torch.zeros(n_target, device=device, dtype=torch.float32)
        local_rows = torch.repeat_interleave(
            torch.arange(block_rows, dtype=torch.int64, device=device),
            crow[1:] - crow[:-1],
        )
        vals0 = a_block.index_select(0, local_rows) + b_full.index_select(0, col)
        with torch.no_grad():
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
                mask = torch.sparse_csr_tensor(crow, col, vals0, size=(block_rows, n_target), device=device)
            out_sp = torch.sparse.sampled_addmm(mask, f_block, g_full.t(), beta=1.0, alpha=-1.0)
        value_count = int(out_sp.values().numel())
        torch.cuda.synchronize(device)
        return {
            "device": str(device),
            "dim": int(dim),
            "block_rows": int(block_rows),
            "n_target": int(n_target),
            "nnz": int(nnz),
            "value_count": int(value_count),
        }

    def _prewarm_assignment_topk() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA is not available"}
        from hello_ot.initialization.state import _topk_ip_indices_streamed_cuda

        dim = int(np.asarray(source_f).shape[1]) if np.asarray(source_f).ndim == 2 else 2
        device = torch.device(f"cuda:{int(gpu_id)}")
        query = np.zeros((4, dim), dtype=np.float32)
        database = np.zeros((8, dim), dtype=np.float32)
        database_bias = np.zeros(8, dtype=np.float32)
        _, topk_idx, profile = _topk_ip_indices_streamed_cuda(
            int(query.shape[0]),
            lambda start, stop: torch.as_tensor(
                query[start:stop], dtype=torch.float32, device=device
            ).contiguous(),
            database,
            database_bias,
            top_k=1,
            device=device,
            tracer=None,
            trace_args=None,
            trace_prefix="prewarm.augment.topk",
        )
        torch.cuda.synchronize(device)
        return {
            "device": str(device),
            "feature_dim": int(dim),
            "topk_count": int(topk_idx.numel() if torch.is_tensor(topk_idx) else np.asarray(topk_idx).size),
            "backend": str((profile or {}).get("backend")),
        }

    def _prewarm_dual_feasibility_scan() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA is not available"}
        from hello_ot.refinement.stopping import lowrank_dual_feasibility_infeasibility

        dim = int(np.asarray(source_f).shape[1]) if np.asarray(source_f).ndim == 2 else 2
        f_mat = np.zeros((4, dim), dtype=np.float32)
        g_mat = np.zeros((8, dim), dtype=np.float32)
        source_cost = np.zeros(4, dtype=np.float32)
        target_cost = np.zeros(8, dtype=np.float32)
        dual_uv = np.zeros(12, dtype=np.float32)
        value = lowrank_dual_feasibility_infeasibility(
            f_mat,
            g_mat,
            source_cost,
            target_cost,
            dual_uv,
            gpu_id=int(gpu_id),
        )
        torch.cuda.synchronize()
        return {"dim": int(dim), "dual_feasibility": float(value)}

    def _prewarm_lp_matrix_build() -> dict[str, Any]:
        if str(dual_assignment_pipeline).lower() != "gpu":
            import scipy.sparse as sp

            data = np.ones(8, dtype=np.float32)
            row_indices = np.asarray([0, 2, 0, 3, 1, 2, 1, 3], dtype=np.int32)
            indptr = np.arange(0, 10, 2, dtype=np.int32)
            _ = sp.csc_matrix((data, row_indices, indptr), shape=(4, 4))
            return {"backend": "scipy_csc"}

        import torch

        from hello_ot._internal.lp.cupdlpx import CuPDLPxSolver
        from hello_ot.restricted_ot.backend import _build_device_ot_csr

        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA is not available"}
        n_source = 1024
        n_target = 1024
        diag = np.arange(n_source, dtype=np.int64)
        rows = np.concatenate([diag, diag])
        cols = np.concatenate([diag, (diag + 1) % n_target])
        c = ((rows * 17 + cols * 31) % 997).astype(np.float64) / 997.0
        masses_s = np.full(n_source, 1.0 / float(n_source), dtype=np.float64)
        masses_t = np.full(n_target, 1.0 / float(n_target), dtype=np.float64)
        engine = str(solver_engine).strip().lower()
        if engine != "cupdlpx":
            raise ValueError(f"Unsupported prewarm solver_engine: {engine}")
        device_data = _build_device_ot_csr(
            rows=rows,
            cols=cols,
            c=c,
            masses_s=masses_s,
            masses_t=masses_t,
            n_source=n_source,
            n_target=n_target,
            requested_device=f"cuda:{int(gpu_id)}",
            variable_bound_mode=str(variable_bound_mode),
            matrix_value_mode=str(matrix_value_mode),
            cupdlpx_python_pre_rescale=engine == "cupdlpx",
        )
        lp_solver = CuPDLPxSolver(cupdlpx_python_pre_rescale=True)
        result = lp_solver.solve(
            device_data["c"],
            None,
            device_data["rhs"],
            device_data["lb"],
            device_data["ub"],
            int(device_data["n_eqs"]),
            tolerance={"objective": 1e-4, "primal": 1e-4, "dual": 1e-4},
            verbose=0,
            device_problem_data=device_data,
            vector_sum_mode=str(vector_sum_mode),
            cupdlpx_python_pre_rescale=engine == "cupdlpx",
        )
        torch.cuda.synchronize(device_data["device"])
        return {
            "backend": "device_csr_solve",
            "solver_engine": engine,
            "device": str(device_data["device"]),
            "n_source": int(n_source),
            "n_target": int(n_target),
            "n_vars": int(device_data["n"]),
            "variable_bound_mode": str(variable_bound_mode),
            "matrix_value_mode": str(matrix_value_mode),
            "vector_sum_mode": str(vector_sum_mode),
            "iterations": int(result.iterations or 0),
            "termination_reason": result.termination_reason,
            "solver_runtime_sec": float(result.duration or 0.0),
            "solver_success": bool(result.success),
        }

    _run_stage("numba_support_and_northwest", _prewarm_numba_support)
    _run_stage("torch_cuda_stitch_primitives", _prewarm_torch_cuda_ops)
    _run_stage("sddmm_sampled_addmm", _prewarm_sddmm_sampled_addmm)
    _run_stage("assignment_topk", _prewarm_assignment_topk)
    _run_stage("dual_feasibility_scan", _prewarm_dual_feasibility_scan)
    _run_stage("lp_matrix_build", _prewarm_lp_matrix_build)

    info["time"] = float(time.perf_counter() - start_total)
    print(
        "[Profile][HierarchyColdPathPrewarm] "
        f"success={bool(info.get('success'))} time={float(info['time']):.2f}s",
        flush=True,
    )
    return info
