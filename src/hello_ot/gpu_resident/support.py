from __future__ import annotations

from dataclasses import dataclass

import torch

from hello_ot._internal.lp.cupdlpx import CuPDLPxSolver
from hello_ot._internal.instrumentation.memory_accounting import current_solve_memory_tracker
from hello_ot._internal.instrumentation.driver_memory import DriverMemoryTracker
from hello_ot.refinement.support_budget import POSITIVE_SUPPORT_TOLERANCE, primal_support_cap
from hello_ot.restricted_ot.support_sparsification import sparsify_transport_support
from hello_ot.restricted_ot.runtime import _termination_solver_params


def normalize_masses(a, b):
    """
    CN: 各层在 GPU 归一化边际并修正总质量舍入误差。
    EN: Normalize level marginals on GPU and correct total-mass roundoff.
    """
    a, b = a / a.sum(), b / b.sum()
    for _ in range(2):
        gap = a.sum() - b.sum()
        a[a.argmax()] -= gap.clamp_max(0)
        b[b.argmax()] += gap.clamp_min(0)
    return a, b


def northwest(a, b):
    """
    CN: 合并累计质量边界，在 GPU 构造含退化零边的 northwest-corner basis。
    EN: Merge cumulative marginal boundaries on GPU, retaining degenerate northwest-corner edges.
    """
    n, m, device = len(a), len(b), a.device
    boundaries = torch.cat((a.cumsum(0)[:-1], b.cumsum(0)[:-1]))
    order = torch.argsort(boundaries, stable=True)
    src = order < n - 1
    rows = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), src.long().cumsum(0)))
    cols = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), (~src).long().cumsum(0)))
    edges = torch.cat((torch.zeros(1, dtype=torch.float64, device=device), boundaries[order], a.sum().reshape(1)))
    return rows * m + cols, edges.diff().clamp_min(0)


@dataclass
class Support:
    """
    CN: GPU active support；所有稀疏索引、流量和保护标记留在设备上。
    EN: GPU active support retaining sparse indices, flow, and protection flags on device.
    """

    keys: torch.Tensor
    values: torch.Tensor
    structural: torch.Tensor
    n: int
    m: int

    @classmethod
    def initial(cls, candidates, a, b):
        nw, values = northwest(a, b)
        keys = torch.unique(torch.cat((candidates, nw)), sorted=True)
        x = torch.zeros(len(keys), dtype=torch.float64, device=a.device)
        structural = torch.zeros(len(keys), dtype=torch.bool, device=a.device)
        positions = torch.searchsorted(keys, nw)
        x.index_add_(0, positions, values)
        structural[positions] = True
        return cls(keys, x, structural, len(a), len(b))

    def rows_cols(self):
        return torch.div(self.keys, self.m, rounding_mode="floor"), self.keys.remainder(self.m)

    def extend(self, candidates):
        keys = torch.unique(torch.cat((self.keys, candidates)), sorted=True)
        old = torch.searchsorted(keys, self.keys)
        x = torch.zeros(len(keys), dtype=torch.float64, device=keys.device)
        structural = torch.zeros(len(keys), dtype=torch.bool, device=keys.device)
        fresh = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
        x[old], structural[old], fresh[old] = self.values, self.structural, False
        self.keys, self.values, self.structural = keys, x, structural
        return fresh

    def prune(self, cost, dual, fresh, budget):
        remove_count = max(0, len(self.keys) - budget)
        if not remove_count:
            return self.keys[:0]
        rows, cols = self.rows_cols()
        eligible = (~self.structural) & (~fresh) & (self.values.abs() <= POSITIVE_SUPPORT_TOLERANCE)
        candidates = torch.nonzero(eligible).flatten()
        if len(candidates) < remove_count:
            raise RuntimeError("GPU budgeted pruning has insufficient old zero-flow edges")
        reduced = cost.pairs(rows[candidates], cols[candidates]) - dual[rows[candidates]] - dual[self.n + cols[candidates]]
        removed = candidates[torch.argsort(reduced, descending=True, stable=True)[:remove_count]]
        removed_keys = self.keys[removed]
        keep = torch.ones(len(self.keys), device=self.keys.device, dtype=torch.bool)
        keep[removed] = False
        self.keys, self.values, self.structural = self.keys[keep], self.values[keep], self.structural[keep]
        return removed_keys

    def solve_lp(self, cost, a, b, dual, config, norm, transfers):
        rows, cols = self.rows_cols()
        edges, vertices = len(self.keys), self.n + self.m
        if edges > (2**31 - 1) // 2 or vertices >= 2**31:
            raise ValueError("GPU restricted LP exceeds CuPDLPx int32 capacity")
        counts = torch.cat((torch.bincount(rows, minlength=self.n), torch.bincount(cols, minlength=self.m)))
        rowptr = torch.cat((torch.zeros(1, device=rows.device, dtype=torch.int64), counts.cumsum(0))).int()
        columns = torch.cat((torch.arange(edges, device=rows.device), torch.argsort(cols, stable=True))).int()
        costs = cost.pairs(rows, cols)
        rhs = torch.cat((a, b))
        payload = dict(row_ptr=rowptr, col_ind=columns, values=torch.ones(2 * edges, device=rows.device, dtype=torch.float64),
                       c=costs, rhs=rhs, lb=0.0, ub=float("inf"), m=vertices, n=edges, n_eqs=vertices,
                       variable_bound_mode="constant", matrix_value_mode="implicit_aty", device=rows.device)
        tracker = current_solve_memory_tracker()
        entry = tracker.begin_lp(metadata={"support_size": edges}) if tracker else None
        driver = DriverMemoryTracker(device=rows.device.index, enabled=tracker is not None)
        result = CuPDLPxSolver().solve(costs, None, rhs, 0.0, float("inf"), vertices,
            warm_start_primal=self.values, warm_start_dual=dual, device_problem_data=payload,
            trace_args={"lp_entry_used_mem_mib": driver.current_mib},
            solver_params=_termination_solver_params({"return_device": "cuda", "vector_sum_mode": "direct_reduce",
                           "memory_accounting_component": "gpu_resident_lp"}, norm), verbose=0)
        if tracker:
            tracker.finish_lp(entry, delta_peak_mib=result.solver_diag.get("lp_backend_delta_peak_mem_mib"))
        if not result.success:
            raise RuntimeError(f"GPU restricted LP failed: {result.termination_reason}")
        self.values = result.x
        dual = result.y
        cap = primal_support_cap(config.support_budget_factor, config.pricing_topk, self.n, self.m)
        if int(torch.count_nonzero(self.values > POSITIVE_SUPPORT_TOLERANCE)) > cap:
            reason = "support_sparsification"
            sparse = sparsify_transport_support(transfers.host(rows, reason), transfers.host(cols, reason),
                transfers.host(self.values, reason), transfers.host(costs, reason),
                n_source=self.n, n_target=self.m, target_nnz=cap)
            self.values = transfers.move(sparse.values, rows.device, reason)
            result.x = self.values
            result.obj_val = float(costs @ self.values)
            shift = ((rhs @ dual - result.obj_val) / a.sum()).clamp_min(0)
            dual = dual - 0.5 * shift
            result.y = dual
            result.dual_obj_val = float(rhs @ dual)
            result.gap = abs(result.obj_val - result.dual_obj_val) / (1 + abs(result.obj_val) + abs(result.dual_obj_val))
        return dual, result
