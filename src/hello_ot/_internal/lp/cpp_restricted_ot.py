from __future__ import annotations

import time

import numpy as np
import torch

from hello_ot._cpp_backend import load_cpp_extension
from .restricted_ot_pdlp import RestrictedOTPDLP
from .wrapper import SolverResult


def _array(value, dtype):
    """
    CN: 接收 CPU 数组或张量，保留可共享存储；拒绝隐式 GPU 到 CPU 搬运。
    EN: Accept CPU arrays or tensors and share storage when possible; reject implicit GPU-to-CPU transfers.
    """
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError("C++ restricted OT accepts CPU tensors only")
        value = value.detach().numpy()
    return np.ascontiguousarray(value, dtype=dtype).reshape(-1)


class CppRestrictedOTPDLP(RestrictedOTPDLP):
    """
    CN: 复用 Torch 重试协议的C++ Halpern-PDHG，实现相同 SolverResult 合约。
    EN: C++ Halpern-PDHG reusing the Torch retry protocol and SolverResult contract.
    """

    def __init__(self, *, num_threads=16, device="cpu", **kwargs):
        if torch.device(device).type != "cpu":
            raise ValueError("C++ restricted OT supports CPU only")
        super().__init__(device="cpu", **kwargs)
        self.num_threads = torch.get_num_threads() if num_threads is None else int(num_threads)
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        self.extension = load_cpp_extension("lp")

    def _solve_once(self, *, rows, cols, costs, source_mass, target_mass,
                    warm_start_primal, warm_start_dual, objective_tolerance,
                    primal_tolerance, dual_tolerance, scalar_rescaling, verbose):
        """
        CN: 保持参考算法的输入、缩放、检查点和返回格式，计入转换与索引构建。
        EN: Preserve reference inputs, scaling, checkpoints, and outputs, including conversion and index setup in timing.
        """
        started = time.perf_counter()
        rows, cols = _array(rows, np.int64), _array(cols, np.int64)
        costs = _array(costs, np.float64)
        a, b = _array(source_mass, np.float64), _array(target_mass, np.float64)
        if not a.size or not b.size:
            raise ValueError("both marginals must be nonempty")
        if abs(float(a.sum()) - float(b.sum())) > 1e-12 * max(1.0, float(np.max(np.abs(np.r_[a, b])))):
            raise ValueError("balanced restricted OT requires equal source and target total mass")
        x = np.zeros(costs.size) if warm_start_primal is None else _array(warm_start_primal, np.float64)
        y = np.zeros(a.size + b.size) if warm_start_dual is None else _array(warm_start_dual, np.float64)
        payload = self.extension.solve_once(
            rows, cols, costs, np.concatenate((a, b)), x, y, a.size, self.num_threads,
            self.evaluation_frequency, self.time_limit_seconds, self.iteration_limit,
            objective_tolerance, primal_tolerance, dual_tolerance, scalar_rescaling,
            self.record_state_trace,
        )
        payload["x"] = torch.from_numpy(payload["x"])
        payload["y"] = torch.from_numpy(payload["y"])
        payload["duration"] = time.perf_counter() - started
        payload["solver_diag"]["runtime_sec"] = payload["duration"]
        payload["solver_diag"]["extension_path"] = str(self.extension.__file__)
        self.last_diagnostics = payload["solver_diag"]
        if verbose:
            print(f"[CppRestrictedOTPDLP] termination={payload['termination_reason']} "
                  f"iterations={payload['iterations']} threads={self.num_threads}")
        return SolverResult(**payload)
