from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch

from .wrapper import SolverResult


class RestrictedOTPDLP(ABC):
    """
    CN: Torch 与 C++ restricted OT 共用的配置、终止合约和数值重试协议。
    EN: Shared configuration, termination contract, and numerical retry protocol for Torch and C++ restricted OT.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        evaluation_frequency: int = 200,
        time_limit_seconds: float = 3600.0,
        iteration_limit: int = np.iinfo(np.int32).max,
        record_state_trace: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.evaluation_frequency = int(evaluation_frequency)
        self.time_limit_seconds = float(time_limit_seconds)
        self.iteration_limit = int(iteration_limit)
        self.record_state_trace = bool(record_state_trace)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TorchRestrictedOTPDLP requested CUDA, but torch.cuda.is_available() is false.")
        if self.evaluation_frequency < 1:
            raise ValueError("evaluation_frequency must be >= 1")
        if self.time_limit_seconds <= 0.0 or self.iteration_limit < 1:
            raise ValueError("time_limit_seconds and iteration_limit must be positive")

    def solve_restricted(
        self,
        *,
        rows: Any,
        cols: Any,
        costs: Any,
        source_mass: Any,
        target_mass: Any,
        warm_start_primal: Optional[Any] = None,
        warm_start_dual: Optional[Any] = None,
        tolerance: float | Dict[str, float] = 1e-6,
        verbose: bool = False,
    ) -> SolverResult:
        """
        CN: 直接从 active-support 端点求解平衡 restricted OT，不物化 CSR/CSC 矩阵。
        EN: Solve balanced restricted OT directly from active-support endpoints without materializing CSR/CSC matrices.
        """
        if isinstance(tolerance, dict):
            objective_tolerance = float(tolerance.get("objective", 1e-6))
            primal_tolerance = float(tolerance.get("primal", objective_tolerance))
            dual_tolerance = float(tolerance.get("dual", objective_tolerance))
        else:
            objective_tolerance = primal_tolerance = dual_tolerance = float(tolerance)
        first = self._solve_once(
            rows=rows,
            cols=cols,
            costs=costs,
            source_mass=source_mass,
            target_mass=target_mass,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            objective_tolerance=objective_tolerance,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
            scalar_rescaling=False,
            verbose=bool(verbose),
        )
        if first.termination_reason != "NUMERICAL_DIVERGENCE":
            return first
        retry = self._solve_once(
            rows=rows,
            cols=cols,
            costs=costs,
            source_mass=source_mass,
            target_mass=target_mass,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            objective_tolerance=objective_tolerance,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
            scalar_rescaling=True,
            verbose=bool(verbose),
        )
        retry.solver_diag = dict(retry.solver_diag or {})
        retry.solver_diag.update(
            {
                "numerical_rescaling_retry": True,
                "numerical_rescaling_first_termination_reason": first.termination_reason,
                "numerical_rescaling_first_iterations": int(first.iterations),
            }
        )
        return retry

    @abstractmethod
    def _solve_once(self, **kwargs) -> SolverResult:
        """
        CN: 由具体后端执行一次指定缩放模式的求解。
        EN: Execute one solve with the requested scaling mode in the concrete backend.
        """
        raise NotImplementedError
