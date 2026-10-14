from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from typing import Dict


@dataclass
class SolverResult:
    """
    CN: 统一的 restricted-LP 求解结果。
    EN: Unified restricted-LP solver result.
    """
    success: bool
    # CN: primal transport plan 变量。
    # EN: Primal transport-plan variables.
    x: np.ndarray = None
    # CN: 对偶势。
    # EN: Dual potentials.
    y: np.ndarray = None
    obj_val: float = 0.0
    dual_obj_val: float = None
    duration: float = 0.0
    iterations: int = 0
    # CN: 以 MiB 计的峰值内存。
    # EN: Peak memory in MiB.
    peak_mem: float = 0.0
    termination_reason: str = None
    # CN: 求解器收敛指标。
    # EN: Solver convergence metrics.
    primal_feas: float = None
    dual_feas: float = None
    gap: float = None
    # CN: backend wrapper 的计时与诊断信息。
    # EN: Backend-wrapper timing and diagnostics.
    solver_diag: Dict = None


class LPSolver(ABC):
    """
    CN: restricted-LP 求解器的抽象基类。
    EN: Abstract base class for restricted-LP solvers.
    """
    @abstractmethod
    def solve(
        self,
        c: np.ndarray,
        A_csc: sparse.csc_matrix,
        b_eq: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_eqs: int,
        warm_start_primal=None,
        warm_start_dual=None,
        tolerance: Dict[str, float] = {
            'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        verbose: int = 0,
        **kwargs
    ) -> SolverResult:
        pass
