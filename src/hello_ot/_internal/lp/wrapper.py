from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from typing import Dict


@dataclass
class SolverResult:
    """统一的求解结果对象"""
    success: bool
    x: np.ndarray = None        # Primal solution (transport plan) values
    y: np.ndarray = None        # Dual solution (potentials)
    obj_val: float = 0.0
    dual_obj_val: float = None
    duration: float = 0.0
    iterations: int = 0
    peak_mem: float = 0.0       # Memory usage in MB
    termination_reason: str = None
    # 新增：收敛指标 (HALO 风格)
    primal_feas: float = None   # Primal feasibility (infeasibility)
    dual_feas: float = None     # Dual feasibility
    gap: float = None           # Duality gap
    solver_diag: Dict = None    # Backend wrapper timing/details


class LPSolver(ABC):
    """
    LP 求解器的抽象基类。
    """
    @abstractmethod
    def solve(
        self,
        c: np.ndarray,            # Cost vector
        A_csc: sparse.csc_matrix,  # Constraint matrix
        b_eq: np.ndarray,         # RHS (marginals)
        lb: np.ndarray,           # Lower bounds
        ub: np.ndarray,           # Upper bounds
        n_eqs: int,
        warm_start_primal=None,
        warm_start_dual=None,
        tolerance: Dict[str, float] = {
            'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        verbose: int = 0,
        **kwargs
    ) -> SolverResult:
        pass
