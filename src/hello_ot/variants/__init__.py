"""
CN: HELLO 变体求解器（半离散 OT、非平衡 OT、Gromov-Wasserstein）。
EN: HELLO variant solvers (semi-discrete OT, unbalanced OT, Gromov-Wasserstein).
"""

from .gromov import GromovIterationRecord, GromovResult, solve_gromov
from .semidiscrete import SemiDiscreteIterationRecord, SemiDiscreteResult, solve_semidiscrete
from .unbalanced import UOTConvergenceBounds, UOTIterationRecord, UnbalancedResult, solve_unbalanced

__all__ = [
    "GromovResult",
    "GromovIterationRecord",
    "SemiDiscreteIterationRecord",
    "SemiDiscreteResult",
    "UOTConvergenceBounds",
    "UOTIterationRecord",
    "UnbalancedResult",
    "solve_gromov",
    "solve_semidiscrete",
    "solve_unbalanced",
]
