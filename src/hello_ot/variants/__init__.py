"""
CN: HELLO 变体求解器（半离散 OT、非平衡 OT、Gromov-Wasserstein）。
EN: HELLO variant solvers (semi-discrete OT, unbalanced OT, Gromov-Wasserstein).
"""

from .gromov import GromovResult, solve_gromov
from .semidiscrete import SemiDiscreteResult, solve_semidiscrete
from .unbalanced import UnbalancedResult, solve_unbalanced

__all__ = [
    "GromovResult",
    "SemiDiscreteResult",
    "UnbalancedResult",
    "solve_gromov",
    "solve_semidiscrete",
    "solve_unbalanced",
]
