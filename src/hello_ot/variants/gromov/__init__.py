"""
CN: HELLO Gromov-Wasserstein（低秩平方欧氏结构与 full-step 外层迭代）。
EN: HELLO Gromov-Wasserstein (low-rank squared-Euclidean structure with full-step outer updates).
"""

from .algorithm import solve_gromov
from .types import GromovIterationRecord, GromovResult

__all__ = ["GromovIterationRecord", "GromovResult", "solve_gromov"]
