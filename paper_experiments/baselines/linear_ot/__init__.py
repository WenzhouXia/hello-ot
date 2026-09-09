"""CN: 线性 OT baseline 的延迟导入入口。EN: Lazy import surface for linear-OT baselines."""

from importlib import import_module
from pathlib import Path
from typing import Any

from .problem import LinearOTProblem
from .result import LinearOTResult, TransportEvaluation




HELLO_WARMSTARTED_LAZY_EMD_METHOD = "pot_lazy_emd_hello_warmstart"
HELLO_WARMSTARTED_EMD_METHOD = "pot_emd_hello_warmstart"


_ROUTES = {
    "solve_pot_proximal_point": (".pot_proximal", "solve_pot_proximal_point"),
    "solve_ott_jax_sinkhorn_l1_negdot_std": (".ott_sinkhorn", "solve_ott_jax_sinkhorn_l1_negdot_std"),
    "evaluate_transport": (".evaluation", "evaluate_transport"),
    "HelloSolveArtifacts": (".hello_solver", "HelloSolveArtifacts"),
    "solve_hello": (".hello_solver", "solve_hello"),
    "solve_hello_with_duals": (".hello_solver", "solve_hello_with_duals"),
    "solve_hello_warmstarted_pot_lazy_emd": (
        ".hello_warmstarted_lazy_emd",
        "solve_hello_warmstarted_pot_lazy_emd",
    ),
    "solve_hello_warmstarted_pot_emd": (
        ".hello_warmstarted_pot_emd",
        "solve_hello_warmstarted_pot_emd",
    ),
    "solve_mdot_tnt": (".mdot_tnt_adapter", "solve_mdot_tnt"),
    "solve_neufeld_cutplane": (".neufeld_cutplane_adapter", "solve_neufeld_cutplane"),
    "solve_pot_lazy_emd": (".pot_lazy_emd", "solve_pot_lazy_emd"),
    "solve_pot_emd": (".pot_emd", "solve_pot_emd"),
    "solve_zanetti_ipm": (".zanetti_ipm_adapter", "solve_zanetti_ipm"),
}

# CN: 仅暴露当前仓库实际包含的 adapter，导出时无需改写源码。
# EN: Expose only adapters present in this repository without rewriting sources during export.
_ROUTES = {name: route for name, route in _ROUTES.items()
           if Path(__file__).with_name(route[0][1:] + ".py").is_file()}
__all__ = ["LinearOTProblem", "LinearOTResult", "TransportEvaluation",
           "HELLO_WARMSTARTED_LAZY_EMD_METHOD", "HELLO_WARMSTARTED_EMD_METHOD", *_ROUTES]


def __getattr__(name: str) -> Any:
    """
    CN: 只在使用某个 baseline 时导入其可选依赖。
    EN: Import optional dependencies only when a baseline is actually used.
    """

    if name not in _ROUTES:
        raise AttributeError(name)
    module_name, attribute = _ROUTES[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
