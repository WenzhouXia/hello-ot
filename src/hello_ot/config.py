from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

DualFeasibilityNorm = Literal["l2", "linf"]
BackendName = Literal["native", "torch"]
TorchDeviceName = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class SolverOptions:
    """
    CN: 面向高级用户和论文实验的扁平选项；普通调用无需构造该对象。
    EN: Flat options for advanced users and paper experiments; ordinary calls do not need it.
    """

    split_count: int = 4
    coarsest_size_threshold: int = 1024
    assignment_topk: int = 16
    pricing_topk: int = 2
    support_budget_factor: float = 10.0
    fused_scan_memory_floor_mib: float = 768.0
    verbose: bool = False
    record_trace: bool = False
    profile_memory: bool = False
    consume_input_features: bool = False
    backend: BackendName = "native"
    torch_device: TorchDeviceName = "auto"

    def __post_init__(self) -> None:
        if self.split_count not in {2, 4, 8}:
            raise ValueError("split_count must be one of: 2, 4, 8")
        if self.coarsest_size_threshold < 1:
            raise ValueError("coarsest_size_threshold must be >= 1")
        if self.assignment_topk < 1 or self.pricing_topk < 1:
            raise ValueError("assignment_topk and pricing_topk must be >= 1")
        if self.support_budget_factor <= 0.0:
            raise ValueError("support_budget_factor must be > 0")
        if self.fused_scan_memory_floor_mib <= 0.0:
            raise ValueError("fused_scan_memory_floor_mib must be > 0")
        if self.backend not in {"native", "torch"}:
            raise ValueError("backend must be one of: native, torch")
        if self.torch_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("torch_device must be one of: auto, cpu, cuda")

    def _to_internal_config(self, *, max_iterations: int, random_seed: int) -> "_AlgorithmConfig":
        """
        CN: 构造采用论文性能默认值的内部配置。
        EN: Build the internal configuration with the paper-performance defaults.
        """
        return _AlgorithmConfig(
            split_count=self.split_count,
            coarsest_size_threshold=self.coarsest_size_threshold,
            random_seed=random_seed,
            assignment_topk=self.assignment_topk,
            pricing_topk=self.pricing_topk,
            support_budget_factor=self.support_budget_factor,
            primal_tolerance=1e-10,
            max_iterations=max_iterations,
            fused_scan_memory_floor_mib=self.fused_scan_memory_floor_mib,
            verbose=self.verbose,
            record_trace=self.record_trace,
            profile_memory=self.profile_memory,
            consume_input_features=self.consume_input_features,
            backend=self.backend,
            torch_device=self.torch_device,
        )


@dataclass(frozen=True)
class _AlgorithmConfig:
    """
    CN: `solve` 解析后的私有算法设置；固定的论文参数不暴露给调用方。
    EN: Private resolved algorithm settings; fixed paper parameters are not exposed to callers.
    """

    split_count: int
    coarsest_size_threshold: int
    random_seed: int
    assignment_topk: int
    pricing_topk: int
    support_budget_factor: float
    primal_tolerance: float
    max_iterations: int
    fused_scan_memory_floor_mib: float
    verbose: bool
    record_trace: bool
    profile_memory: bool
    consume_input_features: bool
    backend: BackendName
    torch_device: TorchDeviceName
    dual_feasibility_tolerance: float = 1e-6
    dual_feasibility_norm: DualFeasibilityNorm = "l2"
    lp_tolerance: float = 1e-6
    lp_stopping_norm: DualFeasibilityNorm = "l2"
    ensure_final_dual_feasible: bool = False
    record_dual_linf_diagnostics: bool = False
    variable_bound_mode: str = "constant"
    matrix_value_mode: str = "implicit_aty"
    vector_sum_mode: str = "direct_reduce"
    bound_objective_rescaling: Union[bool, Literal["auto"]] = "auto"
    cupdlpx_python_pre_rescale: bool = True


@dataclass
class SolverRuntimeConfig:
    """
    CN: HELLO 组件共享的扁平运行时视图；只在算法内部使用，不属于公开 API。
    EN: Flat runtime view shared by HELLO components; internal to the algorithm and not public API.

    CN: 它暂时保持机械迁移所需的字段名，但不依赖任何 legacy solver config。
    EN: It temporarily preserves field names needed by the mechanical migration without depending on a legacy solver config.
    """

    cost_type: str
    solver_engine: str
    tolerance: float
    max_inner_iter: int
    dual_feasibility_tol: float
    lp_termination_norm: str
    pricing_topk: float
    cleaning_primal_tol: float
    cleaning_threshold_factor: float
    lp_solver_verbose: bool
    enable_profiling: bool
    printing: Dict[str, Any]
    profiling: Dict[str, Any]
    variable_bound_mode: str
    matrix_value_mode: str
    vector_sum_mode: str
    bound_objective_rescaling: Union[bool, Literal["auto"]]
    cupdlpx_python_pre_rescale: bool
    backend: BackendName = "native"
    torch_device: str = "cuda"
    dot_scale: float = 1.0
    pricing_strategy: str = "nodewise_full"
    pricing_direction: str = "both"
    cleaning_strategy: str = "dual_gap"
    convergence_criterion: str = "dual_feasibility"
    require_dual_feasibility_convergence: bool = False
    require_added_convergence: bool = False
    use_fused_lowrank_feasibility_pricing: bool = True
    fused_lowrank_scan_memory_floor_mib: float = 768.0
    fused_lowrank_scan_resident_multiplier: float = 1.10
    record_pricing_dual_feasibility: bool = True
    record_dual_linf_diagnostics: bool = False
    ensure_final_dual_feasible: bool = False
    dual_stabilization_mode: str = "off"
    dual_stabilization_alpha: float = 0.5
    dual_stabilization_gauge_normalization: bool = False
    report_added_violation_stats: bool = False
    added_violation_rel_threshold: float = 1e-6
    runtime_logging: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None

    @classmethod
    def from_options(cls, config: _AlgorithmConfig, *, cost_type: str) -> "SolverRuntimeConfig":
        printing = {
            "enabled": bool(config.verbose),
            "progress": bool(config.verbose),
            "warm_start": bool(config.verbose),
            "profile_iter": bool(config.verbose),
            "profile_level": bool(config.verbose),
            "profile_run": bool(config.verbose),
            "iter_interval": 10,
        }
        profiling = {
            "enabled": False,
            "write_trace_json": False,
            "trace_json_path": None,
            "capture_component_breakdown": True,
            "memory": {
                "enabled": bool(config.profile_memory),
                "include_state_bytes": False,
                "include_driver": True,
                "mode": "light",
                "jsonl_path": None,
            },
        }
        runtime = cls(
            cost_type=str(cost_type),
            solver_engine="cupdlpx" if config.backend == "native" else "torch_pdlp",
            tolerance=float(config.lp_tolerance),
            max_inner_iter=int(config.max_iterations),
            dual_feasibility_tol=float(config.dual_feasibility_tolerance),
            lp_termination_norm=str(config.lp_stopping_norm),
            pricing_topk=float(config.pricing_topk),
            cleaning_primal_tol=float(config.primal_tolerance),
            cleaning_threshold_factor=float(config.support_budget_factor),
            lp_solver_verbose=bool(config.verbose),
            enable_profiling=False,
            printing=printing,
            profiling=profiling,
            variable_bound_mode=str(config.variable_bound_mode),
            matrix_value_mode=str(config.matrix_value_mode),
            vector_sum_mode=str(config.vector_sum_mode),
            bound_objective_rescaling=config.bound_objective_rescaling,
            cupdlpx_python_pre_rescale=bool(config.cupdlpx_python_pre_rescale),
            backend=config.backend,
            torch_device=str(config.torch_device),
            record_dual_linf_diagnostics=bool(config.record_dual_linf_diagnostics),
            runtime_logging=dict(printing),
            ensure_final_dual_feasible=bool(config.ensure_final_dual_feasible),
            fused_lowrank_scan_memory_floor_mib=float(config.fused_scan_memory_floor_mib),
        )
        runtime.validate()
        return runtime

    def normalized_tolerance(self) -> Dict[str, float]:
        scalar = float(self.tolerance)
        return {"objective": scalar, "primal": scalar, "dual": scalar}

    def normalized_profiling(self) -> Dict[str, Any]:
        return dict(self.profiling)

    def normalized_runtime_logging(self) -> Dict[str, Any]:
        return dict(self.runtime_logging or self.printing)

    def validate(self) -> None:
        if float(self.dot_scale) not in {1.0, 2.0}:
            raise ValueError("dot_scale must be exactly 1 or 2")
        if self.solver_engine not in {"cupdlpx", "torch_pdlp"}:
            raise ValueError("HELLO solver_engine must be cupdlpx or torch_pdlp")
        if self.max_inner_iter < 1:
            raise ValueError("max_inner_iter must be >= 1")
        if self.tolerance <= 0.0 or self.dual_feasibility_tol < 0.0:
            raise ValueError("LP tolerance must be > 0 and dual-feasibility tolerance must be >= 0")
        if self.lp_termination_norm not in {"l2", "linf"}:
            raise ValueError("lp_termination_norm must be one of: l2, linf")
        if self.pricing_direction != "both":
            raise ValueError("formal HELLO pricing_direction must be 'both'")
        if self.cleaning_strategy != "dual_gap":
            raise ValueError("formal HELLO cleaning_strategy must be 'dual_gap'")


__all__ = [
    "SolverOptions",
    "SolverRuntimeConfig",
]
