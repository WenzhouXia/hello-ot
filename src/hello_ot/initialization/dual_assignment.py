from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch

from hello_ot.cost import (
    HelloCostContext,
    assign_from_complete_dual,
    complete_dual,
)
from hello_ot.initialization.state import (
    _normalize_dual_assignment_state,
    _state_support_size,
)
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState
from hello_ot.types import DualPotentials, PreparedDual


ArrayLike = Union[np.ndarray, torch.Tensor]
DualPreparation = Literal["preserve", "source_from_target", "target_from_source"]
DualAssignment = Literal["nodewise_bidirectional", "rowwise", "columnwise"]


@dataclass(frozen=True)
class DualAssignmentProblem:
    """
    CN: 计算当前 cost 所需的最小 dual-assignment problem 表示。
    EN: Minimal dual-assignment problem representation needed to evaluate the current cost.
    """

    cost_context: HelloCostContext
    source_representation: ArrayLike
    target_representation: ArrayLike
    source_cost_vector: Optional[ArrayLike] = None
    target_cost_vector: Optional[ArrayLike] = None


def _numpy_float32(value: ArrayLike, *, ndim: int, name: str) -> np.ndarray:
    """
    CN: 将输入数组或 Tensor 转换为指定维度的连续 float32 NumPy 数组。
    EN: Convert an input array or Tensor to a contiguous float32 NumPy array with the specified ndim.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if int(array.ndim) != int(ndim):
        raise ValueError(f"{name} must be {ndim}D; got shape={array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _numpy_float64(value: ArrayLike, *, ndim: int, name: str) -> np.ndarray:
    """
    CN: 将标量 cost/dual 输入转换为连续 float64 NumPy 数组。
    EN: Convert scalar cost/dual inputs to contiguous float64 NumPy arrays.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if int(array.ndim) != int(ndim):
        raise ValueError(f"{name} must be {ndim}D; got shape={array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _normalized_problem(problem: DualAssignmentProblem) -> Dict[str, Any]:
    """
    CN: 校验并提取 DualAssignmentProblem 中的特征、代价向量和全局索引，统一转换为标准 float32/int64 NumPy 数组。
    EN: Validate and extract representations, cost vectors, and global indices from DualAssignmentProblem into normalized float32/int64 NumPy arrays.
    """
    source = _numpy_float32(problem.source_representation, ndim=2, name="source_representation")
    target = _numpy_float32(problem.target_representation, ndim=2, name="target_representation")
    if int(source.shape[1]) != int(target.shape[1]):
        raise ValueError("source_representation and target_representation must have matching feature dimensions.")
    n_source = int(source.shape[0])
    n_target = int(target.shape[0])
    score_family = str(problem.cost_context.score_family)
    if score_family == "inner_product":
        if problem.source_cost_vector is None or problem.target_cost_vector is None:
            raise ValueError("lowrank cost requires source_cost_vector and target_cost_vector.")
        source_cost = _numpy_float64(problem.source_cost_vector, ndim=1, name="source_cost_vector")
        target_cost = _numpy_float64(problem.target_cost_vector, ndim=1, name="target_cost_vector")
    elif score_family == "norm_cost":
        source_cost = (
            np.zeros(n_source, dtype=np.float64)
            if problem.source_cost_vector is None
            else _numpy_float64(problem.source_cost_vector, ndim=1, name="source_cost_vector")
        )
        target_cost = (
            np.zeros(n_target, dtype=np.float64)
            if problem.target_cost_vector is None
            else _numpy_float64(problem.target_cost_vector, ndim=1, name="target_cost_vector")
        )
    else:
        raise ValueError(f"unsupported dual-assignment score family={score_family!r}.")
    if int(source_cost.size) != n_source or int(target_cost.size) != n_target:
        raise ValueError("cost-vector lengths must match their corresponding representations.")
    return {
        "source_F": source,
        "target_G": target,
        "source_cost_vec": source_cost,
        "target_cost_vec": target_cost,
        "n_source": n_source,
        "n_target": n_target,
    }


def _dual_numpy(value: ArrayLike, *, size: int, name: str) -> np.ndarray:
    """
    CN: 将对偶势转换为长度为 size 的 1D float64 NumPy 连续数组。
    EN: Convert a dual potential input into a 1D float64 contiguous NumPy array of given size.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if int(array.size) != int(size):
        raise ValueError(f"{name} length must be {size}; got {array.size}.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _empty_gpu_state(
    *,
    arrays: Dict[str, Any],
    source_dual: np.ndarray,
    target_dual: np.ndarray,
) -> OTWarmStartGPUState:
    """
    CN: 构造空支撑的 OTWarmStartGPUState，并填入指定的对偶势 dual_uv = [source_dual, target_dual]。
    EN: Construct an empty-support OTWarmStartGPUState filled with specified dual potentials dual_uv = [source_dual, target_dual].
    """
    dual_uv = np.concatenate([source_dual, target_dual], axis=0)
    state = OTWarmStartState(
        rows=np.empty(0, dtype=np.int32),
        cols=np.empty(0, dtype=np.int32),
        x_prev=np.empty(0, dtype=np.float64),
        dual_uv=np.ascontiguousarray(dual_uv, dtype=np.float64),
        n_source=int(arrays["n_source"]),
        n_target=int(arrays["n_target"]),
        northwest_positions=None,
    )
    normalized = _normalize_dual_assignment_state(state, pipeline="gpu")
    if not isinstance(normalized, OTWarmStartGPUState):
        raise RuntimeError("GPU dual preparation must return OTWarmStartGPUState.")
    return normalized


def prepare_dual(
    problem: DualAssignmentProblem,
    dual: DualPotentials,
    *,
    policy: DualPreparation = "preserve",
) -> Tuple[PreparedDual, Dict[str, Any]]:
    """
    CN: 保留双侧 potentials，或在当前 cost 下用 c-transform 从已知侧补齐另一侧。
    EN: Preserve two-sided potentials or complete the missing side by a c-transform under the current cost.
    """

    started = time.perf_counter()
    arrays = _normalized_problem(problem)
    n_source = int(arrays["n_source"])
    n_target = int(arrays["n_target"])
    mode = str(policy)

    # CN: 策略 1: 直接保留并校验双侧已有对偶势
    # EN: Policy 1: directly preserve and validate two-sided dual potentials
    if mode == "preserve":
        if dual.source is None or dual.target is None:
            raise ValueError("preparation='preserve' requires both source and target dual potentials.")
        prepared = PreparedDual(
            source=_dual_numpy(dual.source, size=n_source, name="source dual"),
            target=_dual_numpy(dual.target, size=n_target, name="target dual"),
        )
        return prepared, {
            "policy": "preserve",
            "score_family": str(problem.cost_context.score_family),
            "total_time": float(time.perf_counter() - started),
        }

    # CN: 策略 2: 单侧已知，准备空支撑 GPU state 以便进行 c-transform 补全
    # EN: Policy 2: single-side known; prepare an empty-support GPU state for c-transform completion
    if mode == "source_from_target":
        if dual.target is None:
            raise ValueError("preparation='source_from_target' requires target dual potentials.")
        known_side: Literal["source", "target"] = "target"
        source_dual = np.zeros(n_source, dtype=np.float64)
        target_dual = _dual_numpy(dual.target, size=n_target, name="target dual")
    elif mode == "target_from_source":
        if dual.source is None:
            raise ValueError("preparation='target_from_source' requires source dual potentials.")
        known_side = "source"
        source_dual = _dual_numpy(dual.source, size=n_source, name="source dual")
        target_dual = np.zeros(n_target, dtype=np.float64)
    else:
        raise ValueError(f"unsupported dual preparation policy={policy!r}.")
    state = _empty_gpu_state(
        arrays=arrays,
        source_dual=source_dual,
        target_dual=target_dual,
    )

    # CN: 按当前代价类型调度底层 c-transform 补全缺失侧对偶势
    # EN: Dispatch underlying c-transform to complete missing dual potentials under current cost
    completed, completion_profile = complete_dual(
        cost_context=problem.cost_context,
        state=state,
        source_F=arrays["source_F"],
        target_G=arrays["target_G"],
        source_cost_vec=arrays["source_cost_vec"],
        target_cost_vec=arrays["target_cost_vec"],
        known_side=known_side,
        completion_candidate=None,
        tracer=None,
        trace_args={},
        trace_prefix="dual_warm_start.prepare_dual",
    )
    if not isinstance(completed, OTWarmStartGPUState) or completed.dual_uv is None:
        raise RuntimeError("GPU c-transform completion did not return GPU dual potentials.")

    # CN: 提取补全后的双侧对偶势构造 PreparedDual 并记录统计 profile
    # EN: Extract completed two-sided dual potentials into PreparedDual and record profile
    prepared = PreparedDual(
        source=completed.dual_uv[:n_source],
        target=completed.dual_uv[n_source:],
    )
    return prepared, {
        "policy": mode,
        "known_side": known_side,
        "score_family": str(problem.cost_context.score_family),
        "completion_profile": dict(completion_profile),
        "total_time": float(time.perf_counter() - started),
    }


def _state_from_prepared(
    *, problem: DualAssignmentProblem, arrays: Dict[str, Any], dual: PreparedDual
) -> OTWarmStartGPUState:
    """
    CN: 从 PreparedDual 构造初始 OTWarmStartGPUState（保留 CUDA Tensor 或转为空支撑 GPU state）。
    EN: Construct an initial OTWarmStartGPUState from PreparedDual (preserving CUDA tensors or converting to an empty-support GPU state).
    """
    # CN: 若输入已为 CUDA Tensor，直接拼装并初始化 OTWarmStartGPUState
    # EN: If inputs are already CUDA tensors, directly assemble and initialize OTWarmStartGPUState
    if isinstance(dual.source, torch.Tensor) and isinstance(dual.target, torch.Tensor):
        source = dual.source.detach().reshape(-1)
        target = dual.target.detach().reshape(-1)
        if not source.is_cuda or not target.is_cuda or source.device != target.device:
            raise ValueError("PreparedDual tensors must be CUDA tensors on the same device.")
        if int(source.numel()) != int(arrays["n_source"]) or int(target.numel()) != int(arrays["n_target"]):
            raise ValueError("PreparedDual lengths must match the dual-assignment problem.")
        device = source.device
        return OTWarmStartGPUState(
            rows=torch.empty(0, dtype=torch.int32, device=device),
            cols=torch.empty(0, dtype=torch.int32, device=device),
            x_prev=torch.empty(0, dtype=torch.float64, device=device),
            dual_uv=torch.cat(
                [source.to(dtype=torch.float64), target.to(dtype=torch.float64)], dim=0
            ).contiguous(),
            n_source=int(arrays["n_source"]),
            n_target=int(arrays["n_target"]),
            device=str(device),
            keys=torch.empty(0, dtype=torch.int64, device=device),
            northwest_positions=None,
        )
    # CN: 若输入为 NumPy 数组，通过 _empty_gpu_state 构造标准 GPU state
    # EN: If inputs are NumPy arrays, construct a standard GPU state via _empty_gpu_state
    return _empty_gpu_state(
        arrays=arrays,
        source_dual=_dual_numpy(dual.source, size=int(arrays["n_source"]), name="source dual"),
        target_dual=_dual_numpy(dual.target, size=int(arrays["n_target"]), name="target dual"),
    )


def _augment_one_direction(
    *,
    problem: DualAssignmentProblem,
    arrays: Dict[str, Any],
    state: OTWarmStartGPUState,
    known_side: Literal["source", "target"],
    assignment_topk: int,
) -> Tuple[OTWarmStartGPUState, Dict[str, Any], Dict[str, Any]]:
    """
    CN: 在指定方向（rowwise 或 columnwise）上执行单向 top-k 对偶分配（dual assignment），并将候选边合并入 state。
    EN: Execute one-directional top-k dual assignment along the specified direction (rowwise or columnwise) and merge candidate edges into state.
    """
    result, stats, profile, _completion_candidate = assign_from_complete_dual(
        cost_context=problem.cost_context,
        state=state,
        source_F=arrays["source_F"],
        target_G=arrays["target_G"],
        source_cost_vec=arrays["source_cost_vec"],
        target_cost_vec=arrays["target_cost_vec"],
        known_side=known_side,
        assignment_topk=int(assignment_topk),
        tracer=None,
        trace_args={},
    )
    if not isinstance(result, OTWarmStartGPUState):
        raise RuntimeError("GPU dual assignment must return OTWarmStartGPUState.")
    return result, dict(stats), dict(profile)


def _build_warm_start_from_dual_impl(
    problem: DualAssignmentProblem,
    dual: Union[DualPotentials, PreparedDual],
    *,
    preparation: DualPreparation = "preserve",
    assignment: DualAssignment = "nodewise_bidirectional",
    assignment_topk: int = 16,
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 从 dual potentials 构造纯 dual-assigned warm-start support，不加入 NW feasible basis。
    EN: Build a dual-only warm-start support from dual potentials without adding a NW feasible basis.
    """

    started = time.perf_counter()

    # CN: 模块 1: 参数校验与问题表示归一化
    # EN: Module 1: parameter validation and problem representation normalization
    if int(assignment_topk) <= 0:
        raise ValueError("assignment_topk must be positive.")
    arrays = _normalized_problem(problem)

    # CN: 模块 2: 对偶准备阶段（检查 PreparedDual 或调用 prepare_dual 执行 c-transform 补全）
    # EN: Module 2: dual preparation phase (verify PreparedDual or invoke prepare_dual for c-transform completion)
    if isinstance(dual, PreparedDual):
        if str(preparation) != "preserve":
            raise ValueError("PreparedDual only supports preparation='preserve'.")
        prepared = dual
        preparation_profile = {"policy": "preserve", "skipped": True, "total_time": 0.0}
        anchor_side: Literal["source", "target"] = "target"
    elif isinstance(dual, DualPotentials):
        prepared, preparation_profile = prepare_dual(problem, dual, policy=preparation)
        anchor_side = "source" if str(preparation) == "target_from_source" else "target"
    else:
        raise TypeError("dual must be DualPotentials or PreparedDual.")

    # CN: 模块 3: 初始化空支撑 GPU warm-start state
    # EN: Module 3: initialize empty-support GPU warm-start state
    state = _state_from_prepared(problem=problem, arrays=arrays, dual=prepared)
    mode = str(assignment)
    passes: list[Dict[str, Any]] = []

    # CN: 模块 4: 对偶分配策略调度 (rowwise / columnwise / nodewise_bidirectional / global)
    # EN: Module 4: dual assignment strategy dispatch (rowwise / columnwise / nodewise_bidirectional / global)
    if mode == "rowwise":
        # CN: 行向单侧 top-k 选边（以 target dual 为依据选源点边）
        # EN: Rowwise one-sided top-k edge selection (source edges guided by target dual)
        state, stats, augment_profile = _augment_one_direction(
            problem=problem, arrays=arrays, state=state, known_side="target", assignment_topk=int(assignment_topk)
        )
        passes.append(
            {
                "direction": "rowwise",
                "stats": stats,
                "profile": augment_profile,
                "support_size": int(_state_support_size(state)),
            }
        )
    elif mode == "columnwise":
        # CN: 列向单侧 top-k 选边（以 source dual 为依据选目标边）
        # EN: Columnwise one-sided top-k edge selection (target edges guided by source dual)
        state, stats, augment_profile = _augment_one_direction(
            problem=problem, arrays=arrays, state=state, known_side="source", assignment_topk=int(assignment_topk)
        )
        passes.append(
            {
                "direction": "columnwise",
                "stats": stats,
                "profile": augment_profile,
                "support_size": int(_state_support_size(state)),
            }
        )
    elif mode == "nodewise_bidirectional":
        # CN: 双向 top-k 选边：先 anchor 侧再对侧，依次增强支撑并去重合并
        # EN: Bidirectional top-k edge selection: anchor side first, then opposite side to augment support
        sides: Tuple[Literal["source", "target"], Literal["source", "target"]] = (
            (anchor_side, "target" if anchor_side == "source" else "source")
        )
        for side in sides:
            state, stats, augment_profile = _augment_one_direction(
                problem=problem, arrays=arrays, state=state, known_side=side, assignment_topk=int(assignment_topk)
            )
            passes.append(
                {
                    "direction": "columnwise" if side == "source" else "rowwise",
                    "stats": stats,
                    "profile": augment_profile,
                    "support_size": int(_state_support_size(state)),
                }
            )
    else:
        raise ValueError(
            f"unsupported dual assignment={assignment!r}; the global ablation was removed. "
            "See commits 7896574 and 0151292."
        )

    # CN: 模块 5: 清空 NW 占位、组装诊断 profile 并返回
    # EN: Module 5: clear NW placeholder, assemble diagnostic profile, and return
    state.northwest_positions = None
    profile = {
        "preparation": preparation_profile,
        "preparation_policy": str(preparation),
        "assignment": mode,
        "assignment_topk": int(assignment_topk),
        "score_family": str(problem.cost_context.score_family),
        "cost_type": str(problem.cost_context.cost_type),
        "backend": "gpu",
        "passes": passes,
        "support_size": int(_state_support_size(state)),
        "total_time": float(time.perf_counter() - started),
    }
    return state, profile


def build_warm_start_from_dual(
    problem: DualAssignmentProblem,
    dual: DualPotentials | PreparedDual,
    *,
    preparation: DualPreparation = "preserve",
    assignment: DualAssignment = "nodewise_bidirectional",
    assignment_topk: int = 16,
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 使用内置 CUDA scan 从 dual potentials 构造纯 dual-guided warm start。
    EN: Build a pure dual-guided warm start from dual potentials with the bundled CUDA scan.
    """
    return _build_warm_start_from_dual_impl(
        problem,
        dual,
        preparation=preparation,
        assignment=assignment,
        assignment_topk=int(assignment_topk),
    )


def build_related_problem_warm_start(
    problem: DualAssignmentProblem,
    previous_state: OTWarmStartState | OTWarmStartGPUState,
    *,
    assignment: DualAssignment = "nodewise_bidirectional",
    assignment_topk: int = 16,
) -> Tuple[OTWarmStartState | OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 仅提取相关问题的 dual potentials，并重建纯 dual-guided warm start。
    EN: Extract only the related problem's dual potentials and rebuild a pure dual-guided warm start.
    """

    if not isinstance(previous_state, (OTWarmStartState, OTWarmStartGPUState)):
        raise TypeError("previous_state must be an OTWarmStartState or OTWarmStartGPUState.")

    arrays = _normalized_problem(problem)
    n_source = int(arrays["n_source"])
    n_target = int(arrays["n_target"])
    if int(previous_state.n_source) != n_source or int(previous_state.n_target) != n_target:
        raise ValueError(
            "previous_state dimensions do not match the current dual-assignment problem."
        )
    if previous_state.dual_uv is None:
        raise ValueError("dual-only related-problem warm start requires previous_state.dual_uv.")
    dual_uv = previous_state.dual_uv
    if isinstance(dual_uv, torch.Tensor):
        dual_flat: ArrayLike = dual_uv.detach().reshape(-1)
        if int(dual_flat.numel()) != n_source + n_target:
            raise ValueError("previous_state.dual_uv has an incompatible length.")
    else:
        dual_flat = np.asarray(dual_uv, dtype=np.float64).reshape(-1)
        if int(dual_flat.size) != n_source + n_target:
            raise ValueError("previous_state.dual_uv has an incompatible length.")
    dual = DualPotentials(
        source=dual_flat[:n_source],
        target=dual_flat[n_source:],
    )
    state, profile = build_warm_start_from_dual(
        problem,
        dual,
        preparation="preserve",
        assignment=assignment,
        assignment_topk=int(assignment_topk),
    )
    profile = dict(profile)
    profile.update(
        {
            "mode": "dual_preserve",
            "reused_primal": False,
            "reused_support": False,
        }
    )
    return state, profile


__all__ = [
    "DualAssignmentProblem",
    "DualPotentials",
    "PreparedDual",
    "build_related_problem_warm_start",
    "build_warm_start_from_dual",
    "prepare_dual",
]
