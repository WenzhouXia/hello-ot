"""
CN: 层级完成点产生的强类型记录，供算法、阶段转换和公开 API 共用。
EN: Typed completion records shared by the algorithm, stage transitions, and public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from hello_ot.types import (
    CoarsestSolveStats, CostStageResult, InitializationStats,
    LevelResult, RefinementResult, SolveStageResult,
)


@dataclass(frozen=True)
class HierarchyLevelRecord:
    path: str
    result: LevelResult
    lp_solve_time_total: float
    support_before_solve: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    solve_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result, LevelResult):
            raise TypeError("HierarchyLevelRecord.result must be LevelResult")
        if self.result.kind == "coarsest":
            if not isinstance(self.result.solve, CoarsestSolveStats) or self.result.initialization is not None:
                raise TypeError("coarsest records require CoarsestSolveStats and no initialization")
        elif self.result.kind == "refined":
            if not isinstance(self.result.solve, RefinementResult) or not isinstance(self.result.initialization, InitializationStats):
                raise TypeError("refined records require RefinementResult and InitializationStats")
        else:
            raise ValueError("unsupported hierarchy level kind")

    def with_stop_reason(self, reason: str) -> HierarchyLevelRecord:
        """
        CN: 用新记录标记阶段中断，不修改已经记录的阶段。
        EN: Mark a stage interruption with a new record without mutating recorded stages.
        """
        solve = self.result.solve
        if not isinstance(solve, RefinementResult):
            raise TypeError("only refinement levels have a stop reason")
        return replace(self, result=replace(self.result, solve=replace(
            solve, summary=replace(solve.summary, stop_reason=reason),
        )))

    def to_diagnostics(self) -> dict[str, Any]:
        """
        CN: 仅在诊断输出边界转换成原有字典形式，内部不再解析该字典。
        EN: Convert to the existing mapping only at the diagnostic output boundary; internal code never reparses it.
        """
        level = self.result
        solve = level.solve
        record = dict(self.diagnostics)
        summary = dict(self.solve_diagnostics)
        summary["lp_solve_time_total"] = self.lp_solve_time_total
        record.update(path=self.path, depth=level.level_index, n_source=level.n_source,
                      n_target=level.n_target, time=level.wall_time)
        if isinstance(solve, CoarsestSolveStats):
            summary.update(distance=solve.objective)
            record.update(kind="leaf", support_size=solve.support_size,
                          lp_solve_time_total=self.lp_solve_time_total)
        else:
            initial = level.initialization
            assert initial is not None
            summary.update(distance=solve.summary.final_objective, time=solve.summary.wall_time,
                           objective_trace=list(solve.iterations))
            record.update(kind="node", support_before_solve=self.support_before_solve,
                          support_after_solve=solve.summary.final_active_support_size,
                          support_size=solve.summary.final_active_support_size,
                          initialization_time=initial.wall_time,
                          topk_forward_support_size=initial.support_after_forward,
                          topk_reverse_support_size=initial.support_after_reverse,
                          stop_reason=solve.summary.stop_reason)
        record["solve_summary"] = summary
        return record


@dataclass(frozen=True)
class HierarchyStageRecord:
    name: str
    levels: tuple[HierarchyLevelRecord, ...]
    wall_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple) or not all(isinstance(x, HierarchyLevelRecord) for x in self.levels):
            raise TypeError("stage levels must be a tuple of HierarchyLevelRecord")

    def to_result(self) -> CostStageResult:
        return CostStageResult(self.name, SolveStageResult(
            levels=tuple(level.result for level in self.levels), wall_time=self.wall_time,
        ))
