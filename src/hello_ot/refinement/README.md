# Refinement execution

`algorithm._solve_hierarchy` owns the HELLO hierarchy and each level's iteration
loop, including the active-support re-entry detector and automatic cost-stage
switching. It directly calls `iterations.solve_lp`, `check_optimality`, and
`update_support`.

`loop.py` serves standalone warm-start callers, including the variants adapter.
It uses those same operations but does not own the hierarchy's automatic
cost-perturbation switching. Both entry points retain their existing behavior.

The shared operations in `iterations.py` depend on:

- `dual_stabilization.py` for optional EMA and dual gauge alignment;
- `scan_dispatch.py` for full optimality scans by cost family;
- `_internal/instrumentation/refinement.py` for pruning and stabilization
  diagnostics.

None of those dependencies may reach `loop.py`. Local imports are included in
the transitive dependency test. Diagnostic helpers may synchronize GPU arrays;
moving their definitions does not move their invocation or change its order.
