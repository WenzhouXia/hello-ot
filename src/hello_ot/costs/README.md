# Cost implementations

`hello_ot.cost` remains the common entry for cost representation and dispatch.
New cost implementations belong in this directory:

- `metric.py`: L1, L2, and L-infinity leaf solves, dual assignment, support
  initialization, and full optimality-check orchestration.
- `bilinear.py`: bilinear leaf-solve and dual-assignment adapters. Public squared
  L2 is lowered to this representation by `hello_ot.cost`.
- `pairs.py`: shared cost caches and sparse pair evaluation, used by the solver
  and cost families without depending on either workflow.
- `metric_perturbation.py`: shared perturbation arithmetic and CUDA argument
  packing, used by NumPy, Torch, and native scans.

CUDA scan implementations remain in `hello_ot.kernels`. They may depend on cost
arithmetic and scan contracts, but must not import hierarchy, initialization,
refinement, or solver orchestration. The package initializer intentionally does
not eagerly import the workflow modules.

This migration preserves the numerical function bodies and existing dispatch
branches. A common polymorphic family interface is a separate follow-up; this
change first establishes the dependency boundaries it would need.
