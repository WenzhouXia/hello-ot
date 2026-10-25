# HELLO terminology: paper ↔ code

This glossary is the authority for names shared by the paper and implementation. Paper labels and code symbols are used instead of line numbers because line numbers drift.

## HELLO

Dual-guided **H**ierarchical **E**dge **L**ocalization for **L**arge-scale **O**ptimal transport. The production implementation is `hello_ot`; the public entry point is `hello_ot.solve`.

Avoid using `EMD`, `emd`, `asymmetric chain`, or `recursive solver` as the name of HELLO. EMD is commonly associated specifically with Wasserstein-1, whereas HELLO supports several ground costs.

## GPU-resident execution

`SolverOptions.gpu_resident=True` selects the optional balanced-OT execution path
that retains both point clouds, active support, LP arrays, and results on CUDA.
Leaf POT `emd` and triggered support sparsification are explicit CPU exceptions.
The default resident-streamed path retains its previous memory policy.

In this mode, `consume_input_features=True` permits temporary in-place feature
reordering with restoration on exit and bounded scratch. It does not transfer
ownership, erase PyTorch tensor version counters, or enable solver autograd.

## BuildHierarchy

Constructs a coarse-to-fine dual-guided hierarchy. The implementation applies one deterministic permutation, then repeatedly partitions the larger side and follows the selected child range. See `_solve_hierarchy` and `HierarchyNodeRange`.

By default the shuffle-once feature layout is materialized. With explicit `consume_input_features=True`, writable C-contiguous FP32 feature matrices are permuted in place during the solve and restored on normal return or ordinary exceptions.

## Dual propagation

Transfers dual information between adjacent hierarchy levels. One shared side is inherited and the other is completed by a c-transform. Paper dual potentials $(f,g)$ are represented by `dual_uv` in code.

## Dual assignment

Selects high dual-score edges in both directions to form the candidate support. `assignment_topk` is the paper parameter $\kappa$. The production path is always nodewise and bidirectional.

## Initial active support

The union of dual-assigned candidates and a northwest-corner feasible basis. Use `initial active support` or `construct_initial_active_support`; avoid the informal term `seed` for this mathematical object.

## Restricted OT / SolveLP

The transport LP restricted to the current active support. The production backend is the bundled CuPDLPx implementation. `solve_lp` is an internal operation, not a second public solver API.

## CPU C++ backend

The optional `backend="cpp"` is the official CPU implementation of balanced
point-cloud HELLO, displayed as `HELLO (C++)`. It shares the hierarchy and
refinement algorithm, uses the independently installed `hello-ot-cpp` kernels,
and defaults to `num_threads=16`. The `torch` and `native` backends remain available.

## Dual score and reduced cost

The paper uses the dual score $\sigma_{ij}=f_i+g_j-c_{ij}$. Code may compute the reduced cost $r_{ij}=c_{ij}-u_i-v_j=-\sigma_{ij}$. A dual violation means $\sigma_{ij}>0$, equivalently $r_{ij}<0$.

## Full optimality check

Checks dual feasibility over the complete edge set after a restricted LP solve. The public tolerance is fixed to the paper value rather than exposed as a user option.

`SolverOptions.stopping_norm` selects a hierarchy-wide stopping policy. `l2` uses L2 for restricted-LP stopping and the full optimality check at every refinement level. `finest_linf` uses L2 at coarser levels and L-infinity for both checks at the finest level, so the returned solution has an L-infinity certificate. The default is `l2`. `finest_linf` currently requires the native squared-L2 path. The native squared-L2 scan records both L2 and L-infinity diagnostics from the same scan regardless of which norm controls stopping.

`full_scan_time` is the wall time of all complete-edge traversals in one refinement iteration. A fused feasibility, dual-violation, and candidate-selection pass is counted once. When feasibility and violation detection require separate complete-edge passes, their times are added. Candidate deduplication, active-support insertion, and budgeted pruning are excluded.

## Dual-violation insertion

Adds full-edge dual violators to the active support. This is the first part of the paper's support update.

## Budgeted pruning

Controls active-support size while protecting the feasible basis, current primal support, and newly detected violating edges. `support_budget_factor` corresponds to the paper budget factor $\beta$.

## Active support

The paper notation is $\mathcal{N}$, consistent with HELLO; use it also in restricted operators and hierarchy-level support indices.

The sparse set of edges on which the restricted OT problem is solved. Prefer `active support`; avoid `active set` and `working set` in documentation.

## HELLO costs

The public cost names are `l2^2`, `l1`, `l2`, and `linf`. `l2^2` is lowered internally to a bilinear representation without materializing the dense cost matrix. Internal representation names are not public cost types.

## Cost perturbation

An auxiliary cost problem used to obtain dual potentials before final refinement with the original cost. `SolverOptions.cost_perturbation` selects `on`, `off`, or `auto`; this activation policy is distinct from the noise representation (rank-2 or index hash). The public balanced-OT default is `auto`. A solve uses one globally consistent perturbation definition and scale, and can activate it at most once.

After `auto` activates at a hierarchy level, the first perturbed-cost LP certificate is compared with the triggering original-cost certificate at the same level. The **dual-feasibility jump ratio** is `perturbed_first / original_last`. A ratio greater than or equal to 10, or a non-finite perturbed certificate, discards the transferred current-level state and restarts the complete perturbed hierarchy from its coarsest level. This safety restart reuses the same perturbation scale and seed; it does not resample the cost. The probe and restarted traversal belong to one `perturbed` cost stage.

Cost-stage names have one contract in both execution paths: `off` and `auto` without activation return `original`; `on` returns `perturbed`, `original_final`; activated `auto` returns `original`, `perturbed`, `original_final`.

For norm costs, index hash uses a shared FP32 approximate-remainder rule in NumPy, Torch, and native CUDA, with fixed operation order and FP64 amplitude multiplication. It is not an exact integer-modulo hash; boundary values may slightly exceed one and are not clipped.
