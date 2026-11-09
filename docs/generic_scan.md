# Unified CUDA scans

`hello_ot` and `hier_uot` use one CUDA scan backend for all four costs,
mixed precision and FP64, initial selection, pricing and certificate-only checks.
There is no runtime backend selector or retained legacy scan implementation.
This applies to streamed inputs and `gpu_resident=True` alike.

Matching native extensions must be rebuilt from this checkout. Previously
published wheels do not acquire the unified modules automatically. Research
source-mode validation uses `HELLO_OT_SOURCE_BUILD=1`; release builds use the
repository's pinned native-wheel profile.

| Interface | Coverage |
| --- | --- |
| Norm initial | L1/L2/Linf, FP32/FP64 accumulation, integer k=1–32 via six buckets; fixed-cost statistics and rank2/index_hash perturbation |
| Norm pricing | Both directions and full L2/Linf statistics, all six buckets and both precisions |
| Certificate-only | Complete reductions without candidate buffers, optional denominator, raw and norm costs |
| Raw initial/pricing | FP32/FP64 features, general FP64 offsets and biases, dot scale 1 or 2 |
| Directional raw API | Compatibility name forwarding to unified initial selection |
| GPU-resident scans | Four costs, both precisions including finest-FP64 levels, perturbation and directional initialization |
| Sparse pair costs | Separate retained CUDA kernel for selected edges; no full scan |

GPU residency controls feature/state transfers independently of directional
selection. Initial selection requests the needed direction before or after a
potential update; pricing returns both directions and the full certificate in
one traversal. Feature arrays, candidates and statistics remain on GPU; only
existing scalar convergence metrics are read back. Integer requests 1–32 round
up to 1/2/4/8/16/32 and truncate the returned width.

Mixed precision uses FP32 feature arithmetic and FP64 potentials, candidates and
statistics. Solver stopping tolerances are unchanged. The existing `hier_uot`
FP32 configuration still requires `refinement_forcing_check=False`.

The former inner-product extension and the norm full-scan CUDA kernels are
removed from production sources and native-wheel builds. The norm extension
module now exposes only sparse pair costs. `ScanExtension` is an ABI adapter to
the unified implementation, not a dispatcher between implementations.
Invalid requests raise errors instead of falling back. Existing raw profiling
arguments remain accepted for caller compatibility; the retired four-phase
CUDA timing return is a zero placeholder. Wall-clock stage timing remains.

GPU-resident squared-cost perturbations retain separate low-rank dots. Their
FP64 combination is inserted into each raw dot tile, changing final FP64
association relative to the former resident Torch path while preserving dot
precision and the noise formula. Norm L2 keeps the validated mixed-precision
expansion; FP64 uses direct distances. Resident FP64 coordinates are cast only
for current tiles, avoiding a full FP64 point-cloud copy.

With `cost_precision="fp64"`, norm coordinates retain the native FP32 storage
format and are promoted before subtraction. Differences, accumulation, costs,
scores and statistics use FP64. This shares the coordinate-tile framework with
mixed precision; the initial FP64 policy uses one 32-by-64 point-pair geometry
at all ordinary dimensions, followed by the existing matrix selection queues.
L2 uses direct squared differences and FP64 sqrt, without a norm expansion.
Raw FP64 scans consume FP64 coordinates and use cuBLAS DGEMM, preserving caller
costs, general biases and score association. Native raw initial streaming and
small-pricing wrappers share their implementations across both dot types.
FP64 norm scratch planning includes eight bytes per base cost and sixteen bytes
when the base and perturbed tiles coexist.

For perturbed norm scans, generate a bounded FP32 base-cost tile and add the
established CUDA noise in FP64. Global source/target indices, seed, coefficient
conversion and explicit rounding follow the old formulas. The tile planner
accounts for the simultaneous FP32 base tile and FP64 perturbed tile (12 bytes
per pair before candidate buffers). No noisy full global matrix is retained.
Initial queues can accumulate the complete squared-cost sum and absolute maximum
while selecting candidates; disabled statistics use separate compile-time paths.
Certificate-only calls reduce cost tiles directly without maintaining top-k or
allocating partial candidates. Top-32 perturbed pricing uses the generic fused
bidirectional path rather than the legacy three-scan decomposition.

The FP64 perturbed cost is formed as `base + noise`. Norm initial evaluates
`g - (base + noise)` while old norm initial subtracts base and noise successively;
this is an FP64 association difference, independent of the unchanged noise
formula. Perturbed pricing and certificate-only evaluate `(f + g) - cost`, matching
the previous norm certificates. Base-cost reductions retain the existing mixed-precision
comparison tolerances. Sparse pair-cost evaluation retains its original FP64
arithmetic and independent implementation. Raw arithmetic is unchanged.

The raw path consumes caller-supplied costs and biases; it never assumes they
are squared norms. FP32 dots use native cuBLAS SGEMM with the same math mode as
the previous raw implementation, independently of PyTorch's global TF32 option.
Costs and scores retain the original FP64 evaluation order. L2 retains the
validated FP32 norm expansion and a full-tile cancellation check with direct
distance fallback. Its engineering comparison tolerance is separate from the
solver stopping tolerance.

Both feature axes can be tiled. Candidate indices are shifted only when valid;
missing candidates retain negative sentinels. Row and column candidates merge
incrementally, and full statistics count each pair once. Scratch planning uses
at most 1 GiB or one quarter of currently reusable device memory, including an
allowance for matrices, partial candidates, merges and tile coordinate copies.
Contiguous resident coordinates do not consume an additional copy allowance.
Device matrix scans use a common budget model and search up to 8192 query rows
by 65536 database columns. They prefer wider database tiles, then larger query
tiles that fit. These are candidate bounds, not a fixed geometry for each
ordinary dimension. The model includes live cast, noise and sum matrices when
perturbation is requested.
Raw initial streams caller-sized tiles through one reusable dot buffer with
the same point-count cap: at most 64 MiB for FP32 and 128 MiB for FP64. There is
no cost-specific exception. Its native loop merges candidates in
place and emits the caller's int32 indices directly. The strided-input path
retains the general adapter. Streamed raw pricing jointly plans uploads,
residency and backend scratch in `scan_planner.py`; its execution adapter uses
the supplied tiles directly, without imposing a second 8192-column cap.
Top-1/2 loops and candidate merges execute natively. Other buckets retain
the tiled adapter while consuming the same already-planned geometry. Resident
and norm entry points select their device tiles once through the same planning
module. Streaming upload batches and CUDA thread-block geometry remain distinct
from distance-matrix tiles.
Final outputs and the caller's resident inputs are separate unavoidable storage.
There is no complete global distance matrix or cross-direction initial reuse.

Repeated device scans reuse deterministic plans for identical shape, layout,
workspace requirements and effective budget, through a bounded 256-entry cache.
Device free-memory samples are reused for at most 100 ms. Growth in active
PyTorch allocations immediately lowers the cached available-memory estimate;
allocation decreases do not raise it above the sampled estimate. Expiry,
exhausted snapshots or an infeasible automatic plan trigger a fresh driver
query. Abrupt external allocations can still cause an allocation failure before
the next sample: an automatically planned scan then discards its local outputs,
invalidates the sample and retries the whole generic scan once. Nested calls
share this retry limit. Explicit workspace budgets and caller-supplied plans
are not rewritten; sustained OOM propagates, without changing scan implementation.
This policy does not reserve or retain an additional workspace tensor.

Raw certificate-only scans execute their two-axis loop natively, reusing one
bounded dot buffer and one statistics accumulator. Both return ABIs share the
same cost/score arithmetic and reduction kernel; legacy statistic ordering is
packed natively without double-to-integer-to-double copies. Coordinate casts
and contiguous copies are restricted to current tiles. Complete certificate
reductions, across all costs and precisions, merge warps inside each block before
one set of global atomic updates per block. Their common grid rule caps work at
eight blocks per SM and at the number needed for the point pairs. It contains no
per-cost or ordinary-dimension tuning table.

Ordinary dimensions use runtime feature tiles. Only 2D and 3D are fixed-dimension
specializations. Dispatch uses a finite cost/dimension-range/k-bucket policy,
including lightweight top-1 initial and top-1/2 pricing. No fallback is selected
merely because a measured low-dimensional configuration is slower.
Unperturbed norm initial top-1 uses a lane-per-pair reduction without per-tile barriers when
the runtime dimension does not exceed warp width. Database coordinates are
transposed for coalesced reads, and database groups supply parallel work based
on SM count. The coordinate copy must fit the shared workspace budget.
This supports both FP32 and FP64 accumulation without materializing a distance matrix; it does not compile
a kernel for an individual ordinary dimension. Larger dimensions retain matrix
or shared-coordinate paths. Local matrix-selection indices use int32, matching
the validated point-count bound; public indices remain int64.
Materialized-matrix top-1/2/4/8 pricing reduces rows and columns separately,
without a full matrix-sized column-candidate intermediate. Initial top-1 reuses
the same row reducer, including optional fixed cost statistics. Both reducers
share `matrix_selection.cuh`, independent of the ordinary coordinate dimension.
Direct-coordinate pricing and larger buckets retain fused selection with
database splits when query blocks alone cannot supply twice the device SM count.
Those splits own disjoint column ranges and full statistics count each pair
once. Warp queue merge
extracts sorted candidates through shuffles rather than an extra shared-memory
staging array.
The small native raw pipeline combines index conversion and statistic packing
in one output kernel.

The five CUDA sources and their shared matrix-selection header under
`src/hello_ot/_native/generic_scan/` are production
sources copied from the validated prototype, with raw-bias and cost
preprocessing adapters. They import no experiment files and add no Triton
dependency. Both source-mode builds and the native wheel build source list
include them. They are loaded only when exercised.

`hello_ot.kernels.generic_scan.scan_routes(reset=True)` returns and resets actual
adapter route counts. Pricing diagnostics and raw initial profiles include
`kernel_backend`. Raw memory estimates describe the planned active geometry;
they remain estimates rather than measured peaks.
Tests cover both solvers, GPU residency, perturbations, precision modes and
all top-k buckets. `scan_routes` remains a diagnostic counter, not a selector.

## Archived comparisons

Commit `be75c37` preserves the completed implementation with its former
comparison selector and the paired experiment scripts. Use an isolated checkout
of that commit to reproduce those runs. The scripts under
`experiments/cost_scan_survey/` and their protocols describe frozen historical
measurements; they must not be used to compare two runtime implementations in
the current checkout. Raw results, inputs, source archives and reports retain
original provenance and are not rewritten by this removal.

The last pre-removal dispatch retest on A100 passed 665 regression tests and
91 paired observations. Seven targeted configurations each used three independent
pairs; other controls used one pair. See batch
`20261108_scan_dispatch_final_a100`. This is kernel/API evidence, not a fresh
complete end-to-end timing grid for the removal commit.
