# Accuracy–runtime Pareto

This representative reproduction uses Gaussian-to-ImageNet PCA features,
N=65536, D=4/32/256/2048, squared-L2 cost and seed=42. It reproduces the trend
with one seed, rather than repeating the paper's multiple-seed experiment.

The four methods are HELLO (cost perturbation off), OTT-JAX Sinkhorn,
IPOT (KeOps), and MDOT-TNT (KeOps). Default sweeps live in `config.json`.
Algorithm adapters are shared with `paper_experiments/baselines/linear_ot`;
they are repository files, not part of the hello_ot wheel. IPOT's dense
implementation is retained for small correctness comparisons.

Install the native HELLO build and experiment dependencies in a dedicated
environment. The tested combination uses Python 3.10, Torch 2.5.1/CUDA 11.8,
NumPy 1.26.4, JAX 0.4.25 with GPU jaxlib 0.4.25+cuda11.cudnn86,
OTT-JAX 0.4.6 and PyKeOps 2.3. After installing the extra, install GPU jaxlib
using JAX's archived CUDA wheels:

```bash
python3 -m pip install -e '.[pareto]'
python3 -m pip install 'jax[cuda11_pip]==0.4.25' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CUDA_VISIBLE_DEVICES=0 python3 -m export_experiments.accuracy_runtime_pareto.run
python3 -m export_experiments.accuracy_runtime_pareto.plot
```

Choose an available GPU on your machine. The runner isolates each configuration
in a subprocess, disables JAX preallocation, and continues after failed cases.
It does not change or install any dependencies automatically.
The worker requires GPU JAX and does not silently use CPU. See the
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#installing-older-jaxlib-wheels)
for archived builds. If JAX reports an older cuBLAS than its build requires,
ensure the worker's library path points to the matching CUDA libraries, rather
than an older system installation. Do not disable JAX's compatibility checks.

A smaller run, using the same data and method parameters:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m export_experiments.accuracy_runtime_pareto.run --n 128 --d 4
```

Use `--feature-file /absolute/path/to/d4_seed42.npy` with one `--d` to read
local data. Otherwise the dimension-specific published artifact is downloaded
and verified. N is configurable up to 262144; seed remains 42.
`--method mdot --parameter 64` runs a single parameter for that method.

The reference is HELLO-warmstarted dense POT EMD. At N=65536, the FP64 cost
matrix alone requires 32 GiB of host memory; solver workspaces require more.
If allocation fails, the runner records a failure and suggests:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m export_experiments.accuracy_runtime_pareto.run --reference lazy
```

There is no automatic switch. A killed reference worker is also reported.
Without a valid reference, no relative-error points are plotted. Reference
files bind the objective to the actual point clouds and masses; each invocation
recomputes its references.

Each method receives a same-shape, same-parameter warmup before measurement.
Solver runtime includes cost preprocessing and transfers; import/JIT warmup,
data preparation, objective evaluation and feasible rounding are excluded.
The error is the relative original-cost objective gap after feasible rounding.
Clearly negative gaps fail validation; roundoff within 1e-6 is treated as zero.
The plot uses a symlog error axis so exact zero can be displayed.

Public results are `results.csv`, `summary.md` and PDF/PNG plots. Each row
contains only method, parameter, N, D, seed, solver seconds, relative error,
status and a failure reason. Request files and worker logs support debugging.
Nonconverged or failed configurations remain in the table and summary.
Do not merge results from different software versions into one output directory.

MDOT-TNT retains its upstream PolyForm Noncommercial license in
`paper_experiments/baselines/linear_ot/mdot_tnt/LICENSE`.
The internal budgeted PyPI-lowmem experiment is not exported.
