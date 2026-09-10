# Paper experiments

This directory contains the public reproduction entry points. Its name remains
`export_experiments` in the exported repository, so commands and imports are
identical before and after export. `experiments/` is reserved for exploration;
`paper_experiments/` contains formal internal experiments and shared baseline
modules. Public entry points should reuse exportable baseline/evaluation code
instead of maintaining independent copies of those implementations.

The public repository contains four representative experiment families:

- `main_scaling`: native HELLO scaling on Gaussian-to-ImageNet PCA features.
- `exactness_verification`: exactness against a known Brenier permutation.
- `parameter_sensitivity`: one-factor sweeps over public `SolverOptions`.
- `accuracy_runtime_pareto`: HELLO, Sinkhorn, IPOT and MDOT-TNT sweeps with EMD references.

The experiment scripts are repository artifacts; they are not installed into
the `hello_ot` wheel. Run them from the repository root with `python3 -m`.

All public paper experiments use seed=42 only, including PCA fitting and
Gaussian source generation. Future public experiments must follow this rule.
Main scaling and parameter sensitivity download a separate PCA file for each
requested dimension (4, 32, 256, 2048), about 2.29 GiB for all four.
Generation and publication instructions are in `main_scaling/README.md`.
Automatic downloads require a generated checksum manifest and pinned release. The smallest main-scaling
case is also a convenient installation check:

```bash
python3 -m export_experiments.main_scaling.run --n 16384 --d 4 --seed 42
```

Exactness needs no external data:

```bash
python3 -m export_experiments.exactness_verification.run \
  --n 16384 --d 4 --seed 42 --method hello
```

## Environment & Dependencies

To run all reproduction experiments (including baseline solvers and Pareto plotting):

```bash
bash scripts/install_jax_gpu.sh hello_ot
conda activate hello_ot
```

HELLO-only experiments (`main_scaling`, `exactness_verification`, `parameter_sensitivity`) require only the core package. `install_jax_gpu.sh` adds GPU OTT-JAX, PyKeOps, and plotting tools to the same environment without replacing PyTorch's cuDNN. HiRef is an optional third-party checkout documented in `third_party/README.md`.

Formal HELLO timings always use `SolverOptions(backend="native")`. The portable
PyTorch backend is intentionally not used for paper timing.
