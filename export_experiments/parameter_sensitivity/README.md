# Parameter Sensitivity

## Overview

This experiment reproduces the hyperparameter sensitivity analysis in **Section 5.3** and **Appendix Table 8** of the paper using a single seed (`seed=42`).

It performs a one-factor-at-a-time (OFAT) sweep across four dimensions $d \in \{4, 32, 256, 2048\}$ on Gaussian-to-ImageNet transport instances at fixed sample size $n = 262144$ ($2^{18}$).

### Tested Hyperparameters

| Paper Symbol | Description | Public `SolverOptions` | Tested Values (Default in Bold) |
|---|---|---|---|
| $\rho$ | Hierarchy sampling ratio | `split_count` ($1/\rho$) | 0.5, **0.25** (`split_count=4`), 0.125 |
| $\kappa$ | Dual-assignment budget | `assignment_topk` | 8, **16**, 32 |
| $\gamma$ | Detection factor | `pricing_topk` | 1, **2**, 4 |
| $\beta$ | Support budget factor | `support_budget_factor` | 5.0, **10.0**, 20.0 |

## Running the Experiment

### Quick Test

Run a fast test on a smaller problem size ($n=16384, d=4$) comparing two configurations:

```bash
python3 -m export_experiments.parameter_sensitivity.run \
  --n 16384 --d 4 --seed 42 --configuration default split_count_2
```

### Full Benchmark

Run the complete sensitivity matrix defined in `paper_matrix.json` (36 cases: 9 configurations $\times$ 4 dimensions at $n = 262144$):

```bash
python3 -m export_experiments.parameter_sensitivity.run
```

Results are saved to `results/parameter_sensitivity/`.

## Dataset & Caching

The benchmark features are hosted on Hugging Face (`WenzhouXia/hello-ot-imagenet-pca`). On first run, the requested dimension files are automatically downloaded, verified against SHA-256 checksums, and cached under `~/.cache/hello_ot/`.

To use a pre-downloaded or local feature file, pass `--feature-file <path>` with a single `--d`.
