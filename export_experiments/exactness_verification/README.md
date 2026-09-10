# Exactness Verification

## Overview

This experiment reproduces the numerical exactness results in **Table 1** of the paper using a single seed (`seed=42`).

## Running the Experiment

### Quick Test

Run a fast single case ($n=16384, d=4$) to verify HELLO:

```bash
python3 -m export_experiments.exactness_verification.run --n 16384 --d 4 --seed 42 --method hello
```

### Full Benchmark

Run the complete matrix defined in `paper_matrix.json` ($n=65536, d \in \{4, 128, 4096\}$):

```bash
python3 -m export_experiments.exactness_verification.run
```

Results are saved to `results/exactness_verification/`.

## Baseline Dependencies

- **HELLO**: Requires only the core package with native CUDA backend.
- **Sinkhorn (OTT-JAX)**: Install with `bash scripts/install_jax_gpu.sh hello_ot`.
- **HiRef**: Optional third-party baseline. Clone its repository into `third_party/HiRef` if evaluating it; otherwise, it will be skipped automatically.
