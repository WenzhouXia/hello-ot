# Accuracy–runtime Pareto

## Overview

![Accuracy vs. Runtime Pareto (2x2)](../../docs/figures/accuracy_runtime_pareto_2x2.png)

This experiment reproduces the Figure 2 of the paper using a single seed.

The four methods are:
- HELLO;
- Sinkhorn (OTT-JAX);
- IPOT (KeOps);
- MDOT-TNT (KeOps).

## Environment & Dependencies

Add the experiment dependencies and GPU JAX to the recommended environment:

```bash
bash scripts/install_jax_gpu.sh hello_ot
conda activate hello_ot
```

## Running the Experiment

### Quick Smoke Test

A smaller run to verify the pipeline using the same data and method parameters:

```bash
python3 -m export_experiments.accuracy_runtime_pareto.run --n 128 --d 4
```

### Full Benchmark & Plotting

Run the complete sweep and generate plots:

```bash
python3 -m export_experiments.accuracy_runtime_pareto.run
python3 -m export_experiments.accuracy_runtime_pareto.plot
```

> **Note**: At $n=65536$, computing the default dense POT EMD reference requires over 32 GiB of host RAM. If memory is constrained, switch to the lazy reference:
> ```bash
> CUDA_VISIBLE_DEVICES=0 python3 -m export_experiments.accuracy_runtime_pareto.run --reference lazy
> ```
