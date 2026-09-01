# Experiments

The public reproducibility scope contains two experiment families:

- `main_scaling`: runtime and memory scaling.
- `accuracy_runtime_pareto`: accuracy–runtime Pareto evaluation.

Both families use deterministic synthetic Brenier problems by default so they can run without downloading private datasets. Small smoke configurations are intended for installation checks; paper-scale configurations cover `n` in `{65536, 262144}` and `d` in `{4, 32, 256, 2048}`.

Run one case from the repository root:

```bash
python3 -m experiments.main_scaling.run --n 65536 --d 32 --output results/scaling.json
python3 -m experiments.accuracy_runtime_pareto.run --n 65536 --d 32 --output results/pareto.json
```

Both public runners call `hello_ot.solve` directly and use only files included by
`scripts/export_public.py`. The Pareto runner sweeps HELLO's support parameters;
the paper's third-party baseline implementations remain outside this minimal
public closure until each vendored source has auditable upstream provenance and
licensing. Generated results, caches, and plots are not part of the source
distribution.
