# Parameter sensitivity

This experiment varies one public `SolverOptions` value at a time while keeping
the other paper defaults fixed. It does not call private refinement functions or
modify the HELLO algorithm.

Run the complete matrix:

```bash
python3 -m export_experiments.parameter_sensitivity.run
```

Run one representative case:

```bash
python3 -m export_experiments.parameter_sensitivity.run \
  --n 16384 --d 4 --seed 42 --configuration default split_count_2
```
