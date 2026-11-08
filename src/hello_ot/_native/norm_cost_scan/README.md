# Sparse norm pair costs

This extension retains only `fused_gcost_pairs`: FP64 costs for selected L1,
L2 and Linf edges, with existing rank2/index-hash perturbations. Each warp
computes one edge without materializing an $E\times d$ difference array.

Full scans, top-k selection and certificate-only reductions use the production
unified backend in `../generic_scan/`. The retired norm full-scan templates,
30/32D specialization and column-partial headers are removed. The native module
name is retained for the sparse-pair ABI; loading unified scans does not load it.
