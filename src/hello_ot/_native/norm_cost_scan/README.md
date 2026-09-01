# Norm-cost scan extension

该扩展服务于 `l1`、`l2` 与 `linf` cost，并保持三者共享 Python operator API、分别走 cost-specialized CUDA template。

- directional scan：供 initialization 的 c-transform seed 与 dual assignment 使用；
- bidirectional scan + certificate：一次 pairwise traversal 同时返回双向 violators、L2 certificate 与 L∞ certificate；
- standalone certificate：供只需 full dual-feasibility check 的路径使用；
- `k` 支持 `1..32`，Python planner 映射到 `1/2/4/8/16/32` template bucket；
- Python 层只让一侧完整 feature 常驻 GPU，另一侧按 chunk 流式上传。

custom extension 不可用或问题超出单卡支持范围时直接报错。
