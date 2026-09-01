# Inner-product scan extension

该扩展服务于 `l2^2` 的双线性内部表示。

- directional scan：供 initialization 的 dual propagation、c-transform seed 和 dual assignment 使用；
- bidirectional scan + certificate：供 refinement 的 dual-violation detection 与 dual-feasibility check 使用；`k<=8` 使用一遍 fused traversal，`k=16/32` 在专用 column reduction 完成前使用三遍 streamed custom traversal；
- `k` 支持 `1..32`，Python planner 映射到 `1/2/4/8/16/32` template bucket；
- Python 层只让一侧完整 feature 常驻 GPU，另一侧按 chunk 流式上传。

custom extension 不可用时直接报错；公开实现不包含备用 scan backend。
