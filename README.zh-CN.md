# HELLO

HELLO 是一个面向大规模点云间无正则离散最优传输（OT）的高性能求解器，通过对偶引导的多层级结构与 GPU 高效并行，以低显存、低耗时、高精度给出稀疏解，单卡 A100/H100 上能求解百万规模、上万维度的 OT 问题。

### 为什么选择 HELLO？
- **高精度稀疏解**：求解标准无正则化 OT 问题，输出严格满足 KKT 阈值的稀疏解（与对应对偶解）；
- **百万级单卡求解**：显存占用保持 $\mathcal{O}(n+m)$，**“能存点云就能算”**，单卡峰值显存仅 $41.6\text{ GiB}$ 即可求解 $n=m=1.28\times10^6, d=8192$ 规模问题；
- **高维也能算**：从低维到高维（$d>10^3$），算法均保持高效；
- **适用于不同 cost**：不仅是 $\ell_2^2$，还支持 $\ell_1, \ell_2, \ell_\infty$ 等一般的代价函数（cost）；
- **双后端开箱即用**：提供纯 PyTorch 后端（保证开箱即用）与 CUDA 原生后端（极致性能）；
- **更多 OT 变体**：开箱支持半离散 OT（Semi-discrete OT）、Gromov-Wasserstein（GW）、非平衡 OT（UOT）。

<table style="width: 100%;">
  <tr>
    <td width="60%" align="center">
      <img src="docs/figures/hello_framework.png" alt="算法流程概览" />
    </td>
    <td width="40%" align="center">
      <img src="docs/figures/accuracy_runtime_pareto_2x2.png" alt="精度与耗时对比" />
    </td>
  </tr>
</table>

## 快速上手

`examples/quickstart.py` 可以直接运行，采用合成的 Brenier 映射问题（$n=m=2048, d=32$）：

```python
import numpy as np
import hello_ot

# 1. 构造自带最优解的测试案例
rng = np.random.default_rng(42)
source = rng.normal(size=(2048, 32)).astype(np.float32)
target = source + 2.0 * np.tanh(source)
target = target[rng.permutation(len(target))]

# 2. 求解最优传输，默认使用 均匀质量分布 与 平方欧氏距离
result = hello_ot.solve(
    source,
    target,
    options=hello_ot.SolverOptions(backend="torch", torch_device="auto"),
)

# 3. 读取结果
print(f"最优目标值: {result.objective:.6f}")
print(f"非零传输个数: {result.solution.values.size}")

# 稀疏传输矩阵 (scipy.sparse.coo_matrix) 与对偶势 (f, g)
x = result.solution.to_sparse_matrix()
f, g = result.solution.source_dual, result.solution.target_dual
```

数组调用默认使用均匀质量与平方欧氏代价，也可以直接传入 `source_mass`、`target_mass`，并用 `cost="l1"`、`"l2"` 或 `"linf"` 选择其他代价函数。

高级选项集中在 `hello_ot.SolverOptions` 中（如 `backend="native", "torch"`）。

> **提示**：仓库内提供完整的验证脚本：
> ```bash
> python3 examples/quickstart.py
> ```
> 脚本会自动完成端到端求解，并输出原始可行性、对偶可行性与 Brenier 解析真值误差。

## 安装指南

### 1. PyTorch：开箱即用（`backend="torch"`）

如果已经安装了 PyTorch，直接安装 HELLO-OT：

```bash
python3 -m pip install .
```

如果尚未安装 PyTorch，或者后续需要测试 native backend 或复现论文实验，
创建推荐的 `hello_ot` 环境：

```bash
bash scripts/create_hello_ot_env.sh hello_ot
```

脚本会安装 Python 3.12、带 CUDA 11.8 runtime 的 PyTorch 2.7.1、
NumPy 1.26.4、SciPy 1.15.3 和 portable HELLO-OT，然后运行 torch backend
quickstart。脚本优先使用 Conda，找不到时使用 Micromamba，并在完成后打印
对应的激活命令。将最后一个参数改成 `hello_ot_test`，即可创建另一个测试环境。

---

### 2. CUDA：极致性能（`backend="native"`）

包含高性能 CUDA 流式扫描算子与 CUDA 原生 LP 求解器
- **支持架构**：NVIDIA GPU（算力 `sm_80`、`sm_86`、`sm_89`、`sm_90`，覆盖 A100、RTX 3080 Ti/3090、RTX 4060 Ti/4090、H100 等）；
- **目标环境**：Linux x86-64（glibc ≥ 2.29）、Python 3.12、PyTorch 2.7.1+cu118。

#### 方式 A：安装预编译 Wheel（推荐，无需本地编译）

完成上面的环境创建后，安装并验证 native wheel：

```bash
bash scripts/install_native.sh hello_ot
```

该命令会用匹配的 native wheel 覆盖 portable 发行包，同时保留
`backend="torch"` 和 `backend="native"`。

#### 方式 B：从源码编译构建

开发环境需要 Python 3.12、PyTorch 2.7.1+cu118、CUDA 11.8 NVCC 与
CCCL 头文件、兼容的 C++ 编译器、Ninja 和 pybind11：

```bash
conda install -c nvidia cuda-nvcc=11.8 cuda-cccl=11.8.89
python3 -m pip install '.[native-build]'
scripts/build_native_wheel.sh dist
python3 -m pip install dist/*.whl
```

### 3. GPU JAX 与论文实验复现

论文核心实验（大规模扩展性、Pareto 曲线、精确度验证与参数敏感性分析）的复现脚本整理在 [`export_experiments/`](export_experiments/) 目录下。将 GPU JAX、OTT-JAX 和其余复现依赖加入同一个环境：

```bash
bash scripts/install_jax_gpu.sh hello_ot
conda activate hello_ot
```

安装脚本会保留 PyTorch 使用的 cuDNN 9，并把 CUDA 11 JAX 需要的 cuDNN 8
放在独立目录中。激活环境时会自动选择正确的 CUDA 11.8 工具和动态库。
Micromamba 环境使用 `micromamba activate hello_ot`。

各实验的具体说明与运行指南请参阅 [export_experiments/README.md](export_experiments/README.md)。
