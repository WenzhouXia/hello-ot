# HELLO

HELLO 是一个面向大规模平衡最优传输的点云求解器。默认 `native` backend 是论文实验使用的 CUDA 实现；用户显式选择 `torch` backend 后，可以只依赖普通 PyTorch 算子在 CPU 或 CUDA 上运行。安装或加载 native 失败时绝不会静默 fallback 到 Torch。

公开接口保持简洁：

```python
import numpy as np
import hello_ot

rng = np.random.default_rng(0)
source = rng.normal(size=(2048, 32))
target = rng.normal(size=(2048, 32))

result = hello_ot.solve(
    source,
    target,
    options=hello_ot.SolverOptions(backend="torch", torch_device="auto"),
)
print(result.objective)
```

数组调用默认使用均匀质量与平方欧氏代价，也可以直接传入 `source_mass`、`target_mass`，并用 `cost="l1"`、`"l2"` 或 `"linf"` 选择其他代价。普通用户无需构造配置；论文实验需要的少数高级选项集中在扁平的 `hello_ot.SolverOptions` 中。LP tolerance 与全局 dual-feasibility tolerance 固定为论文设置，不作为公开参数。

## Portable 安装

默认安装不编译 C++/CUDA 扩展：

```bash
python3 -m pip install .
python3 examples/quickstart.py
```

quickstart 使用显式 `backend="torch"`，规模为 `2048×2048, d=32`。它在 64 核服务器 CPU 上端到端约 7.5 秒、host RSS 约 700 MiB；普通电脑可能更慢，其时间不能代表论文 native backend。

## Native wheel

`v0.1.0-rc2` 的目标环境为 Linux x86-64（glibc 2.29 或更新）、Python 3.10/3.11、PyTorch 2.5.1+cu118，并包含 `sm_80`、`sm_86`、`sm_89`、`sm_90` 与最高目标 PTX，覆盖 A100、RTX 3080 Ti/3090、RTX 4060 Ti/4090 和 H100。这个候选版本使用平台相关的 `linux_x86_64` tag，不宣称是 manylinux wheel。

先安装指定 PyTorch，再安装与 Python 版本匹配的 GitHub Release wheel：

```bash
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --no-deps --only-binary=:all: ./hello_ot-0.1.0rc2-*.whl
python3 -m hello_ot.diagnose --require-native
python3 examples/verify_native.py
python3 scripts/validate_release.py --backend both --output hello_ot_validation.json
```

WSL2 使用同一个 Linux wheel，CUDA driver 来自 Windows 主机；不要在 WSL2 内安装 Linux NVIDIA display driver。首版不支持原生 Windows Python。

portable `torch` backend 可以使用包元数据允许的较新 PyTorch；预编译 native wheel 则与上述发行矩阵的 ABI 绑定。PyTorch/CUDA runtime 不匹配时，程序会在加载扩展前明确报错。

需要从源码构建时：

```bash
python3 -m pip install '.[native-build]'
scripts/build_native_wheel.sh dist
```

源码构建需要 CUDA 11.8、兼容的 C++ compiler、Ninja 和 pybind11；安装预编译 wheel 不需要这些工具。

算法与代码的对应关系见 `docs/algorithm.md`，公开实验入口见 `experiments/README.md`。
