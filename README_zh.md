# MESA 精简开源包

本包包含 MESA 核心实现、论文固定参数、三例自生成的非线性 EIT 测试数据、
运行与绘图示例，以及回归测试。采用 MIT 许可证。

## 一键运行

在解压目录打开终端，使用 Python 3.10 或以上版本：

```bash
python -m pip install -e ".[torch,demo]"
python examples/demo_eit.py --device cpu --output outputs/eit_demo
python -m unittest discover -s tests -v
```

使用 CUDA 时先安装与设备兼容的 PyTorch，再把 `--device cpu` 换成 `--device cuda`。
若不安装 PyTorch，也可以安装 `.[demo]` 后用 `--backend numpy` 运行 CPU 演示。

输出包含真值、初始场、更新后场、最终输出的比较图，以及重建向量和诊断。
算子准备与逐帧重建分开；每帧独立，不输入真值或 SNR，也不使用测试答案缓存。

## 两种实现

- `mesa.torch_backend`：EIT 论文参考实现，支持 CPU/CUDA、掩膜网格、LOOCV/1SE，
  并包含真实 EIT 使用的有符号/无符号分支。
- `mesa.numpy_backend`：论文 ECT/MPI 使用的 NumPy 实现，完整二维矩形网格，
  float64 精度，不要求 GPU。

数值约定并不完全相同。核对论文 EIT 输出时用 PyTorch 分支；NumPy 的 EIT
示例只演示同一机制，不能当作论文的逐元素数值复现。论文的 19.10 ms 对应
RTX 4090 Laptop GPU、四个 CPU 线程，不代表 CPU 或其他设备的运行时间。

## 数据与范围

随包仅有圆、L 型、环形各一例：原论文非线性 FEM 测试集的第一噪声重复，
名义 60 dB；前向网格和重建网格不同。选择用于展示不同形状，没有调参。
`data/eit_examples.npz` 包含算子、索引坐标、观测及展示真值，约 2 MB。

ECT/MPI 的核心代码和参数已经提供，完整原始数据不随包上传；获取来源见
[docs/DATA.md](docs/DATA.md)。此包不包含第三方照片、比较算法、旧稿、历史
搜索结果、训练权重或完整原始数据。三例示例也不用于重算论文总体指标。

详细接口见 [docs/API.md](docs/API.md)，实现来源和数据索引见
[docs/provenance.json](docs/provenance.json)，验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。
