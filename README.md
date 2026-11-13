# Restore Edge Select Research

用 CPU 事件模拟研究：多个请求共享有限显存预取额度时，如何选择混合模型前缀缓存的恢复边界。第一阶段固定 Qwen3.5-4B 文本主干，模拟器只处理缓存对象和事件，无需 GPU 或模型权重。

在仓库根目录创建环境并运行：

```sh
conda env create -f environment.yml
conda activate restore-edge-select
python scripts/check_fixture.py
python scripts/simulate.py --prefetch-mib 160 --policy local-cost capacity offline
```

验收脚本回放四档预算下的全部 36 个手算组合，并检查共享对象和检查点交接。模拟命令输出 JSON；160 MiB 下，逐请求最低成本策略平均用时 31 ms，容量感知策略和离线穷举均为 30.75 ms。这些数字来自合成轨迹，不能视为实测 TTFT。

[实验环境与运行说明](docs/environment.md) · [第一阶段实验约定](docs/phase1.md) · [手算轨迹](docs/phase1-hand-trace.md) · [协作约定](CONTRIBUTING.md)
