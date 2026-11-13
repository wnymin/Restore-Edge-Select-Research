# Restore Edge Select Research

用 CPU 事件模拟研究：多个请求共享有限显存预取额度时，如何选择混合模型前缀缓存的恢复边界。第一阶段固定 Qwen3.5-4B 文本主干，模拟器只处理缓存对象和事件，无需 GPU 或模型权重。

在仓库根目录运行（Python 3.10+，仅使用标准库）：

```sh
python scripts/check_fixture.py
```

该命令检查首条轨迹的候选依赖、对象字节数、预算和预期结果表。模拟器尚未实现；[手算轨迹](docs/phase1-hand-trace.md)及 [JSON](fixtures/phase1-hand-trace.json) 提供了第一批验收用例。实现规则见[第一阶段实验约定](docs/phase1.md)，协作方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。
