# Restore Edge Select Research

研究有限显存预取额度下的混合模型缓存恢复边界。用事件模拟和百炼请求回放比较选点策略，用归档的硬件读数检验成本假设。当前证据支持在已覆盖的常规成本条件下保留逐请求选点 `local-cost`；队列剪枝作为实现优化。

从仓库根目录运行下列命令即可验收与运行手算轨迹。只需 Python 3.12 或 3.13；不需要 GPU、模型权重或下载轨迹。

~~~sh
python scripts/check_fixture.py
python scripts/simulate.py --prefetch-mib 160 --policy local-cost remaining-wait-fast offline
~~~

验收覆盖四档预取预算的 36 个手算组合。在 160 MiB 手算轨迹上，逐请求最低成本为 31 ms，剩余等待与离线穷举均为 30.75 ms；这是模拟预填充时间。

| 证据 | 入口 |
| --- | --- |
| 模拟范围、状态与计费规则 | [模拟器运行、规则与手算验收](docs/phase1.md) |
| 真实服务的归档读数与离线外推 | [驻留前缀与冷恢复](docs/gpu-boundary-restore.md) |
| 负载规律、策略与反例 | [CPU 研究](docs/boundary-findings.md) |
| 成本依据、保留数据和复现命令 | [实验口径与复现](docs/experiment-index.md) |

[研究分工](docs/work-plan.md) · [协作约定](CONTRIBUTING.md)
