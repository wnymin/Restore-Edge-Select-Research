# Restore Edge Select Research

研究有限显存预取额度下，混合模型应从哪个检查点恢复。真机配对中，当浅点已驻留、深点只多覆盖 128 token 时，恢复更深的整状态文件让同一请求慢了 51–85 ms。在已测代码 Agent 轨迹中，近邻候选与同模型跨分支重叠很少同时出现，逐请求成本判断也未显示稳定的总体收益。[代码 Agent 结果](docs/r11-coding-agent-recovery.md) · [真机配对](docs/archive/gpu-studies/gpu-boundary-restore.md)

从仓库根目录运行下列命令即可验收与运行手算轨迹。只需 Python 3.12 或 3.13；不需要 GPU、模型权重或下载轨迹。

~~~sh
python scripts/check_fixture.py
python scripts/simulate.py --prefetch-mib 160 --policy local-cost adaptive offline
~~~

验收覆盖四档预取预算的 36 个手算组合。在 160 MiB 手算轨迹上，离线最优的平均响应为 30.75 ms，逐请求基线为 31 ms。默认自适应策略也选 31 ms，因为该例的均值改善达不到它的改选门槛。

[研究文档导航](docs/README.md) · [统一研究报告](docs/dynamic-gpu-lookahead-report.md) · [研究阶段台账](docs/work-plan.md) · [实验复现](docs/experiment-index.md) · [协作约定](CONTRIBUTING.md)
