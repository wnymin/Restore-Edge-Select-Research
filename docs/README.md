# 实验仓库文档导航

本仓库研究共享 HBM 预取额度下的混合模型缓存恢复边界选择。

| 想查什么 | 入口 |
| --- | --- |
| 研究范围、固定语义与因素取舍 | [统一研究协议](recovery-research-protocol.md) · [动态因素优先级](dynamic-factor-priority.md) |
| 研究结论、证据与适用边界 | [统一研究报告](dynamic-gpu-lookahead-report.md) |
| R01 至今的分工、状态与验收 | [研究阶段台账](work-plan.md) |
| R07：冻结主干后的泛化检验 | [研究设计与当前状态](r07-entry-replay-generalization.md) |
| 模拟、策略对照与实验复现 | [单槽模拟基线](simulator-baseline.md) · [选点与离线最优](trace-oracle.md) · [实验索引](experiment-index.md) |
| 单项实验、计划与预登记 | [过程文档归档](archive/README.md) |

本层保留综述和阶段验收；单项实验和过程记录在 `archive/`，验收脚本使用的数据在 `evidence/`。

## 归档验收

在实验仓库根目录运行：

```powershell
python scripts/check_fixture.py
python -m scripts.validation.check_r03_delivery
python -m scripts.validation.check_r03_cpu
python -m scripts.validation.check_r04_snapshots
python -m scripts.validation.check_r04_delivery
python -m scripts.validation.check_r05_delivery
python -m scripts.validation.check_r06_delivery
```

完整轨迹回放所需的原始数据和准备方法见[实验索引](experiment-index.md)。R06 路径分布可用 `python -m scripts.validation.audit_r06_routes` 核对，范围见 [R06 结项](r06-completion.md)。
