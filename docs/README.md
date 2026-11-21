# 研究文档导航

本仓库研究共享 HBM 预取额度下的混合模型缓存恢复边界选择。

| 内容 | 入口 |
| --- | --- |
| 固定执行语义与因素取舍 | [统一研究协议](recovery-research-protocol.md) |
| 分工与验收状态 | [研究阶段台账](work-plan.md) |
| R03 动态因素评估 | [结项](r03-completion.md) · [因素结果序列](r03-factor-sequence.md) · [协作者因素盘点与配对证据](r03-cpu-factor-inventory.md) |
| R04 在线动态选点 | [CPU 验收](r04-final-acceptance.md) |
| R05 静态与动态策略切换 | [计划](r05-strategy-portfolio-plan.md) · [预登记](r05-preregistration.md) · [CPU 报告](r05-strategy-portfolio-report.md) |
| R06 模型与成本适用域 | [结项](r06-completion.md) |
| 早期 CPU 与 GPU 证据 | [实验索引](experiment-index.md) · [统一报告](dynamic-gpu-lookahead-report.md) |

## 归档验收

在仓库根目录运行。

```powershell
python scripts/check_fixture.py
python -m scripts.validation.check_r03_delivery
python -m scripts.validation.check_r03_cpu
python -m scripts.validation.check_r04_snapshots
python -m scripts.validation.check_r04_delivery
python -m scripts.validation.check_r05_delivery
python -m scripts.validation.check_r06_delivery
```

归档证据位于 `docs/evidence/`，R03 过程记录位于 `docs/archive/r03-process/`。完整轨迹回放所需的原始数据及准备方法见[实验索引](experiment-index.md)。

??? Trace B ????????? `python -m scripts.validation.audit_r06_routes` ?? R06 ???
