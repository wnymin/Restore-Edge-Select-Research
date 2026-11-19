# R02-INT-01：案例交接与交叉复算

2026-11-19：GPU 新留出本机复算通过；CPU 新回件和协作者独立复核尚未收到。本页不代替阶段验收。

## 交换接口 `R02-case/v1`

选点前表、事后表以 `(case_id, arm)` 连接。配对差与标签用 `case_id`。缺失记 `null` 加原因，无适用量注明不适用，不能从真值反填选点字段。

| 字段 | GPU 表映射 | CPU 回件 |
| --- | --- | --- |
| 案例／配置 | `case_id`、`config.protocol`、`fixture_sha256` | 配对 ID、冻结输入／种子及成本、执行器版本 |
| 边界／额度／身份 | `arm`（16/48 MiB 代理，无 token 位置）、`quota_mib * 1048576`；B/C/D 独立 | 原边界、B 为单位的容量、对象身份与共享、在途和已预留 |
| 队首／钟 | 下个固定消费 B，之后 C/D，非服务 FIFO；`before.query_bracket_ns` 末端为决策时刻，另有主机零点括号 | 原 FIFO 队首、固定后继计划、精确模拟时刻和同刻优先级 |
| 观测／缺失 | `remaining_ms` 是估计窗尾区间，活动计算／活动同行传输为 `null`，本夹具不可观测 | 提供构键用的状态、缺失与适用域外条件，不填 GPU 余量 |
| 事后评价 | `outcomes` 是设备实测；审计 `differences_ms`、`timing`、`observed_delta_ms`、`truth`（`null` 为近似平局） | 完整求值的动作、精确平局及逐请求完成向量；对照的回退与原因；保真后再测开销 |

交接例：`R02-GPU-01/v1:tight-short:13`（紧额窗尾未覆盖、拒判）和 `R02-GPU-01/v1:wide-short:13`（宽额、无准入阻塞的对照）。[GPU 报告](r02-gpu-observation.md)附数据与方法。GPU 只能补充敏感性工况，CPU 事件、目标和精确平局不照搬。

## 复算和待回件

CPU 可从[完整求值开发参照](r02-cpu-reference.md)直接开工；开发用，不算独立留出。

```powershell
python -m scripts.gpu.experiments.study_r02_gpu_observation --check docs/evidence/r02-gpu-observation-7941-audit.json
python -m scripts.cpu.lookahead.staging.study_lookahead_dynamic_release --check docs/evidence/dynamic-lookahead-dynamic-release-7884.json
python -m scripts.cpu.lookahead.staging.study_lookahead_transfer_order --check docs/evidence/dynamic-lookahead-transfer-order-7820.json
python -m scripts.gpu.analysis.r02_probe_ablation docs/evidence/r02-gpu-probe-ablation-7942.csv
python scripts/check_fixture.py
```

- 两项 CPU 回归通过，不当作 R02 新留出。7884 共 64 个切面，145 MiB 见证的完成向量为 `[30,137/4,35,157/4]` ms，C 在 30 ms 重试预留；7820 同汇总不同顺序，代价差为 `11/24` 和 `-13/24` ms，获胜边界分别为 0 和 1152。
- `R02-CPU-01` 协作者回件须含充分判据或可复现反例、冻结输入与逐请求核对；保真后才比较净耗时。收到后由本方独立复算其新增证据。
- `R02-GPU-01` 请协作者独立运行 GPU 审计命令，核对上述两例及失败边界；本方自复算不代替交叉复核。在结论未覆盖的工况，不换策略，不跳过完整求值参照。交叉复核后更新[阶段台账](work-plan.md)。
