# R02-INT-01：最终复算与汇总

2026-11-19 完成阶段末复算，R02 按限定子域验收。现有恢复边界选择策略继续使用完整求值。

## 验收结论

- **CPU 保真。** 单计算槽、串行传输、固定容量与后继计划下，在同一空闲计算边界比较状态。保留按请求预留的对象集合、后继依赖中的就绪对象、在途剩余时间及与后继有关的在途对象身份。时间平移限于无未来到达且执行规则随时间平移不变的子域，已完成前缀单独计分。设计和合成留出各 81 条候选，精简键与完整键各命中 14 和 3 次，拼回后的完成向量一致。共享检查点用例仅差 1 字节即改变胜者，深减浅的响应和从 +9/4 ms 变为 −17/4 ms。 [CPU 交付与证据](r02-cpu-results.md)
- **CPU 净成本。** 保留原到达时间并触发回退时，54 个用例的无计时注入闭环动作和完成向量与完整参照一致。留出 216 次预测含 135 次回退、54 次命中，事件数从 1371 降到 1344。整合复测中，留出集的选点耗时配对中位差为全可见增加约 226 µs、原到达增加约 123 µs，两组计时注入后的平均响应改善中位数均小于零。原提交保留逐轮记录，整合复测用于核对方向。 [CPU 计时复现](r02-cpu-results.md)
- **GPU 可观测性。** 7941 紧额短窗和长窗合计 32 对全部拒判，宽额 16 对的判断方向与原基线一致。7941 查询均为 `pending`，7943 的无核对照也先 `pending` 后 `done`。活动计算和活动同行搬运的实际剩余时间仍未知。CPU 精确状态键的线上可获得性、多步保证和真实服务性能留作开放问题。 [GPU 留出与探针诊断](r02-gpu-observation.md)

## 分工核对

CPU 交付使用 `3458577` 中已存在的执行器、CPU 入口、配置和旧 GPU 归档。与 GPU 线选用的 `9c20261` 对照，CPU 执行器、对应配置和旧回归的输入一致。GPU 新留出在 CPU 交付合入前已完成。双方均可单独生成并自验最终成果。CPU 报告中的旧 GPU 复算属于可自行完成的前置复用。跨线对照和台账更新由汇总人一次完成。

## 最终交付接口 `R02-case/v1`

选点前表、事后表以 `(case_id, arm)` 连接。配对差与标签用 `case_id`。缺失记 `null` 加原因，无适用量注明不适用，不能从真值反填选点字段。

| 字段 | GPU 表映射 | CPU 交付 |
| --- | --- | --- |
| 案例／配置 | `case_id`、`config.protocol`、`fixture_sha256` | 配对 ID、冻结输入／种子及成本、执行器版本 |
| 边界／额度／身份 | `arm`（16/48 MiB 代理，无 token 位置）、`quota_mib * 1048576`；B/C/D 独立 | 原边界、B 为单位的容量、对象身份与共享、在途和已预留 |
| 队首／钟 | 下个固定消费 B，之后 C/D，非服务 FIFO；`before.query_bracket_ns` 末端为决策时刻，另有主机零点括号 | 原 FIFO 队首、固定后继计划、精确模拟时刻和同刻优先级 |
| 观测／缺失 | `remaining_ms` 是估计窗尾区间，活动计算／活动同行传输为 `null`，本夹具不可观测 | 提供构键用的状态、缺失与适用域外条件，不填 GPU 余量 |
| 事后评价 | `outcomes` 是设备实测；审计 `differences_ms`、`timing`、`observed_delta_ms`、`truth`（`null` 为近似平局） | 完整求值的动作、精确平局及逐请求完成向量；对照的回退与原因；保真后再测开销 |

交接例：`R02-GPU-01/v1:tight-short:13`（紧额窗尾未覆盖、拒判）和 `R02-GPU-01/v1:wide-short:13`（宽额、无准入阻塞的对照）。[GPU 报告](r02-gpu-observation.md)附数据与方法。GPU 只能补充敏感性工况，CPU 事件、目标和精确平局不照搬。

[可机器读取的交接样例](evidence/r02-cross-review-v1.json)包含上述两组 GPU 案例，以及 120 MiB 下不共享／仅共享 KV 的两组 CPU 开发案例。`observations` 和 `outcomes` 以 `(case_id, arm)` 一对一连接，`evaluations` 单列。GPU 两个候选使用同一条选点前查询，窗尾区间从该观测和 7449 校准重新计算，不读事后真值；CPU 用模拟时钟上的精确状态。两者的 `remaining_kind` 不同，不能直接换用。CPU 的 `queue_head` 是 A，`decision_target` 是 B；两臂 B 均在 30 ms 开始计算，但只有 B=576 臂释放已预留容量，B=0 的 `release_ms` 为 `null`（不适用）。GPU `release_kind` 则标明原夹具的暂存准入标记，不能与 CPU 的预留释放混用。CPU 样例不是协作者回件或独立留出。

## 自验入口与阶段末汇总

CPU 最终交付可用下方首条命令完整复算。另保留[完整求值开发参照](r02-cpu-reference.md)供回归。

```powershell
python -m scripts.validation.check_r02_cpu
python -m scripts.gpu.experiments.study_r02_gpu_observation --check docs/evidence/r02-gpu-observation-7941-audit.json
python -m scripts.cpu.lookahead.staging.study_lookahead_dynamic_release --check docs/evidence/dynamic-lookahead-dynamic-release-7884.json
python -m scripts.cpu.lookahead.staging.study_lookahead_transfer_order --check docs/evidence/dynamic-lookahead-transfer-order-7820.json
python -m scripts.gpu.analysis.r02_probe_ablation docs/evidence/r02-gpu-probe-ablation-7942.csv
python -m scripts.gpu.analysis.r02_marker_visibility docs/evidence/r02-gpu-marker-visibility-7943.csv
python -m scripts.validation.export_r02_handoff --check docs/evidence/r02-cross-review-v1.json
python scripts/check_fixture.py
```

- 两项 CPU 回归通过，不当作 R02 新留出。7884 共 64 个切面，145 MiB 见证的完成向量为 `[30,137/4,35,157/4]` ms，C 在 30 ms 重试预留；7820 同汇总不同顺序，代价差为 `11/24` 和 `-13/24` ms，获胜边界分别为 0 和 1152。
- CPU 最终案例与 GPU 对照按各自的案例 ID 检索。CPU 的容量临界表和 GPU 的窗口对照支持机制层面的比较，分别使用模拟响应和夹具设备完成时间。
- 阶段状态及交付索引见[阶段台账](work-plan.md)。
