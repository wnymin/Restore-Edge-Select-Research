# R04 在线动态选点策略设计与验证

Final status (2026-11-20): R04-CPU-01 delivered. No reviewable collaborator handoff artifact was available for R04-CPU-02, so the user took over and completed that line. The planned independent two-line execution did not occur for CPU-02. CPU-03 was triggered by positive-debt snapshots and completed as a same-snapshot ablation; see the report and final acceptance below. The CPU research scope is technically accepted; this does not claim GPU/service validation.

## 因素与独立分工

R03 固定合成子域内，移除增量 RAM／SSD 搬运评分使 100/144 个配置改变选点，平均响应增加 9.0766 ms。F08/F10 覆盖窗口共用一组字段，对应 53/144 个配置改变选点、平均响应增加 1.3012 ms。F13 的现有记录来自覆盖窗口、在途残余和队列债务的联合消融；单独队列债务消融已于用户接替后完成，结果为 56 个固定快照中 3 个改变排序分数、0 个改变最终选点，未观察到独立收益。上述数字沿用 R03 归档口径。[因素序列](r03-factor-sequence.md) · [消融汇总](evidence/r03-factor-ablations-summary.json) · [实现](../scripts/validation/study_r03_factor_ablations.py)

| 任务 | 负责人 | 依赖 | 交付与验收 | 状态 |
| --- | --- | --- | --- | --- |
| `R04-CPU-01` F11 成本感知选点 | 用户 | 共同起点、归档成本表 | 策略、充分性论证、成本近盲消融、公开轨迹留出、决策开销与遗憾审计已交付，见 [分线报告](r04-cpu-01-cost-aware.md)。 | 2026-11-20 已完成 |
| `R04-CPU-02` F08/F10 coverage-window-aware selection | User (took over collaborator line, 2026-11-20) | Shared baseline, archived cost table | Algorithm, applicability proof, crossover fixture, trace ablation, holdout replay and decision-cost evidence delivered; see [report](r04-cpu-02-coverage-window.md). | Completed by takeover; no independent collaborator artifact was available for review. |
| `R04-CPU-03` F13 queue-debt proxy | User takeover, 2026-11-20 | R04-CPU-02 snapshots | Independently ablated queue debt at fixed snapshots, varying capacity, preceding boundary, head compute and arrival. See the fixed-snapshot evidence and CPU-02 report. | Completed; no changed final action in 56 snapshots; no standalone benefit claimed. |
| `R04-INT-01` final synchronization and acceptance | User | R04-CPU-01 and takeover-completed R04-CPU-02 | Cross-checked common event semantics, certificates, fixtures, heldout evidence, and recorded applicability/negative results in [final acceptance](r04-final-acceptance.md). | Completed 2026-11-20; stage accepted for the CPU research scope. |

### 用户线 `R04-CPU-01`

从 `entry` 的仅计算下界出发，按候选依赖的对象并集计算尚需的 RAM／SSD 搬运成本，计入必要复制和后缀续算。用合法的最早消费约束推导不可覆盖的等待下界，再据此筛选或排序候选。已经就绪的对象、共享依赖与在途工作按对象身份核对。证明明确哪些操作可重叠、哪些位于同一关键路径，避免把可覆盖时间重复计入下界。重叠或释放顺序无法安全界定时，交给原执行器复核。对照 `entry` 原下界与推演次数相同的排序版本，验证新增成本信息是否减少推演或改变预算受限时的选点质量。

用固定快照构造搬运与续算成本接近的候选，分别改变决策可见的成本参数。检验排序翻转位置、平局和分层遗憾。成本分量的消融只作用于选点器，执行器在一组对照中使用同一成本真值。跨工况的参数变化与估计误差分别评价。

### 协作者线 `R04-CPU-02` 与 `R04-CPU-03`

用归档 F11 成本表，直接从决策快照推导已提交前序工作的剩余计算、候选依赖的可搬运时段和最早消费时点。设计覆盖窗口感知的选点策略，在可判定的子域用事件次序或状态等价证明候选排序、平局结果或遗憾上界。容量不足时保留合法候选和队首按需加载路径。释放、重试、同刻次序及在途交接能改变消费时间时，细化区间或触发完整求值。

用固定快照只改变前序可覆盖工作，构造同一对候选跨过遮蔽临界点的配对案例。联合消融 F08/F10，再在公开轨迹中检查覆盖收益能否抵消计算窗口与复核的开销。F13 条件满足时保持覆盖和在途状态，在评分中只移除队列债务。汇报它在不同可见队列规模下的增量影响。

## 共同起点与理论口径

- 两线各自从 `ad5250b1675380b1ec066283644a6134fae041cf` 获取 R03 归档、原执行器和基线。以本计划作为共同口径。归档成本、夹具和轨迹复现入口均已在仓库，双方可各自完成需求定义、实现和实验。[实验索引](experiment-index.md) · [策略基线](trace-oracle.md)
- 固定 Qwen3.5-4B CPU 元数据事件模型、精确检查点、缓存放置、单计算槽、串行传输、共享暂存额度、首次入窗选点及原事件优先级。特征取自选点时可见状态和已登记的成本参数。未来到达和事后完成时刻用于离线评价。[研究协议](recovery-research-protocol.md)
- 固定快照以原执行器对**全部合法候选**完整求值为参照。固定已提交边界、可见窗口和后续边界规则。共同实验沿用 `entry` 的后续 `local-cost` 口径，目标为窗口内响应和，同分取更长边界。理论按单次决策陈述适用域，整条轨迹上的闭环效果由实验评价。
- 每线给出至少一个能通过执行器核对的充分条件或有界近似结论。若候选真实窗口响应和 $J(c)$ 有有效区间 $[L(c),U(c)]$，选中候选 $a$ 后，可用 $U(a)-\min_c L(c)$ 给出相对完整求值的单步遗憾上界。精确选点还需证明对每个竞争候选 $c$ 有 $U(a)<L(c)$，或在弱不等式下核对平局优先级。区间覆盖、未精确复核的候选、成本参数适用域和拒判路径均需在证明中交代。
- 每线提供能区分因素作用的反例或临界配对。在合法同快照上改变目标因素后，参照发生排序或边际收益变化时，策略应在声明的子域正确反映变化，或给出覆盖真实损失的上界。记录快速判定覆盖率和回退率。只在回退时得到正确结果的样本计入回退覆盖。

## 实验与验收

1. **固定快照。** 从手算夹具、R02 边界证据和本线最小构造提取候选。覆盖容量临界和同刻两侧、共享和在途状态、近似平局。完整求值逐条核对候选排序、平局、遗憾上界及因素消融。复现任何证书违例并缩小声明域或修正判据。[R02 验收](r02-cross-review.md)
2. **闭环回放。** 各线用归档公开轨迹分别冻结校准和留出清单，记录轨迹、窗口或请求范围、种子、成本档和与历史实验的重叠。两线均保留归档 `adaptive-holdout-109` 配置作为共同回归，新增留出覆盖常规与受压工况。对照 `local-cost`、`entry`、原有快照评分、本线完整策略及仅移除目标因素的版本。凡声明因素可联合省略的子域，增加对应联合消融。
3. **验收指标。** 固定快照报告动作、平局、实测遗憾、证书上界及回退率。闭环报告零开销平均响应、P95、最大单请求退化、决策开销和净收益。状态采集、判定、推演和回退均计入开销。有离线最优区间时同步给出。首次查看留出结果前，各线登记目标、尾部容忍预算和配对汇总规则。采用策略以相对 `entry` 有含开销净收益且尾部满足预算为条件。反例、覆盖狭窄或净收益不足均可作为可复现的研究交付，汇总时按实际证据登记达成度。

## 独立交付、接替与一次验收

- 双方各自研究并自验。公共执行器保持共同版本，临时接口适配随本线提交。脚本、结果、报告分别用 `r04-cpu-01` 或 `r04-cpu-02` 标识，F13 的条件实验随协作者线提交。定案前的例程记录保存在各自分支或文件范围。协作安排只有最终一次同步和验收。
- 协作者在本线达到可复现节点时保存代码版本、实验配置、复现命令、已完成项和剩余事项，供用户按需查看。**用户可随时依据进度接替协作者的全部或部分任务**，无需等待阶段末。优先从最近可复现版本续做，必要时从共同起点接手。在阶段计划登记日期、任务原编号、最新可用版本、接手范围、原因和影响。指定接手范围的写入负责人，保留已有证据和贡献记录。接手时可单向记录负责人变更，最终仍统一验收一次。
- 每线提交一份可复算成果，含算法说明、证明与适用域、版本、冻结清单、命令、结果及反例。先完成的分线可独立登记交付。`R04-INT-01` 在两线就绪后一次核对共同输入上的理论、实验和采用条件。汇总可保留两套策略及各自适用域。

Takeover record (2026-11-20): no collaborator branch or reproducible CPU-02 handoff artifact was present in the shared workspace. User assumed ownership from the recorded R04 common baseline and completed the coverage-window line. The originally planned independent, non-communicating two-line execution therefore did not occur for CPU-02; final acceptance is a technical review of the available implementation/evidence, not an independent replication. CPU-03 was triggered and completed via a queue-debt-only ablation.
