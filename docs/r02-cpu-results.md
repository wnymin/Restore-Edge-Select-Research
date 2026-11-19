# R02-CPU-01 后缀等价、留出与净成本

**结论：** 固定后继、无未来到达的模拟子域支持对象级后缀等价判据。粗汇总会漏掉对象身份和搬运顺序，精简键在单次求值和闭环计时中均未显示净收益。本阶段保留完整求值。

从仓库根目录运行 `python -m scripts.validation.check_r02_cpu` 可复算冻结输入、固定计划、后缀键、边界反例和原到达闭环。分项命令在下文，重生成结果前先创建 `outputs/`；计时数值取决于运行机器。

协作者交付见 `46d6b7a`，实验以 `3458577` 的原执行器为基线。旧释放见证 7884（含 64 组审计）和顺序见证 7820 均已复算。7820 的两组汇总量相同，完整求值得到的获胜边界分别为 0 和 1152，给出了粗汇总键失效的反例。

冻结清单含两组各 27 个四请求工况。设计种子 72841，留出种子 72842，搬运时间倍率分别取 1、2、3 和 4、5、7。交叉独立对象、共享 KV、共享 KV 和检查点，容量围绕 B 浅边界与 C 深边界对象并集上下各一字节，C 到达围绕原见证的 30 ms 上下各 1/8 ms。由于修改搬运时长后实际事件时刻可能变化，30 ms 是参考锚点，不能用来宣称已命中真实的就绪或释放同刻。各候选均使用同一容量与成本。

[冻结清单](evidence/r02-cpu-inputs.json) · [设计集逐方案结果](evidence/r02-cpu-design-reference.json)

完整固定计划执行为参照，B 扫 0、576、1152，其他请求边界保持固定。逐方案给出精确完成向量、响应和、预留时间及按响应和再按较深边界选择的胜者。固定计划结果用于核对，状态键另从相同决策快照比较。

~~~sh
python -m scripts.cpu.lookahead.staging.study_lookahead_dynamic_release --check docs/evidence/dynamic-lookahead-dynamic-release-7884.json
python -m scripts.cpu.lookahead.staging.study_lookahead_transfer_order --check docs/evidence/dynamic-lookahead-transfer-order-7820.json
python -m scripts.cpu.lookahead.staging.study_r02_reference --inputs docs/evidence/r02-cpu-inputs.json --out outputs/r02-design.json
python scripts/check_fixture.py
~~~

## B 完成后的后缀等价

另做全可见变体，将冻结清单中的四个请求都置于 0 ms。三条候选从同一快照开始，使用原执行器的 pause_at_head 在 B 完成后暂停。此时没有运行中的计算、待释放的额度、仍被计算持有的对象或主机侧缓存状态。之后的 C、D 边界固定，释放规则为开始时释放，搬运成本固定，没有未来到达。以相同案例内的后缀时间差复用结果，B 及此前已完成请求的时间原样保留。

充分条件是预留的请求与对象集合相同、后继依赖中的就绪对象集合相同、在途剩余时间相同且在途对象的后继身份相同。无关在途对象可去掉身份，但仍须保留其占道时间，并区分无在途的状态。按事件归纳：准入与重试比较相同的预留并集和容量；加载顺序由固定后继边界的依赖列表给出；无关对象不会被后继需求读取，且相同剩余时间保证下一次占道释放同步。两个状态只差起始时刻时，由此得到相同的相对完成向量。

[设计集快照结果](evidence/r02-cpu-design-suffix.json)与[留出集快照结果](evidence/r02-cpu-holdout-suffix.json)各核对 81 条候选。完整和精简键都在设计集命中 14 次、留出集命中 3 次，拼接后的完整向量逐项一致。精简键在该批没有增加命中。汇总键在该批也没发生错误，旧顺序见证仍限制其适用范围。审计运行会执行完整参照，所得命中数不能用来推断净耗时或闭环收益。

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_suffix_keys --out outputs/r02-design-suffix.json
python -m scripts.cpu.lookahead.staging.study_r02_suffix_keys --split holdout --out outputs/r02-holdout-suffix.json
~~~

## 后缀复用的实际成本

在相同全可见窗口上，直接完整求值与两种缓存键按三种模式的六个排列交错运行，每个排列重复四次。每次计量三个候选的总耗时，缓存每次清空。计时包括暂停、构键、缓存读写、命中后的向量拼接以及未命中时续跑。当次计时结束后核对三个候选的逐请求完成向量。未来依赖集合在本实验计时前为共同固定计划准备一次，因此该结果未计入在线构造这一集合的成本。

| 组别 | 完整求值中位数 / P95 | 完整键中位数 / P95 | 精简键中位数 / P95 |
| --- | ---: | ---: | ---: |
| 设计 | 224.05 / 358.80 µs | 271.75 / 478.20 µs | 273.95 / 485.70 µs |
| 留出 | 218.90 / 336.00 µs | 272.75 / 438.10 µs | 274.35 / 429.30 µs |

留出仅有 3/81 条候选命中。其余 78 条候选暂停后续跑，续跑会多进入一次事件循环，低命中率未能抵消该成本。该测量针对单次快照和固定后续计划。两种闭环计时见下文。

[设计计时逐轮记录](evidence/r02-cpu-design-timing.json) · [留出计时逐轮记录](evidence/r02-cpu-holdout-timing.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_suffix_timing --split design --out outputs/r02-design-timing.json
python -m scripts.cpu.lookahead.staging.study_r02_suffix_timing --split holdout --out outputs/r02-holdout-timing.json
~~~

## 模拟释放与就绪同刻

从 7884 固定计划派生九组单通道回放。调整每次回放内固定的 SSD 带宽，使 D 的检查点搬运分别在 B 开始计算、释放预留的 30 ms 之前 1/8 ms、同刻、之后 1/8 ms 完成。容量分别为 C 深边界与 D 尚需对象的并集大小减一字节、恰等于并集、加一字节。B 和 D 的初始预留在九组中均成功，C 初始均未准入。

容量够用时，C 在 30 ms 重试成功；若 D 仍在途，C 虽已预留却须等到占道释放才能搬运。容量少一字节时，三种时序下均无法预取，C 在 B 完成后进入队首并按需加载。这直接区分重试准入与实际就绪后的搬运进度。此项使用独立对象的固定计划，是原执行器的事件顺序对照；涉及共享 KV 和共享检查点的配对见下文。

[精确完成向量与搬运时间](evidence/r02-cpu-release-tie.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_release_tie --check docs/evidence/r02-cpu-release-tie.json
~~~

## 共享对象与释放重试

从冻结成本清单取设计和留出各两种共享布局，对 B 的浅／深边界和 100、130、145 MiB 预算交叉回放。四个请求同时可见。设计集在 B 取浅边界时，100 MiB 下 C 于 B 开始计算时重试成功，共用 KV 只搬运一次；130 MiB 下 D 可先占预留，C 无法预取并在成为队首后重新搬运该 KV。单独提高容量能改变其他请求的准入，因此不能用容量数值代替按请求保存的预留状态。

B 与 C 都取深边界且共享检查点时，两者在 0 ms 都得到预留，共用的 B.state2 只搬运一次。这个检查点对照验证对象去重；发生释放后重试的浅边界对照实际共享的是 KV。逐请求完成、开始、预留与逐对象传输均附在证据中。

[24 条共享对照](evidence/r02-cpu-shared-retry.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_shared_retry --check docs/evidence/r02-cpu-shared-retry.json
~~~

## 在线闭环复核

复用仓库已有的 Coalescence 在线适配器。它按首次入窗选点，用同一可见窗口完整推演作参照，并在每次决策清空后缀缓存。采用冻结清单的全可见变体，设计和留出各 27 个四请求配置。审计逐候选完成向量，闭环逐请求比较选点与完成时间。零开销下均一致。

每配置分别做固定时钟计时和决策耗时注入，各八轮，以种子 72844 选择首轮顺序，之后正反交替。配对选点总耗时中位差显示，精简键在设计集增加 226.65 µs，在留出增加 197.10 µs。注入后平均响应的配对改善中位数分别为 −0.0001 和 −0.0004 ms，最大单请求额外等待分别为 0.0066 和 0.8999 ms。四请求的最近秩 P95 等于最大响应，逐轮记录同时保留。闭环虽观察到缓存命中，计时未支持替换完整求值。

[逐轮闭环结果](evidence/r02-cpu-closed-loop.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_closed_loop --out outputs/r02-closed-loop.json
~~~

## 共享检查点的释放重试

将 7884 的 D 设为 C 的同前缀请求，C、D 都固定选深边界并共享检查点 C.state2。初始时只有 B 的浅边界能预留。容量恰等于 C 的依赖并集或高一字节时，B 于 30 ms 开始计算并释放预留，C、D 同时重试成功，共享检查点只预取一次。容量低一字节时两者均无法预取，队首各按需读取一次检查点。该对照补上了共享检查点参与重试的直接证据，成本为合成值。

[精确逐请求与搬运记录](evidence/r02-cpu-checkpoint-retry.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_checkpoint_retry --check docs/evidence/r02-cpu-checkpoint-retry.json
~~~

## 容量临界案例表

[版本化案例表](evidence/r02-cpu-exchange.json)以案例 ID 连接决策前观测与事后真值。固定 C、D 为 1152，在同一初始状态对比 B 的 576 与 1152。共享依赖少一字节时，深减浅的响应和为 +9/4 ms；达到门槛及高一字节时均为 −17/4 ms。这给出一个排序翻转和一个相邻容量下排序不变的对照。30 ms 是合成模型给定的队首计算时长。

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_exchange --check docs/evidence/r02-cpu-exchange.json
~~~

## 保留原到达时间的固定计划留出

[留出逐方案结果](evidence/r02-cpu-holdout-reference.json)保留清单中 C 的到达时刻，27 个配置共 81 条计划。深边界、浅边界及重算分别胜出 7、13、7 次。同一案例的全可见变体中，有 19/27 个配置的胜者不同。置零到达同时改变了可见性和部分请求的 FIFO 顺序，这个数字不能单独归因于某一种因素。它说明全可见后缀结果不能代替原到达轨迹的比较。一键复算同时核对两类证据。

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_reference --split holdout --inputs docs/evidence/r02-cpu-inputs.json --out outputs/r02-holdout-reference.json
~~~

## 分开比较到达时刻与 FIFO

增加第三种输入：把到达时刻都提前到 0 ms，同时按原到达顺序排列请求，确保 FIFO 与原始输入一致。原始输入与该变体的比较反映固定 FIFO 下提前到达的影响；两个全可见变体的比较反映同时到达时调整 FIFO 的影响。设计集分别有 0、7 个配置改变胜者，留出集分别有 13、6 个。这里的归因只对应这条干预顺序；提前到达还会改变预取机会，不能解释成单独的在线信息增益。

[三种输入的逐方案结果](evidence/r02-cpu-visibility.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_visibility --check docs/evidence/r02-cpu-visibility.json
~~~

## 原到达时间下的闭环回退

保留冻结输入的到达时间和 FIFO。在原在线入口中，若剩余可见预测没有可复用后继，或快照内仍有未到达请求，直接完整求值当前可见窗口。尚未进入窗口的请求由外层执行器按原到达时间接纳。54 个配置的动作和逐请求完成时间与完整参照一致，审计分别核对了复用和回退路径。

留出 216 次预测中有 135 次回退、54 次命中。事件循环数从 1,371 降至 1,344，设计集则从 1,497 增至 1,502。这次对照只计确定性事件数，净时间结论仍以闭环计时段为准。

[原到达闭环逐配置结果](evidence/r02-cpu-guarded.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_guarded --check docs/evidence/r02-cpu-guarded.json
~~~

## 原到达闭环的净成本

复用闭环计时的八轮交错设计，保留 C 的原到达时间，以 Guarded 适配器执行适用性检查与完整求值回退。设计、留出的选点耗时配对中位数分别增加 118.10、98.30 µs。注入计时后，平均响应改善的配对中位数分别约为 −0.0001、0.0000 ms；最差单请求额外等待分别为 0.4218、0.3697 ms。逐轮数据含回退计数和近邻秩 P95 差（该四请求样本等于最大值）。两种到达设置均支持继续保留完整求值。

[原到达闭环计时](evidence/r02-cpu-guarded-timing.json)

~~~sh
python -m scripts.cpu.lookahead.staging.study_r02_closed_loop --original-arrivals --out outputs/r02-guarded-timing.json
~~~
