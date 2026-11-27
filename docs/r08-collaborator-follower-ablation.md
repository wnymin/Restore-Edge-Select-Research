# R08 后继选点消融与离线余差

复算两个已公开源窗。代码截取冻结策略作出目标决策前的执行器快照，固定已经选定的请求，只改变允许枚举的后继边界。后继必须在快照时已到达。对每个配置另外做闭环回放，分别只强制目标动作和强制目标加后继动作，其余请求继续由冻结策略处理。

| 见证 | 放开的后继（零起始索引） | 目标选择 | 只强制目标的平均改善（ms） | 同时强制后继的平均改善（ms） |
| --- | --- | ---: | ---: | ---: |
| B481，4B／12×／176 MiB，目标 7 | 无、8、15 或两者 | 512 | 13.251 | 13.251 |
| B482，27B 代理／16×／384 MiB，目标 16 | 无 | 1024 | 0 | 0 |
| 同上 | 17 | 512 | 222.192 | 222.192 |
| 同上 | 28 | 1024 | 0 | 0 |
| 同上 | 17 与 28 | 512 | 222.192 | 222.192 |

B481 扩到全部已到达请求后，目标 7 即选 512。额外放开后继不改变排名，强制该目标后达到精确离线最优。相对逐请求基线的最坏额外等待为 8.867 ms。

B482 扩到全部已到达请求仍选 1024。枚举时只需放开索引 17，就能看到目标 16 选 512 的收益。索引 28 在这一对照里没有额外贡献。闭环只强制目标就能兑现相同改善，推演时允许后继变化与执行时强制后继是两件事。

只改索引 16 后，B482 与离线最优仍差 30.478 ms。逐请求比较发现只余索引 70 的边界不同。再强制索引 70 到 512，响应均值精确匹配离线最优。索引 70 来自事后离线计划。两点序列仅作为归因证据。只改索引 16 时仍有一个相对逐请求基线超过 10 ms 的尾部受损请求，最坏为 28.056 ms。

## 复算

先按仓库说明下载并校验百炼原始数据，再从仓库根目录运行。

```powershell
python -m scripts.validation.audit_r08_follower_ablation
```

## 请求 70 当时可见的后继

先强制索引 16 选 512，再截取索引 70 的实际决策快照。此时 92 个请求全部到达，后继 72、75、78、88 均有多个合法边界。

| 允许重新比较的后继 | 索引 70 的选择 | 另一主要候选的快照预测总完成时间差（ms） |
| --- | ---: | ---: |
| 无 | 1024 | 512 多 3215.744 |
| 仅 72 | 512 | 1024 多 2804.003 |
| 仅 75、仅 78 或仅 88 | 1024 | 512 多 3215.744 |
| 四个全部 | 512 | 1024 多 2804.003 |

在快照中，按独立串行成本预测会给后继 72 选 512。放开它的动作后，联合比较给后继 72 选 1024，为当前请求 70 选 512。两请求的依赖对象没有交集，因此仅靠「共享缓存对象」触发联合比较会漏掉这一见证。当前结果说明当时已到达的信息足够用于发现这项改选。测试对象仍来自事后定位，自动筛选后继、额外决策开销与跨窗收益仍需验证。

## 自动筛选的闭环对照

在冻结策略已进入精确评分、且窗口外仍有已到达请求时，诊断规则扩大到全部已到达请求，并放开队列中第一个有多个合法边界的后继。每次仍只提交当前请求的边界。规则没有用见证编号、离线最优边界或未来到达信息。关闭诊断时，隔离的执行器副本复现冻结策略的逐请求选点、完成及搬运事件。

| 公开开发见证 | 仅扩视野平均改善（ms） | 联合最近后继平均改善（ms） | 实测决策时间注入后的改善中位数（ms） |
| --- | ---: | ---: | ---: |
| B481，4B／12×／176 MiB | 13.251 | 13.251 | 约 13.7 |
| B482，27B 代理／16×／384 MiB | 0 | 252.670 | 约 238 |

B482 自动找到后继 17 和 72，仅改变当前索引 16 和 70，零开销回放达到原均值离线最优。六轮轮换计时注入后，仍有一个请求相对逐请求基线超过 10 ms，因此该见证的尾部仍未解决。计时量随运行环境改变。

另复放已公开的 12 个 27B／16×／512 MiB 窗口。零开销时，B480 改善 30.384 ms（仅扩视野即可），B482 改善 252.566 ms（仅扩视野为零），其余 10 窗持平。公开对照未注入决策耗时，未用于独立盲验。

```powershell
python -m scripts.validation.audit_r08_visible_follower
python -m scripts.validation.audit_r08_visible_follower --public-controls
```

## 公开窗口的计时与缩短推演

对原方案在 12 个公开窗口各进行六轮轮换计时，并把每次实测决策耗时注入冻结执行器。再测缩短视野版本。缩短版最多看到首个多候选后继之后四个请求，同时保留原四步视野，仍受已到达边界约束。

| 窗口与配置 | 全视野净改善中位数（ms） | 缩短版净改善中位数（ms） | 全视野／缩短版额外决策耗时中位数（ms） |
| --- | ---: | ---: | ---: |
| B480，27B／512 MiB | +17.371 | +25.274 | 42.838／14.932 |
| B482，27B／512 MiB | +234.738 | +248.169 | 110.519／27.276 |
| A96，27B／512 MiB | −10.587 | −4.316 | 53.368／21.516 |
| A99，27B／512 MiB | −7.009 | −1.442 | 74.108／13.900 |

两档 B482 均保留零开销下的改选收益。公开 512 MiB 档的额外决策耗时由约 110.5 ms 降到 27.3 ms。其余十个公开窗口的零开销持平现象延续，但实际计时中，一些窗口仍因试算回退。缩短版同时漏掉原 B481 的 4B／176 MiB 视野见证，零开销收益由 13.251 ms 降为零，计时后中位数为 −1.382 ms。这给出缩短视野的回归边界。B482 尾部超过 10 ms 的请求依然存在。以上数据来自公开窗口和本次机器计时。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls
python -m scripts.validation.audit_r08_visible_follower --bounded-view
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --bounded-view
```

## 视野长度敏感性

扫描首个多选项后继后保留 0–16 个请求的 17 种设置。每个设置仍至少保留冻结策略的四步，并截断到当时已到达的请求。两个开发见证加 12 个公开对照，共做 238 个零开销配置。

- B481（4B，176 MiB）在后继之后至少保留 6 个请求才出现 13.251 ms 收益。保留 0–5 个请求时与冻结策略持平。
- B482（27B 代理，384／512 MiB）无需在选中的后继之后额外延伸，就保留两处改选和 252.670／252.566 ms 收益。
- B480（27B 代理，512 MiB）至少需要在后继之后保留 3 个请求，才能获得 30.384 ms 改善。其余 10 个公开对照在本次扫描中与冻结策略持平。

选 6 重新做六轮计时，B481 的净改善中位数为 13.680 ms，B482 的 384／512 MiB 两档分别为 248.065 和 246.982 ms。B482 的 512 MiB 档增加约 32.510 ms 的决策 CPU 时间，低于此前全视野的 110.519 ms。A96、A99 的公开对照仍分别出现 4.171 和 1.788 ms 的中位净退化。6 来源于公开开发样本，对它的复算仍属于开发验证。

```powershell
python -m scripts.validation.audit_r08_visible_follower --sweep-view
python -m scripts.validation.audit_r08_visible_follower --public-controls --sweep-view
python -m scripts.validation.audit_r08_visible_follower --view-extra 6
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --view-extra 6
```

## SSD 搬运触发与 864 配置公开复算

在原本进入精确评分的决策上增加一个诊断触发。当前请求和第一个已到达的多候选后继，各自必须至少有一个边界依赖尚未就绪的 SSD 检查点，才能进入扩大视野和联合试算。该条件可从决策现场读取。视野使用后继之后最多再看六个已到达请求。

在两个开发见证中，触发后保留 B481 的 13.251 ms 和 B482 的 252.670 ms 零开销收益。12 个公开计时对照也保留原零开销改选结果。A96 的试算次数由 15 次降至 3 次，计入开销的平均响应退化中位数约由 4.171 降到 1.926 ms。A99 仍有 2.277 ms 退化，因此该触发仍无法确保无收益窗口不变慢。

修正后的 864 个公开配置给出以下零开销结果。所有配置的冻结策略精确均值与边界都先核对归档，关闭触发的隔离副本还核对完成时间和搬运记录。

| 模型载荷 | 改善 | 持平 | 退化 |
| --- | ---: | ---: | ---: |
| 4B | 4 | 284 | 0 |
| 交替层代理 | 0 | 288 | 0 |
| 27B 代理 | 9 | 279 | 0 |

187 个配置至少触发一次，13 个改善集中在 B480、B481、B482 三个源窗。容量和成本组合复用源窗。这一矩阵结果只证明公开数据的零开销表现。新增开销与尾部仍按计时数据判断。

```powershell
python -m scripts.validation.audit_r08_visible_follower --view-extra 6 --ssd-gate
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --view-extra 6 --ssd-gate
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate
```

## 完整公开矩阵的六轮计时

使用六步扩展和 SSD 搬运触发，864 个配置各跑六轮，轮换 local-cost、冻结策略和试验规则的顺序，并把各自实测决策 CPU 时间注入回放。下表按每配置六个配对平均响应差的中位数计数。

| 模型载荷 | 净改善超过 1 ms | 变化在 ±1 ms 内 | 净退化超过 1 ms |
| --- | ---: | ---: | ---: |
| 4B | 4 | 263 | 21 |
| 交替层代理 | 0 | 288 | 0 |
| 27B 代理 | 9 | 258 | 21 |

42 个超过 1 ms 的退化配置在零开销下均与冻结策略持平。最大中位退化为 B480 的 27B／16×／256 MiB，约 6.499 ms。全矩阵这次计时记录了各轮超阈人数。没有配置在至少四轮出现超阈人数比冻结策略更多。要判断是否换成另一批请求超阈，仍需逐请求身份记录。

脚本随后补齐逐请求超阈索引、窗内 P95 和净收益保留率。在 12 个公开对照中已复跑这些指标，未出现新增重复超阈请求。A96 与 A99 的窗内 P95 中位退化分别为 3.427、5.406 ms，均超出计划 1 ms 门槛。完整矩阵上面表格来自较早的计时版本，新字段可用同一命令重新输出。当前证据支持保留此规则作诊断实验，已知回归尚不能通过替代门槛。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate
```

## 缩减后继候选与同执行器计时

缩减组只为后继枚举独立串行成本最优和最深合法两个边界（去重），当前请求仍枚举所有边界。原计时入口的隔离副本绕过了公开 run 入口的加速参数映射，导致试验组未使用冻结策略的相同执行器加速路径。补测加入 --match-executor，将缓存成本、提交后选点和标量回放参数与冻结策略对齐。两次全矩阵的零开销改选索引和指标逐配置相同。

| 864 配置，每个六轮 | 平均改善超过 1 ms | 平均退化超过 1 ms | P95 退化超过 1 ms | 最坏平均退化（ms） |
| --- | ---: | ---: | ---: | ---: |
| 缩减，原隔离副本 | 13 | 34 | 47 | 5.075 |
| 缩减，同执行器路径 | 13 | 11 | 23 | 2.768 |

表中采用六轮配对差的中位数。对齐后仍有 119/647 个六轮均存在正基线收益的配置，净收益保留率中位数低于 99%。最大 P95 中位退化 5.424 ms；按请求索引比较六轮，无新增至少四轮超过 10 ms 的请求。原有尾部超阈仍可能存在。由于参数从公开数据开发，且仍有已知退化，结果继续用于诊断和优化。

同执行器路径的逐配置六轮数据见 [计时归档](evidence/r08-visible-follower-sparse-matched-timed.jsonl.gz)。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor
```

## 四步选点与串行最优发生分歧时再试算

在同执行器、两个后继备选点、SSD 触发的版本上，加一条决策前可观察的条件。四步精确评分最优与独立串行成本最优不同，才扩大视野。864 配置的零开销联合试算从 494 次降为 366 次，逐配置改选及响应指标完全一致，仍有 13 个配置改善。

计时各用六轮配对。增加条件后，平均响应中位退化超过 1 ms 的配置为 13 个，P95 中位退化超过 1 ms 的为 28 个。上一轮分别为 11 和 23 个。最大平均退化 3.260 ms，最大 P95 退化 6.491 ms。两轮计时分批运行，未在同一轮内将两种触发规则直接配对，计时差不能单独归因于新条件。减少试算次数和完整保留零开销行为已验证。计时收益仍缺直接配对证据，且当前版本仍未达到替代门槛。两轮都未观察到新增重复超阈请求。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement
```

## 两种触发的同轮配对计时

每个公开配置运行六轮，逐轮交叉计时逐请求基线、冻结策略、仅 SSD 触发、SSD 加四步分歧触发。每对策略在六轮中各有三轮先执行。对 864 个配置分别取配对差的中位数。

在先前零开销对照中试算次数确实下降的 100 个配置里，CPU 决策耗时节省的配置中位数为 1.627 ms。平均响应与 P95 改善的配置中位数均接近零。全矩阵中，相对仅 SSD 触发，增加分歧条件使 3 个配置平均响应改善超过 1 ms，2 个退化超过 1 ms。P95 改善和退化超过 1 ms 的各 4 个。

| 相对冻结策略 | 平均响应退化超过 1 ms | P95 退化超过 1 ms |
| --- | ---: | ---: |
| 仅 SSD 触发 | 14 | 25 |
| SSD 加四步分歧 | 10 | 26 |

这组配对验证了部分配置的 CPU 时间节省。请求平均响应和 P95 并未同步出现稳定改善，两种触发仍有超门槛的公开配置。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --pair-triggers
```

## 复用四步推演结果

同一决策快照下，只有可见请求范围、当前候选和全部后继边界完全一致，才复用四步评分中已有的精确值。无主机缓存的实验也复用已有的独立串行选点。每个公开配置关闭和开启复用各执行一次，自动校验触发日志、选点、完成时间、预留和搬运事件逐项相同。

保留后继后的六个请求时，864 个公开配置共 6479 次回放，复用前后相同。扩展后的视野长度与四步不同，四步精确分数无法直接复用。12 个公开计时对照中，CPU 节省的各窗中位数介于 −1.969 和 +0.843 ms，未观察到稳定改善。

后继之后保留零个请求的开发见证中，复用将 B482 回放从 79 次降至 69 次，继续获得 252.670 ms 零开销收益。B481 回放从 17 次降至 15 次，但这个短视野原本就错过 B481 的改选机会。由此可区分同视野重复评分与真正扩视野增加的工作量。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-scores
python -m scripts.validation.audit_r08_visible_follower --view-extra 0 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-scores
```

## 后继分支共用执行状态

对同一个当前恢复点，后继候选改变之前存在一段共有执行过程。固定后继最早进入预取窗口的队首位置，在这个位置的完成事件之后暂停推演。保存当时的预留、就绪对象、搬运状态及重叠速度，再分别续跑后继的各个备选边界。每个分支使用独立的容器副本。该实验在独立诊断副本中运行，仍使用冻结执行器的预取窗口。

864 个公开配置逐项比对了每次决策的精确评分、选点、完成时间、预留和搬运。复用前后均相同。回放处理的事件从 91,549 降至 88,704，减少 3.1%，涉及 88 个配置。12 个公开对照的六轮直接计时中，CPU 节省中位数按窗介于 −0.603 和 +0.724 ms，平均响应改善中位数介于 −0.035 和 +0.225 ms。减少了回放事件，端到端改善仍较小。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --pair-triggers
```

## 首次回放时保存共有状态

第一个后继分支照常完整执行，在后继进入预取窗口之前保存快照。其余分支从快照续跑。这省掉了专门跑到快照位置的一次调用。864 个公开配置均校验候选精确评分、选点、完成时间、预留和搬运不变。

| 回放方式 | 回放调用次数 | 处理事件数 |
| --- | ---: | ---: |
| 各分支完整回放 | 6479 | 91549 |
| 先单独生成快照再续跑 | 6989 | 88704 |
| 首个分支回放时保存 | 6479 | 88194 |

首次回放保存相对完整回放减少 3.7% 的处理事件。12 个公开配置各六轮，直接对比两种快照生成方式，按窗的 CPU 节省中位数介于 −0.335 和 +1.006 ms，平均响应改善中位数介于约 0 和 +0.211 ms，P95 改善中位数介于约 0 和 +0.456 ms。处理事件下降已验证，计时收益仍然较小。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --pair-triggers
```

## 完整计时矩阵按离线证书分层

首个分支保存快照版再次完成 864 个公开配置的六轮计时。每配置取六轮配对差的中位数。按原归档的离线精确上下界分层，答复哪些退化发生在原来已证明选点足够好的配置。

| 离线证书 | 配置数 | 平均响应改善超过 1 ms | 平均响应退化超过 1 ms | 窗内 P95 退化超过 1 ms |
| --- | ---: | ---: | ---: | ---: |
| 逐请求基线已最优 | 504 | 0 | 0 | 0 |
| 冻结策略已最优，逐请求基线有空间 | 263 | 0 | 1 | 4 |
| 冻结策略已证实有剩余差距 | 15 | 12 | 0 | 0 |
| 冻结策略剩余差距尚不确定 | 82 | 1 | 0 | 0 |

最坏例为 B480 的 27B／16×／256 MiB。原冻结策略已达到归档离线最优，增加试算后平均响应中位退化 1.656 ms，P95 中位退化 3.245 ms。另有 B480 的 4B、A96 和 A99 的三个 P95 超标配置。没有新增至少四轮超过 10 ms 的请求索引。这里的重复尾部比较相对于原冻结策略，原有的反复超阈仍需单独解决。当前测试继续使用公开窗口，不能作为新留出的验收结论。

逐配置记录见 [计时归档](evidence/r08-visible-follower-inline-timed.jsonl.gz)。

```powershell
python -m scripts.validation.summarize_r08_visible_timing docs/evidence/r08-visible-follower-inline-timed.jsonl.gz
```

## 用四步评分差筛掉扩展试算

用四步回放中串行最优边界与四步最优边界的预测总完成时间差，除以该视野尚未完成的请求数。此值大时，四步已经给出了明确的偏好。开发对照只在这个逐请求平均差不超过 105 ms 时扩视野。105 ms 由公开矩阵中已知改变结果的决策选出，属于事后调参。

864 个公开配置，触发数由 366 降到 70。加阈值前后，逐配置的零开销改选索引和结果一致，保留 13 个改善。新一轮六次计时后，13 个平均响应改善超过 1 ms，没有配置的平均响应退化超过 1 ms。仍有 3 个配置 P95 退化超过 1 ms，最差为 B480 的 27B／16×／256 MiB，退化 1.322 ms。两次全矩阵计时分批进行。阈值仍需要新窗口验证，且当前已知的 P95 门槛尚未满足。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105
```

## 固定 105 ms 阈值的额外公开对照

阈值来自此前 864 配置，原样用在不重叠的已公开设计窗 B440、B441 和原生校准成本回放。后者从归档重建 4B 与 16 层截取模型的实测拟合成本，包含 168 个计算和搬运均用实测拟合的配置及 36 个成本来源对照。所有基线归档选点和冻结策略精确均值先校对。两种阈值设置逐配置在同一组六轮中交叉计时。

| 公开对照 | 配置数 | 扩展次数，无阈值→105 ms | 零开销发生改选 | 有阈值时平均退化 >1 ms | 有阈值时 P95 退化 >1 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 设计窗 | 144 | 5→0 | 0 | 0 | 0 |
| 实测成本与成本来源对照 | 204 | 103→67 | 0 | 4 | 22 |

实测成本分线无阈值时对应 3 个均值退化和 21 个 P95 退化。保留阈值时最大平均退化 1.283 ms，最大 P95 退化 3.475 ms，无新增重复超阈请求。此处两种规则均未改变零开销边界，P95 的退化来自计入决策耗时后的回放。实测成本用于模拟输入，结果没有做真实服务 A/B。新增公开配置里没有出现可改选见证，无法据此估计阈值对新改选机会的漏判率。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-design --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --pair-triggers
python -m scripts.validation.audit_r08_visible_follower --public-native --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --pair-triggers
```

## 用预取窗口约束后继试算

在 105 ms 评分差条件上，再要求首个多候选后继与当前请求的距离不超过执行器预取窗口，并且四步最优候选在当前状态能完成预留。两项判断均使用当时可见状态。本轮规则受公开见证启发。

864 个公开主体配置的触发从 70 次降为 37 次，零开销逐配置改选和指标相同，保留 13 个改善。在 204 个公开实测成本及成本来源对照中，触发由 67 次降为 13 次，零开销仍无改选。六轮直接配对计时中，未增加上述条件时有 1 个平均响应、14 个 P95 中位退化超过 1 ms。加入条件后这两项均为零，最坏 P95 中位退化为 0.901 ms。这一批测试的主体配置只核对了零开销，计时结果只覆盖 204 个实测成本分线配置，且这些都是公开窗口。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower
python -m scripts.validation.audit_r08_visible_follower --public-native --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --pair-triggers
```

## 预取窗口条件的完整主体计时

864 个主体配置各做六轮四臂顺序交叉计时。第四臂关闭预取窗口条件，仍保留 105 ms 阈值，其余相同。零开销改善 13 个，计入耗时后 13 个配置平均改善超过 1 ms。504 个逐请求基线已最优配置没有绝对均值或 P95 超标。

| 相对冻结策略的中位退化 | 不加预取窗口条件 | 加预取窗口条件 |
| --- | ---: | ---: |
| 平均响应超过 1 ms | 2 | 2 |
| 窗内 P95 超过 1 ms | 4 | 2 |

加条件后，超标均出现在已认证冻结策略达到离线最优的配置。B482 的 27B／12×／384 MiB 同时给出最大平均退化 1.284 ms 和最大 P95 退化 1.456 ms。未出现新增的重复超阈请求索引。因绝对退化仍有超标，主体公开对照还不能支持替换。原生分线与主体分线应分别看。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --pair-triggers > r08-near-timed-matrix.jsonl
python -m scripts.validation.summarize_r08_visible_timing r08-near-timed-matrix.jsonl
```

## 当前请求只比较两个候选

扩展试算中的当前请求只枚举四步精确评分最优和独立串行成本最优两个候选。后继仍枚举此前的两个候选。864 个主体配置的零开销改选及响应与原版一致。回放调用从 5012 次降为 4965 次，事件从 56960 降为 56165 次。

12 个公开窗口的六轮直接对照中，按窗中位 CPU 节省介于 −0.089 和 +0.921 ms，平均响应改善介于约 0 和 +0.044 ms。另对上一次主体实验的两个超标配置各重新配对六轮，两个当前候选版均落在 1 ms 绝对退化内。27B／12×／384 MiB 的 P95 中位退化为 0.353 ms，但该配置直接配对的决策 CPU 节省中位数为 −0.0195 ms，无法把两次测量之间的变化归因于减少候选。这是一组公开回归重测，完整候选收缩计时矩阵仍需单独核验。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --two-current
python -m scripts.validation.audit_r08_visible_follower --public-controls --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --two-current --pair-triggers
```

## 两个当前候选的完整计时

864 个主体配置各六轮，四臂同轮交叉计时。第四臂保留全部当前候选，其余条件一致。两个候选版仍有 13 个配置平均响应改善超过 1 ms。相对冻结策略，平均退化超过 1 ms 的配置由全候选臂的 7 个降为 5 个，P95 退化超过 1 ms 的由 11 个降为 7 个。直接比较两臂，所有配置的平均响应差中位数都在 ±1 ms 内。

两个候选版的超标全部在冻结策略已离线最优的配置中。最坏位于 B482／27B 代理／12×／512 MiB，平均及 P95 分别退化 1.255 和 1.980 ms。未发现新增重复超阈请求。此前小范围复测均在门槛内，完整矩阵仍检出退化，故不能用小范围结果判断整体通过。

逐配置计时见 [全矩阵配对归档](evidence/r08-visible-follower-two-current-timed.jsonl.gz)。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --two-current --pair-triggers
python -m scripts.validation.summarize_r08_visible_timing docs/evidence/r08-visible-follower-two-current-timed.jsonl.gz
```

## 按 SSD 搬运与续算成本的比例触发

B482／27B 在 12× 和 16× 两档，目标 16 的四步候选均为 1024 token。未就绪 SSD 成本分别约为 548、730 ms，独立续算均约 440 ms。公开开发对照据此要求 SSD 成本至少达到该候选续算时间的 150%，再开扩展试算。

864 个主体配置各做六轮直接配对。试算数 37→24，13 个零开销改善和逐配置选点保持一致，两档已有差异因此被区分。相对冻结策略，旧版在已认证冻结最优的 263 个配置中有 2 个均值、5 个 P95 超 1 ms，新版两项均为零。新版仍有一个 B481／4B／12×／256 MiB 的均值和 P95 超标，分别约 1.019、1.190 ms。该配置属于冻结策略有离线差距的分层。无新增重复尾部超阈。150% 依公开见证设定，这次证据仍属于开发回放。

```powershell
python -m scripts.validation.audit_r08_visible_follower --public-matrix --time-controls --view-extra 6 --ssd-gate --sparse-follower --match-executor --require-disagreement --reuse-prefix --capture-inline --short-margin-max 105 --near-follower --two-current --ssd-compute-min-pct 150 --pair-triggers
```
