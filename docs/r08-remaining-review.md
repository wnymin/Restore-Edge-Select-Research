# R08 剩余分线：归档复核与解析对照

2026-11-24。下面的复算和真机补实验未使用新窗口。其后经授权，我方接替原协作者任务，另行完成[新源窗验证与结项](r08-takeover-results.md)；未更改候选、未做独立盲验，也不代签。[R08 分工及双门槛](r08-counterexample-optimization-plan.md) · [本方不替换意见](r08-entry-refinement-handoff.md)

## `R08-VAL-01`：见证和两个负对照已复算

- **关键见证。** B481／4B／12×／176 MiB 的请求 7 从四后继的 1024 改为全已到达的 512，单点闭环改善 13.251 ms；B482／27B 载荷／16×／384 MiB 的请求 16 仅扩视野仍选 1024，加入后继联动后选 512，单点改善 222.192 ms、剩 30.478 ms。B481／27B 载荷／16×／512 MiB 的请求 15、16 在原六轮全部超过 10 ms，最坏超过基线 36.253 ms。另从固定轨迹复跑已有最小反事实，结果与已交付机制证据 SHA-256 一致。
- **到达负对照 `R07-A05`。** 在同源窗、同请求数、同跨度和同成本下配对突发与均匀，原跨度三档差为零（浮点舍入内）；低队列压力的 `knee` 配对在 12× 平均离线差变化 −0.358 ms，16× 为 +0.336 ms，1× 为零。每档 12 个源窗，保留逐窗队列可见性和差值，不把突发标签当通用预测信号。
- **容量负对照 `R08-A04`。** B481／27B 架构载荷／16× 的 176→180 MiB，单独装得下的冷候选为 0→14，实际预留 0→13，两点均无可省空间。180 MiB 的 13 个预留各有以本请求发出的依赖搬运，并在模拟 dispatch 前已完成，且被所选边界要求；这仍未给均值选点带来收益。384 MiB 正对照有 14 个预留和 681.075 ms 可省空间。只统计可归属本请求的已完成预取；共享、在途继承、逻辑接管与真机首 token 前消费不能由这项回放互推。

`VAL-01` 的归档复算已完成；新窗口、分层和运行前锁定另见[本方清单](evidence/r08-takeover-seal.json)。未取得第三方封存。

## `R08-GPU-02`：逐条判定

下表是 R07 真实调度分线 `A06–A10`（不是 R07 扩展分线的同号见证）。深减浅使用同提示块和同运行模式；并非 R08 CPU 执行器，也不是策略接管。

| 见证 | 从原始归档复算／真机补对照 | 判定边界 |
| --- | --- | --- |
| 真实调度 `R07-A06` 目标与同行 | 关闭连续合批时，有同行／加后继的目标首 token 分别 −209.36／−217.84 ms，同行最大客户端间隔 +10.29／+13.27 ms，同行整段 +12.24／+21.31 ms。 | 目标获益与同行受损可同时发生；不把这些结果换算成 CPU 的重叠减速或线上发生率。 |
| `R07-A07` 阶段异号 | 打开连续合批时，恢复 RPC 交叠的最大同行间隔 +11.24／+11.31 ms，整段最大间隔却 −51.11／−58.58 ms。 | 交叠只定位时段，不能将因果归给 DMA、RPC 各内部阶段或批设置。 |
| `R07-A08` 可见性与开销 | 扩展 `/slots` 在 16/16 次探针看见已确认恢复长度；排队从 0→1→2 时活动 ID 不变。两次探针中位 189.35 ms，目标首 token 额外等待 187.05–197.51 ms。 | 看不到等待队列身份、所有合法候选、共享暂存和在途状态。探针开销不等于策略开销；不可回填事后状态。 |
| `R07-A09` 16 层流式失败 | R07 正式同跑 21/32 失败；无恢复、无同行的匹配独跑 16 层 5/8 失败、4B 0/8。新增同 16 层权重、同二进制、同提示、固定采样、清槽后单跑，只切换 `--skip-chat-parsing`：两臂均为相同的 5/8 失败，收到相同的 Content-only 500。8/8 正式块的可见 token ID 共同前缀一致。 | 排除恢复或队列是这些已观察失败的必要条件；该开关没有修复故障。不证明关闭所有内容校验后的行为，也不分离衍生权重、生成内容和服务器内部校验，更不把成功子集当可用性通过。 |
| `R07-A10` SSE 片段敏感性 | 第 3 块客户端缺一个可见片段，排除后 7 块只恢复对照 +12.00 ms、续算无探针对照 −52.84 ms，主要方向不变。 | 仍是客户端间隔，不填造缺失 token 时刻，不当硬件阶段计时。 |

归档核对与新的解析对照已交付；当前接口**未通过匹配语义的策略闭环资格**。独立验收意见不由本方代签。解析对照每臂单独启动同一固定版本服务器，无恢复和同跑；不以固定两臂先后顺序作性能比较。

[复算数据与原档 SHA-256](evidence/r08-remaining-archive-review.json) · [解析对照原始配对](evidence/r08-parse-control.json)。从仓库根目录运行：

```powershell
python -m scripts.validation.check_r07_live_scheduling
python -m scripts.validation.check_r08_remaining_archive --out outputs/r08-remaining-reproduction.json
(Get-FileHash outputs/r08-remaining-reproduction.json).Hash -eq (Get-FileHash docs/evidence/r08-remaining-archive-review.json).Hash
# 选择与 R07 归档哈希匹配的本机 llama-server 和 16 层模型文件：
$server = (Get-ChildItem data/raw -Recurse -Filter llama-server.exe | Select-Object -First 1).FullName
python -m scripts.gpu.experiments.study_r08_parse_control --server $server --model outputs/r08-16-layer-calibration.gguf --out outputs/r08-parse-reproduction.json
```

## 接替前的独立收口方案（历史）

- **继续验证：**协作者提供已封存的清单哈希、封存时间及身份；到盲验阶段提交判据、逐窗／逐请求、六轮配对与离线区间的独立产物，注明通过、不通过或证据不足并签署。候选方已知旧回归，不自行另挑新窗后称为独立。冻结哈希和旧资料见[交接](r08-entry-refinement-handoff.md)。
- **针对当前候选停止：**已知非劣门槛失败足以说明不替换。若双方选择止于负结果，记录双方确认来源和日期，明确标 `VAL-01` 封存、`VAL-02` 盲验未执行／停止；不改成通过，不声称独立泛化、真实闭环或 27B 原生验证。

上列原方案未执行；本次结项依据接替授权与新源窗结果，不计为双方签署完成。

### 可选：封存后逐窗执行器

[`run_r08_sealed_holdout.py`](../scripts/validation/run_r08_sealed_holdout.py) 是本方提供给协作者检查的执行入口，**不分配或挑选窗口**，也不替代其独立计时脚本审批、源窗统计方案或签署。若采用，先由协作者检查该脚本并把 SHA-256 写进封存记录；在未揭盲时只传清单哈希，不在此仓库保存清单正文、候选运行结果或未公开窗口。

封存清单为 UTF-8 JSON，字段如下。`auditor`、`sealed_at` 与来源声明仍需外部确认，执行器验证的是载荷与哈希，不验证签署人的身份或封存实际发生的时间。

| 字段 | 内容 |
| --- | --- |
| `schema` | `1`。 |
| `auditor`, `sealed_at`, `interval_method` | 执行者标识、有时区的封存时刻、预先固定的按源窗单侧区间方法；空值不接受。 |
| `frozen_sha256` | 从[冻结汇总](evidence/r08-entry-refinement-selftest.json)的 `provenance` 读取 `config`、`entry_refinement`、`simulator`、`evaluator` 四项。 |
| `driver_sha256` | 本执行器脚本的 SHA-256。 |
| `thresholds` | `mean_ms: 1, p95_ms: 1, keep_net: 0.99, gap_fraction: 0.05, recover_fraction: 0.5, extra_request_ms: 10, repeat_excess: 4, repeats: 6`。 |
| `oracle_nodes` | 每个配置离线分支定界的正整数节点预算；区间跨界时不得判通过。 |
| `strata` | 分层名到 `{pressure, cost_domain, source_windows}` 的映射。覆盖 `low-no-room`、`near-optimal`、`capacity-edge`、`high-pressure`；`proxy` 和 `measured` 不能混在同一层；每层至少两个独立源窗。 |
| `cases` | 每配置 `{trace, window, raw_rows, source_sha256, model, rate, capacity_bytes, stratum}`；`raw_rows` 是 `[352*window, 352*(window+1))` 的两个端点，`source_sha256` 来自仓库固定轨迹清单。实测成本限定归档预填充与搬运配对，`model` 为 `4b-native` 或 `16l-native`；代理成本采用 R07/R08 已有模型配置。 |

在独立封存已登记、双方按计划揭盲之后执行，路径使用协作者准备的清单；预先保留外部发布的哈希，不在揭盲时用同一文件重新生成“封存值”。

```powershell
python -m scripts.validation.run_r08_sealed_holdout --manifest $revealedManifest --sealed-sha256 $publishedSeal --out outputs/r08-independent-results.jsonl.gz
```

执行器检查候选与驱动版本、已有结构化归档和 R07/R08 矩阵的源窗重叠、轨迹哈希、分层窗数与双门槛参数。输出合法动作、零开销闭环结果、逐请求响应、六种配对顺序和离线区间。实际因果标签、按源窗区间与签署由独立执行者交付，不能由通过执行器检查推断通过验收。接替前仅用公开微用例验证入口；接替后用它执行了本方锁定的 120 个配置，未执行独立盲验。
