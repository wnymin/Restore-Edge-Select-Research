# R10-GPU-05 | 本机单槽流式服务实测

## 正式服务留出（2026-11-26，JST）

四个独立输入块的 24 对均留有终止记录，48 臂中 47 臂完成、1 臂启动失败。可比较的 23 对覆盖 1,824 个逐请求配对：贪心输出和结束状态一致，恢复边界也全部一致。对两臂实际出现的 3,744 个决策快照分别套用两个选择器，仍无分歧；其中 1,214 个有非零候选、437 个有多个非零候选，两策略均未在有非零候选时改走全重算。服务曲线显示实际运行路径的差异，但没有恢复边界变化可支撑选点因果归因。

| 留出负载 | 可比较对 | 可见 TTFT P95（基线 → entry，ms） | 帧 ITL P95（ms） | 请求级 TPOT P95（ms） | SLO 达标请求（含失败臂，基线 → entry） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 百炼短输出 | 6/6 | 21,677.12 → 21,665.47 | 21.89 → 21.84 | 16.74 → 16.76 | 219/576 → 219/576 |
| 百炼混合输出 | 5/6 | 38,187.90 → 38,193.22 | 16.83 → 16.96 | 16.32 → 16.60 | 109/576 → 169/576 |
| OASST 短输出 | 6/6 | 8,344.15 → 8,444.12 | 21.52 → 21.56 | 16.46 → 16.46 | 215/384 → 215/384 |
| OASST 混合输出 | 6/6 | 6,655.81 → 6,677.91 | 17.17 → 17.16 | 16.18 → 16.21 | 265/384 → 264/384 |

各组先按臂取 P95，再对可比较配对等权平均；表中的 P95 不是合并全部请求得到的分位数。按单对看，`entry-replay` 的 TTFT P95 变化从改善 341.80 ms 到退化 426.81 ms；样本内单请求最大多等 466.17 ms。实验 SLO 要求可见 TTFT ≤ 2,500 ms、请求级帧 TPOT ≤ 150 ms，且成功结束。可比较的同一批请求上，两策略都是 808/1,824 达标。各保留配对的有效吞吐变化为 −0.020 到 +0.029 请求/秒。逐档的实际提交率、完成率、token 吞吐和 SLO 有效吞吐见[服务汇总](evidence/r10-native-service-holdout-summary.json)和[曲线](evidence/r10-native-service-holdout-curves.svg)。

只有 31/47 个成功臂的实测队列符合锁定目标档位。百炼 thinking 的短、混合低负载点在留出中都落入过载，OASST 的两个短输出低负载点均不满足任何登记档位。保留原标签，并同时报告实际到达和排空；不把校准倍率当作留出的实测负载。成功配对中百炼混合峰值未完成数为 4–81、末次提交后的排空约 1.15–69.27 s；OASST 混合对应为 3–32、0.33–20.08 s。每类每档最多两块，不能据此估计生产稳态容量。

唯一失败为百炼 TraceB 混合输出拐点的 `local-cost`。原生工作进程已加载 Vulkan 后端，随后记下 `0xc000008f`、`phase=startup`，在 `!open` 前退出；没有请求提交或策略选点，未确定异常的具体触发条件。96 个未观测请求保留在正式分母中。全登记分母下，`local-cost` 达标 808/1,920，`entry-replay` 达标 867/1,920；多出的 59 个达标请求全部来自这对中唯一运行完的 `entry-replay` 臂，不能归因为选点收益。一次启动失败也不足以判断两策略可用性的差异。

[执行锁](evidence/r10-local-service-holdout-execution-lock.json) · [原始归档](evidence/r10-native-service-holdout.zip) · [交付核验](evidence/r10-native-service-holdout-delivery-audit.json) · [同快照与档位解释](evidence/r10-native-service-result-explanation.json) · [逐请求表](evidence/r10-native-service-requests.csv) · [逐请求分母](evidence/r10-native-service-requests-audit.json) · [独立归档复算回执](evidence/r10-native-service-reproduction-receipt.json)

一次恢复批次遗漏本地 Python 包导入路径，20 个启动尝试均在原生工作进程启动前退出。已核验并保留此前实际完成的 14 对，修正启动环境后按原冻结条件执行余下 10 对。该启动器事故的目录和哈希在本地运行目录单独保留，不与正式实验中的原生启动失败合计。

## 双序列共享暂存验收

初始两次尝试均为编译器内部错误，未到达 GPU。拆分探针函数后，默认两条物理 KV 流的上下文在首个恢复处记录 `n_stream mismatch`，拒绝单槽生成的一流检查点。改用一个共享 KV 流、仍保持两个活动序列后，同一冻结检查点实际完成了额度拒绝、在途租约加入、RAM 和 SSD 上传、两次设备内复制、占用槽拒绝和序列 A 隔离。此时与从头重算比较，序列 A 保持精确，序列 B 在合批前的状态已经不一致，后续状态和 logits 也不精确。此差异发生在混合提交之前，不能归因于混合提交的行顺序；本实验没有查明 B 的重算为何产生不同字节。

进一步固定检查点本身：对照用同一封存对象重组原状态，经同步接口在 A 存活时恢复 B；实验用原暂存租约和设备复制恢复 B。两种行顺序共 36 个全量 logits 位置、4 个批前检查点状态及 4 个最终状态逐字节一致，微批拆分与各自对照相同；同时核对了两次一页不足拒绝、两次在途共享、活跃槽拒绝和最后一个租约释放后额度归零。这个通过只覆盖一个共享 KV 流、一个 512-token 检查点、固定的双序列状态转移；原来要求的「与从头重算精确一致」未通过，连续合批下两策略的收益未测。

[构建失败 1](evidence/r10-native-two-sequence-staged-v1-summary.json) · [构建失败 2](evidence/r10-native-two-sequence-staged-v2-summary.json) · [双流布局拒绝](evidence/r10-native-two-sequence-staged-v3-summary.json) · [共享流与重算不一致](evidence/r10-native-two-sequence-staged-v4-shared-kv-summary.json) · [同一检查点对照](evidence/r10-native-two-sequence-staged-v5-sealed-control-summary.json) · [结项证据索引](evidence/r10-local-delivery-audit-v2.json)

## 单槽服务方法与资格

按 R10 方案先走单计算槽，同一原生检查点目录、同一 FIFO 控制器和同一搬运通道。预填充完成后继续使用该序列做贪心解码，一个实际 token 一帧，从工作进程发往客户端所在的父进程。请求在解码结束后释放计算槽；其它请求的可用搬运仍按原控制器推进。策略的估算器仍用预填充成本，不预知解码长度；这一工况与纯预填充主实验分开。

短输出和混合输出用固定上限；混合组按源请求序号固定每四条中的一条为较长上限，之后才看实际模型输出。遇到结束 token 提前停止，不把结束 token 算作可见输出。原生检查点只保存模型状态，未保存边界处的 logits，因此服务先限制每个合法候选在输入末尾前至少剩一个提示 token，并预留足够输出上下文。检查不通过就拒绝整块，不在某条策略中删掉候选。

资格先跑同一合成三请求的浅、深两臂，每臂对同请求独立从头重算。校验最终预填充状态和 logits 逐字节一致，之后比较全部贪心 token、长度及结束原因。逐事件核对原生预填充、首 token 发出、父进程收帧、最后收帧与处理完成。父进程从实际提交到第一帧只代表本地进程管道测量，HTTP 接收时钟另跑。

2026-11-25 本机资格通过：同一 82.25 MiB 暂存预算，固定前驱从头重算、后继从 512 token 恢复，目标分别用 512、1,024。两臂各 3 请求，独立的全重算参照与恢复的全部 6 份预填充状态和 logits 逐字节一致。每臂实际生成 8／4／4 个 token，逐 token 与各自参照及对侧一致，均达到长度上限；日志逐一核对了 13 个输出间的设备信号与 16 个客户端 token 帧。参考计算在计时外，但执行带完整正确性复核，只用来验收语义，不使用该轮的延迟比较策略。

[原生资格摘要](evidence/r10-native-service-pipe-qualification-summary.json) · [原始事件、回执和 stderr](evidence/r10-native-service-pipe-qualification.zip)。复跑用当前原生执行器开启 `--service-tokens 4 --service-long-tokens 8 --verify`，再用 `scripts.gpu.analysis.audit_r10_service_pipe` 复核；普通预填充主实验不会进入这个模式。

服务统计区分首 token、逐 token 间隔、请求级 TPOT、总请求吞吐、token 吞吐和 SLO 有效吞吐。第一帧缺失不计入首 token 分位数，但仍保留在 SLO 分母；失败、超时和零输出单列。服务扩展不能替代 R10-GPU-04 的完整 60 对主检验。

服务校准前已登记实验目标：可见文本 TTFT 不超过 2,500 ms，请求级客户端 TPOT 不超过 150 ms，且必须收到正常终止。单输出 token 的 TPOT 不适用，仍须有可见文本并正常结束；没有可见输出、失败或未完成一律计入分母但不达标。这两个数值是本机实验目标，尚无生产 SLO 数据。校准仅使用 OASST cal-03 与百炼 coding-28，固定 16-token 上限，或每四条请求中的一条改为 64-token 上限。每个输入、每种输出设置先测试三个到达乘数，共 12 次基线运行。[SLO、档位与选档规则](../configs/r10-service-calibration-v1.json) · [校准执行器](../scripts/gpu/experiments/run_r10_service_calibration.py)。若档位没有分别覆盖低负载、近饱和和过载，就补做校准或明确缺档，不能用预填充的饱和率替代服务饱和率。

服务留出成员和排重口径见[两类输入与留出冻结](r10-independent-input.md#正式留出)。[输入形状审计](evidence/r10-native-service-input-shapes-summary.json)核查了 2 个校准块和 4 个留出块的两种输出设置。320 条留出请求的合法候选距提示末尾至少 35 token，按混合输出上限生成后最少还剩 448 个上下文位置。

回环传输使用只监听 127.0.0.1 的临时 HTTP 端口，开放环客户端以实际 POST 送出固定目录内的请求索引，服务端按登记的请求顺序提交原 FIFO。此入口不接收任意外部提示。工作进程为每个 token 提供词表字节，网关增量解码 UTF-8 后发 SSE；首个非空文本和内部 token 帧分开计时。CPU 传输试验验证了跨帧字符、无输出结束和 SLO 分母。真机反序复测同一组三请求，浅、深两臂各 3 个都收到可见文本和终止事件，客户端各逐一核对 16 个 token 帧和 13 个间隔，输出仍与各自的独立全重算参照一致。[回环真机资格摘要](evidence/r10-native-service-sse-qualification-summary.json) · [客户端回执、设备事件和 stderr](evidence/r10-native-service-sse-qualification.zip)。

资格轮带完整状态与 logits 复核。复跑在进程管道的命令上加 `--service-sse`。MinGW 以 `-O1 -fno-forward-propagate -fno-reorder-blocks` 编译工作进程，`-O2` 两次在编译器内部优化失败。

定量服务留出采用独立的执行冻结和原生目录。每个成员固定短／混合输出和三个校准锚点，双方运行顺序在同类两块间反序；共 24 对、48 臂。百炼锚点以编码先导的服务／预填充相对倍率迁移到目的源的预填充时标，须报告服务留出的实际队长和排空，不能假定迁移后的锚点达到校准负载。异常结束会保留各客户端已收到的帧和终止状态，尚未提交、在途未完成、客户端错误分别记账；原生后端没有启动时不报持续吞吐。[服务冻结](../scripts/gpu/experiments/freeze_r10_service_holdout.py) · [执行入口](../scripts/gpu/experiments/run_r10_service_holdout.py) · [统计入口](../scripts/gpu/analysis/summarize_r10_service_holdout.py)。

原版 llama.cpp 的连续合批开关确实允许正在解码的槽与新提示共用 batch；选槽和缓存前缀走它自己的相似度与槽缓存规则。当前 R10 原生工作进程把上下文的最大活动序列数固定为一，恢复目标和选择输入也只表示一个活动队首。在当前固定版本的原版 server 实现单元中，没有发现 R10 的对象、租约和选点传递接口。[源码位置及哈希审计](evidence/r10-native-service-batch-contract-summary.json)。因此不能通过切换该开关得到同一路径的两策略配对；原版 server 的合批计时最多成为另一套系统的说明性对照。要研究合批迁移，首个资格用例应让两条序列同时存活，保留各自 attention KV、循环状态、卷积历史和后续 token；在共同预取额度下核对对象租约和设备就绪，再逐一验证两条流；只有两策略共享同一批处理规则并有实发动作证据后，才比较性能。此条与本地单槽服务曲线分开，不据源码审计声称获得合批收益。

## 同步整序列恢复对照

[双序列整序列恢复探针](../scripts/gpu/experiments/r10_two_sequence_probe.cpp)在 A 存活时同步恢复 B 的 512-token 检查点，确认 A 状态保持，再混合提交 A 的一个后续 token 与 B 的 64-token 提示续段，最后做八次双序列续算。两种行顺序各用同形状重算为参照，串行重算单列。

2026-11-26（JST），本机同步整序列探针通过。[结果](evidence/r10-native-two-sequence-v1-summary.json) · [原始设备批日志、数值对照和运行回执](evidence/r10-native-two-sequence-v1.zip)。已核对归档和日志的 SHA-256，并按[复算程序](../scripts/gpu/analysis/audit_r10_two_sequence_probe.py)从归档重算。两种行顺序各与相同混合提交的重算对照比较：两条序列合计 36/36 个位置的完整 logits 及 4/4 份最终序列状态逐字节相同。改用串行重算参照时，只有 2/36 个位置的完整 logits 完全相同，最大绝对差 0.111036；36/36 个位置的贪心 token 相同。跨批形状的差异保留为诊断，不用它判定相同调度下的恢复失败。

原生 `LLAMA_BATCH_DEBUG=1` 日志保留四次 65-token 提交的微批拆分。若 A 的解码行在前，重算和恢复均先出现一个 2-token、2 序列的微批，再计算 B 剩余的 63 行。若 B 的预填充行在前，两组均拆成 64 与 1 行的两个单序列微批。两种情况都支持各自同形状的恢复正确性，但只有前一种实际覆盖设备上的预填充与解码合批。此探针不覆盖共享暂存对象的借用、在途交接或真实服务调度，不能用于估计连续合批时策略的吞吐收益。

共享暂存另用[设备暂存探针](../scripts/gpu/experiments/r10_two_sequence_staged_probe.cpp)检验复用 OASST 的一个 512-token 检查点时的租约、复制与序列隔离。[构建与归档入口](../scripts/gpu/experiments/run_r10_two_sequence_staged_probe.py) · [共享暂存验收](#双序列共享暂存验收)

[本机探针执行入口](../scripts/gpu/experiments/run_r10_two_sequence_probe.py)要求 60 对正式汇总和原始归档已齐，并核对主实验冻结的设备、模型及运行时。`--dry-run` 只显示将使用的命令和环境。正式运行默认限时 180 秒，不监听 HTTP；退出、超时或日志校验不通过时均保留原始事件和失败回执。探针与单槽服务校准分开占用 GPU。

~~~powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m scripts.gpu.experiments.run_r10_two_sequence_probe `
  --primary-lock docs/evidence/r10-local-holdout-execution-lock.json `
  --primary-summary docs/evidence/r10-native-primary-holdout-summary.json `
  --primary-archive docs/evidence/r10-native-primary-holdout.zip `
  --probe outputs/r10/r10_two_sequence_probe.exe `
  --model data/raw/prefill/qwen35-4b-q4km.gguf `
  --runtime outputs/r10/llama-native-vulkan-build/bin `
  --out outputs/r10/two-sequence-v1 `
  --archive docs/evidence/r10-native-two-sequence-v1.zip `
  --summary docs/evidence/r10-native-two-sequence-v1-summary.json
~~~

## 校准、执行冻结与复算

服务档位的补测规则在首臂运行前登记。如果原三档缺低负载或过载，在同一校准块对时钟乘数用 3/2 或 2/3 向外试；两端都已找到、只缺拐点时，先补两端之间最大的未测间隔。每种负载与输出的组合最多加 4 档，原 12 次和失败都保留。仍缺档就明确未覆盖，不用服务留出补档。每臂超时为计划到达跨度加 25% 余量，余量至少 60 秒、整臂至少 360 秒；计算结果超过 900 秒就拒绝运行。不从留出完成时间调整参数。[补档工具](../scripts/gpu/experiments/plan_r10_service_calibration_extension.py)。

从仓库根目录执行下方第一条命令，待 GPU-04 主实验退出且本机没有其它实验使用同一张 GPU 后再启动。先实跑一臂，按回执检查实际客户端帧、请求终止及原生事件，再用同一目录提高 `--max-new-steps` 补齐预登记的 12 臂。若只核对输入与命令，追加 `--dry-run`；此模式不启动模型、不监听端口。实跑的客户端自动访问 `127.0.0.1` 的临时端口，端口随该臂结束而关闭，无须在浏览器打开。第二条命令只在计划中的校准臂全有终止回执后运行。补档时向两条命令均传入相同的各份 `--extension`，并使用新的汇总与归档文件名保留初始记录。

~~~powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$r10Out = 'outputs/r10'
$r10Evidence = 'docs/evidence'
$r10Design = 'configs/r10-service-calibration-v1.json'
python -m scripts.gpu.experiments.run_r10_service_calibration `
  --design $r10Design --root $r10Out --worker "$r10Out/r10_catalog_worker_service_sse_v1.exe" `
  --runtime "$r10Out/llama-native-vulkan-build/bin" --model data/raw/prefill/qwen35-4b-q4km.gguf `
  --observer "$r10Out/vulkan-observer-pipeline/r10_vk_trace_layer.dll" `
  --cost-profile configs/r10-pilot-cost-v1.json --output "$r10Out/service-calibration-v1" --max-new-steps 1
python -m scripts.gpu.analysis.summarize_r10_service_calibration `
  --design $r10Design --plans "$r10Out/native-plans" --runs "$r10Out/service-calibration-v1" `
  --archive "$r10Evidence/r10-native-service-calibration-v1.zip" `
  --out "$r10Evidence/r10-native-service-calibration-v1-summary.json"
~~~

首轮 12 条基线均完成，OASST 混合输出尚缺低负载档。按补档规则将 780 ms 节拍乘以 3/2，补做 1,170 ms 基线后，13 条运行找齐 12 个锚点。[首轮汇总](evidence/r10-native-service-calibration-v1-summary.json) · [首轮归档](evidence/r10-native-service-calibration-v1.zip) · [补档登记](evidence/r10-native-service-calibration-extension-01.json) · [累计汇总](evidence/r10-native-service-calibration-extended-01-summary.json) · [累计归档](evidence/r10-native-service-calibration-extended-01.zip)

[首对验收回执](evidence/r10-native-service-first-pair-audit.json)核对两臂各 64 个请求、1,792 个输出 token、逐请求 token 序列及终止原因，为后续配对的执行依据。

[冻结汇总器](../scripts/gpu/analysis/summarize_r10_service_holdout.py)区分没有非零候选时的全重算和主动选择全重算，保留原生事件、暂存峰值和逻辑搬运量。失败或输出不一致的配对不进入性能曲线，登记请求仍计入 SLO 分母。

[曲线工具](../scripts/gpu/analysis/plot_r10_service_curves.py)按输入和输出设置分面，横轴使用各臂实测提交率。线段仅连接同一来源块的相邻可比较负载点。

[结果解释器](../scripts/gpu/analysis/explain_r10_service_results.py)在保留原标签的前提下复核实测队列档位，比较两臂实发边界并在归档快照上分别复算选择。

从校准归档生成服务执行锁时，使用包含补档的累计汇总和归档。服务原生目录与预填充主留出独立。先用首对核验器验证设备事件、客户端帧、输出 token 和终止回执，再执行余下配对。

基线正式汇总和双序列资格回执都落盘且 GPU 空闲后，可用 `python -m scripts.gpu.experiments.continue_r10_service_after_base --first-pair` 顺序执行这个流程。它使用同一套已登记补档和冻结脚本；每轮保留独立的归档及汇总，重新打开时核对上轮哈希后从未完成处接续。校准或留出目录中的 `stop.request` 都会挡住后续阶段；双序列资格结果单独报告，不影响单槽实验的锚点判定。

首对关闭后，可用同一入口的 `--finish-holdout`。它先用[首对核验器](../scripts/gpu/analysis/audit_r10_service_first_pair.py)核对原始文件哈希、原生目录封条、设备事件、HTTP/SSE 帧和两臂贪心输出及终止原因的一致性，留下不含计时结果的验收回执。首对校验通过后才接续其余配对；全部关闭后调用冻结的汇总入口，再用[交付核验器](../scripts/gpu/analysis/audit_r10_service_delivery.py)逐一比对原始文件、归档成员与登记哈希，并从原始张量、描述和恢复记录重建四个原生目录的封条。核验后绘制曲线。失败、输出不同或原始证据变动时保留原件，禁止自动重跑首对。

~~~powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$r10Out = 'outputs/r10'
$r10Evidence = 'docs/evidence'
$r10Membership = "$r10Evidence/r10-local-service-holdout-layout.json"
$r10ServiceLock = "$r10Evidence/r10-local-service-holdout-execution-lock.json"
$r10CalSummary = "$r10Evidence/r10-native-service-calibration-extended-01-summary.json"
$r10CalArchive = "$r10Evidence/r10-native-service-calibration-extended-01.zip"
$r10Cost = 'configs/r10-local-holdout-cost-v1.json'
python -m scripts.gpu.experiments.freeze_r10_service_holdout `
  --membership $r10Membership --primary-lock "$r10Evidence/r10-local-holdout-execution-lock.json" `
  --primary-summary "$r10Evidence/r10-native-primary-holdout-summary.json" `
  --design configs/r10-service-calibration-v1.json --calibration-summary $r10CalSummary --calibration-archive $r10CalArchive `
  --calibration-stamp "$r10Out/service-calibration-v1/design.json" --worker "$r10Out/r10_catalog_worker_service_sse_v1.exe" `
  --cost-profile $r10Cost --plans "$r10Out/native-plans" --extra-plans "$r10Out/service-native-plans" --out $r10ServiceLock
if ($LASTEXITCODE -ne 0) { throw '服务执行锁核验失败' }
python -m scripts.gpu.experiments.run_r10_service_holdout `
  --lock $r10ServiceLock --membership $r10Membership --plans "$r10Out/native-plans" --extra-plans "$r10Out/service-native-plans" `
  --probe "$r10Out/r10_prepare_catalog_target.exe" --worker "$r10Out/r10_catalog_worker_service_sse_v1.exe" `
  --model data/raw/prefill/qwen35-4b-q4km.gguf --runtime "$r10Out/llama-native-vulkan-build/bin" `
  --observer "$r10Out/vulkan-observer-pipeline/r10_vk_trace_layer.dll" --cost-profile $r10Cost `
  --catalogues "$r10Out/service-native-catalogs-v1" --out "$r10Out/service-holdout-v1" --max-new-pairs 1
~~~

正式汇总从运行目录生成。已有正式交付物时，复算使用新的输出路径。

~~~powershell
python -m scripts.gpu.analysis.summarize_r10_service_holdout `
  --lock $r10ServiceLock --plans "$r10Out/native-plans" --extra-plans "$r10Out/service-native-plans" `
  --campaign "$r10Out/service-holdout-v1" --archive "$r10Evidence/r10-native-service-holdout.zip" `
  --out "$r10Evidence/r10-native-service-holdout-summary.json"
~~~

交付核验后可以从正式归档[导出逐请求表](../scripts/gpu/analysis/export_r10_service_requests.py)。表中每个登记请求占一行，包含状态、选点、输出帧数、可见 TTFT、请求级 TPOT、单请求内最长帧间隔、是否满足实验 SLO，不含提示与生成内容。导出会用归档回执复核冻结汇总的 TTFT、ITL、TPOT、原生预填充结束至首 token、SLO 分母和两臂可配对请求的尾部；部分流中的计时可查，但未终止的请求不算 SLO 达标，也不进入完整配对的延迟统计。

~~~powershell
python -B -m scripts.gpu.analysis.export_r10_service_requests `
  --lock $r10ServiceLock --summary "$r10Evidence/r10-native-service-holdout-summary.json" `
  --archive "$r10Evidence/r10-native-service-holdout.zip" --delivery "$r10Evidence/r10-native-service-holdout-delivery-audit.json" `
  --csv "$r10Evidence/r10-native-service-requests.csv" --out "$r10Evidence/r10-native-service-requests-audit.json"
~~~
