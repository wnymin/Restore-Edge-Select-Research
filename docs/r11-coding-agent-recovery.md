# R11｜代码 Agent 的工具循环与混合缓存恢复边界

较浅检查点已经驻留时，更深的检查点只有在节省的续算超过额外恢复成本时才值得取回。整状态文件的真机配对中，恢复只比浅点多覆盖 128 token 时，请求慢了 51–85 ms。R11 从 2026-11-27 起用代码 Agent 的工具循环与并发分支检验这种差异有多少机会影响选点。[真机配对](archive/gpu-studies/gpu-boundary-restore.md) · [R10 最近点对照](r10-natural-workload.md)

## 研究问题与假设

- 工具返回和回退是否反复命中短间距的检查点？用精确 token 前缀、工具边界、同／跨会话间隔和后继竞争做联合统计。对照追加式普通请求。工具事件与单纯的长空闲分别编码。
- 如果浅热深冷出现，常态 SSD 路径何时比从浅点继续计算更慢？相同请求做两点均热的续算配对和分层恢复配对。检查点需要完整前缀 KV 和对齐的循环／卷积状态。
- 真实到达与固定放置下，已到达请求形成的预取覆盖、在途共享与暂存竞争能否改变逐点选择，并使 `entry-replay` 获得净收益？全样本与符合机制的子群同时报告。将已到达队列中的改选与实际净收益分别计量。

## 分工与验收

| 任务 | 负责人 | 依赖 | 交付与验收 | 状态 |
| --- | --- | --- | --- | --- |
| `R11-DATA-01` 轨迹核验 | Codex | R10 输入协议，公开数据入口 | 固定版本、许可、字段、时间含义和哈希粒度。界定能做时序、前缀或工具事件分析的字段，固化会话级校准／留出划分 | TraceLab 全量与 Weka 24 个完整会话已核验。Weka 全量下载未取得 |
| `R11-ANA-01` 请求与工具循环特征 | Codex | DATA-01；本地百炼 Coder 可做先导对照 | 报告全请求、可恢复请求和工具返回分母，并报告跨会话复现。若源仅含前缀与时序，以代理标注 | 工具循环、完整前缀和跨分支同模型重叠已分项核查 |
| `R11-CPU-01` 原始到达下的策略比较 | Codex | DATA-01、ANA-01、R10 成本与执行器 | 固定放置与候选、保持时间尺度，逐对比较最近点、`local-cost` 和 `entry-replay`。留出报告模拟响应、改选、P95、慢化比例、决策成本与串行基线；机制消融固定到达 | 已完成共同锚点、跨分支回放和单因素对照，未观察到前瞻改选 |
| `R11-GPU-01` 闭环服务配对 | Codex | 独立留出支持 CPU-01 的机制，现有原生目录和真实设备 | 两热续算、深冷搬运和同输入策略 A/B。记录 TTFT、P95、探针与决策开销；覆盖热态和冷态 | 先拆出长上下文续算，完成 36 轮真机配对。独立留出无可复现前瞻改选，闭环 TTFT A/B 未达到启动条件 |
| `R11-INT-01` 论文主线判定 | Codex | DATA、ANA、CPU 和 GPU 分线 | 受控回放支持机制、独立代码 Agent 留出上超过 `local-cost` 的净收益、可解释的真机复现三项齐备时确立论文主线；否则记录规模或机制缺口 | 2026-11-27 未通过主线判据，保留为负载特征和方法适用边界的证据 |

先导采用 R10 实测的模型与 RAM／SSD 分段成本，保留原始到达间隔。以同输入同放置比较三个选择器，保持固定 FIFO、只见已到达请求和原有预取语义。放置或检查点只在校准集上确定，固定后进入留出。更快 GPU 或更大状态尺寸单列为有待真机标定的敏感性。跨会话 bootstrap 或留出块配对用于给出净收益区间；同一会话的连续请求不作为独立样本。

## 已取得的公开轨迹

| 来源 | 可验证内容 | 已知缺口 |
| --- | --- | --- |
| Weka Claude Code，2026-04 和 2026-06-21 | Apache-2.0，样本取自 Dataset Viewer 两个版本的 `train` 行。4 月版保留停止原因，6 月版保留子代理分组和同会话到达。样本行索引与实际文件 SHA-256 随证据记录 | 哈希作用域仅在单条轨迹内。跨轨迹同时到达和物理驻留未公开。4 月版目录版本 `ae1ab9b568b8b5896c561692c40a6703cf5f8525`，6 月版 `23f152f6f0f9399a85901b89a6458def0ef16729`。完整原文件下载超时，两批先使用已取得的完整会话行 |
| TraceLab `v0.0.1`，Claude Code／Codex | CC BY 4.0。完整发布包 SHA-256 与上游一致，含 `timing_events`、`tools`、`current_tool_result_count`、输入／输出 token 数。可按同会话相邻轮次的工具调用 ID 联结发出与返回 | 公开字段只有缓存长度和计数，没有逐轮 token 序列或块身份。可验证工具循环与成本候选的规模条件，单独无法生成精确恢复候选 |
| 百炼 Coder 归档 | 原发请求时序、会话／父会话、16-token 哈希与长度 | 来源标注为代码生成，无法识别真实工具、Agent 类型及物理放置，作为先导代理与跨域对照 |

[Weka 4 月版](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-042026) · [Weka 6 月版](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) · [TraceLab 发布包](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1) · [HiState v1 §4.1](https://arxiv.org/html/2610.41508v1#S4.SS1) · [百炼固定源与输入口径](r10-independent-input.md)

### TraceLab 实测的工具循环

| 上游轨迹 | LLM 轮次 | 输入含工具结果 | 可与上一条相邻轮次的工具 ID 联结 | 联结中上一轮输出不超过 128 token | 可用工具耗时 P50／P90／P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude Code | 140,338 | 120,760（86.0%） | 120,713 | 24,993（20.7%） | 125 ms／14.56 s／59.17 s |
| Codex | 216,823 | 195,268（90.1%） | 157,626 | 69,208（43.9%） | 237 ms／5.18 s／30.00 s |

输入含工具结果覆盖 Claude Code 的 2,268／2,676 个会话与 Codex 的 1,452／1,589 个会话。相邻轮的输出长度只衡量可能保留的短生成段。两个完整检查点能否同时命中仍需精确前缀、存点边界与放置记录。工具耗时为发布包的 `tool_wall_latency_ms`，分位数只使用非负有效值，分别覆盖 141,824 和 290,081 个工具调用。[完整字段与分母](evidence/r11-tracelab-audit.json) · [复算](../scripts/data/audit_r11_tracelab.py)

### Weka 几何和回放口径

对每个会话，按原始到达排序，展开 6 月版子代理请求并保留它们在同会话内的相对时间。每条更早且 API 已返回的请求，可提供完整 64-token 块对齐的输入末端。只匹配同模型、同会话的整段块哈希。逐请求取两个最靠后的不同深度，统计距离不超过 128 token 与另一同模型请求仍在 API 中的交集。API 重叠是同会话的时序观测，模拟器里的排队、载入或预取仍需独立报告。

首轮回放以每版的行 0–5 为先导，行 6–11 为留出。每窗用同一模型的八条历史和四条在线请求，保留到达间隔。候选从共同驻留锚点开始，最浅后续检查点也驻留，其他检查点置于 SSD，增量 KV 位于 RAM。各臂共用 82.25 MiB 预取预算及 R10 的搬运和相对后缀成本，超过 2,560-token 相对位置时取最后一档。这个代理未计入锚点以前的上下文，长上下文成本另见真机对照。

### 首轮结果（2026-11-27）

| Weka 请求几何 | 先导 | 后续核对 | 合计 |
| --- | ---: | ---: | ---: |
| 完整会话／模型请求 | 12／2,244 | 12／1,919 | 24／4,163 |
| 命中两个不同历史检查点 | 2,035 | 1,772 | 3,807 |
| 最近两点距离 ≤128 token | 84 | 64 | 148 |
| 距离 ≤128 且另一同模型请求的 API 尚未返回 | 15 | 6 | 21 |

148 个近邻候选占全部请求的 **3.56%**，与同模型重叠相交后为 **0.50%**。21 个联合位置来自 6 月版的四条轨迹，证据记录了请求位置、边界与其他 API 返回时刻。返回时刻只用于事后核查，不能给选择器当作已知计算余量。4 月版有 1,604 条请求接在一次 `tool_use` 停止之后，其中 26 条满足近邻条件，比例为 1.62%。接在其他已知停止原因之后的 142 条中有 4 条（2.82%）。这批样本未观察到工具调用后近邻候选的富集。两份发布的采样过滤不同，差异只用于选取下一阶段的核查对象。[先导逐会话](evidence/r11-weka-replays.zip) · [后续逐会话](evidence/r11-weka-replays.zip)

| 回放对照 | 先导 | 后续核对 |
| --- | ---: | ---: |
| 可回放窗口／请求 | 84／336 | 87／348 |
| 三种策略中各自发生队首等待的请求 | 2 | 2 |
| `local-cost` 相对最近点改选数 | 3 | 0 |
| `entry-replay` 相对 `local-cost` 改选数 | 0 | 0 |
| `local-cost` 相对最近点的总响应节省，未计选点 | 6.61 ms | 0 ms |
| `entry-replay` 相对 `local-cost` 的总响应节省，未计选点 | 0 ms | 0 ms |
| 模拟响应 P95，三方一致 | 3,743.90 ms | 3,074.74 ms |

先导的三次改选来自同一会话同一窗口，从 1,088 改为 1,024 token（均为相对锚点）。6.61 ms 是三次改选的合计，未在后续会话重现。684 个模拟请求只有四个排队。计入 Python chooser 实测耗时的实验另存于证据中。由于两种策略动作相同，注入后的差值来自当前实现的计时，不作为前瞻改善恢复选点的证据。171 个入选窗口中，7 个的相对后缀全在成本网格内，164 个使用外推。会话内到达维持原尺度，跨会话全局并发不可观测。两批按发布行索引选取，短的样本区间与保留锚点条件都限制了外推范围。[模拟脚本](../scripts/data/study_r11_weka.py) · [先导原始指标](evidence/r11-weka-replays.zip) · [后续核对原始指标](evidence/r11-weka-replays.zip)

### 父子代理与独立分支的回放

6 月版公开行 0–11 共 2,405 条模型请求，包含 700 条嵌套子代理调用。403 条请求到达时，另一个分支有同模型 API 在途。118 条请求命中两个相距不超过 128 token 的历史端点，其中 20 条处在上述跨分支重叠中，占全体请求的 0.83%。跨分支包含主代理与子代理，也包含不同子代理。请求前缀的精确匹配来自块哈希，API 返回时间仅用于标记离线样本和判断何时可以发布历史检查点。

每个目标窗口从最早仍在途的同模型请求开始，保持原始到达距离，并冻结当时已完成的历史目录。不同分支允许独立锚点。常驻显存按完整前缀的并集 KV 加各锚点的循环／卷积状态计费。重叠的已有 KV 依赖按常驻处理。各臂共用 8 GiB 设备上限、512 MiB 执行池、82.25 MiB 预取池、原 FIFO 和 R10 续算代理。单独改变同伴锚点取法、窗口后继的捕获范围和请求间隔，以区分驻留、截断和时序。[回放脚本](../scripts/data/study_r11_subagents.py)

主口径的 20 个目标中，6 个在窗口起点尚无两个近邻候选，5 个超过可用于常驻前缀的 HBM，7 个同伴的完整后缀超过执行池。剩余 2 个窗口来自先导的同一会话，共 4 条回放请求，只有 1 条出现队首等待。三种策略选择完全相同，P95 都为 3,600.93 ms。串行对照和两个单因素变体均未产生分歧。行 6–11 的 6 个近邻目标均未进入主回放，独立留出无法估计该子群的净收益。Dataset Viewer 在进一步取行时连接超时，原文件下载已连到文件服务但传输被取消。[逐请求与排除原因](evidence/r11-weka-replays.zip) · [同伴驻留变体](evidence/r11-weka-replays.zip) · [后继捕获变体](evidence/r11-weka-replays.zip)

移除 128-token 筛选后，403 条跨分支同模型重叠请求中，354 条命中两个不同端点。31 个窗口符合发布与容量约束，涵盖 4 个会话、65 次回放请求，出现 11 次排队，三种策略依然零改选。留出仅 1 个会话、2 个窗口、4 次回放请求，全部无排队。同一条轨迹可能进入多个目标窗口，65 是回放次数，样本仍按会话聚类。[全部间距的回放](evidence/r11-weka-replays.zip)

### 长前缀续算的真机对照

以 RTX 4060 Laptop、Qwen3.5-4B Q4_K_M 和 R10 Vulkan 运行时，保留最深点及其前 64、128 token 的完整状态。三点在同一实例内随机轮换，续算同一输入至相同终点，计时到设备同步。块哈希只能提供上下文尺寸，因此用固定种子生成 token。表中是同轮浅、深配对后的差值中位数，并校验配对内首贪心 token 一致。

| 最深前缀 | 最深点待续算 | 配对轮数 | 多算 64 token | 多算 128 token |
| ---: | ---: | ---: | ---: | ---: |
| 19,264 | 256 | 8 | 58.35 ms | 85.68 ms |
| 40,896 | 512 | 8 | 64.69 ms | 105.78 ms |
| 55,808 | 256 | 8 | 70.58 ms | 118.76 ms |
| 19,264 | 4,800 | 6 | 23.90 ms | 82.09 ms |
| 56,768 | 6,656 | 6 | 62.36 ms | 119.30 ms |

末两行的最深点与后缀匹配实际可回放的两个目标，两组的观测候选间距均为 128 token。R10 同机 49.125 MiB 状态的常规 SSD 暂存约 24.17 ms。深点还需从 RAM 搬运一至两个 2 MiB KV 块，按块计约 0.84–1.68 ms。真实 128-token 形状下，多算成本的配对范围分别为 78.15–86.32 ms 和 105.69–137.19 ms，最小值均高于上述路径。若同一形状改为 64-token 间距，19,264-token 前缀下的多算成本为 20.77–25.32 ms，与常规迁移相当。这说明边界间距、后缀长度和微批切分均应保留到比较中。该成本对照结合了分开的同机实验，GPU 配对使用合成 token 和隔离续算，未测请求服务 TTFT、真实放置或批处理效应。[GPU 配对原始读数与回执](evidence/r11-resident-pair.json) · [R10 搬运实测](r10-native-cost.md)

将这两组 GPU 中位时间逐点代回目标请求的模拟，保持其他请求和搬运模型不变，三种策略仍选同一边界。两个目标作出决定时均无已到达的后继。一个目标可见一条未完成的前驱与一笔在途搬运，另一个目标到达前已清空队列。替换目标成本后的模拟 P95 为 5,232.75 ms，三种策略相同。四条请求的总响应计入 chooser 时间后，前瞻策略比逐点成本多 13.38 ms。该差值来自本机 Python 实现的一次计时回放，样本不足以估计部署开销。[代回成本后的选点快照](evidence/r11-weka-replays.zip)

### 论文主线判定（2026-11-27）

当前证据支持将代码 Agent 作为研究负载，尚不足以把它作为 `entry-replay` 获益的论文主线。工具循环常见，能同时满足精确近邻、跨分支同模型重叠、缓存可用与显存约束的窗口少。两处可回放的实际间距下，继续计算额外 128 token 的成本也已明显高于常规搬运。`local-cost` 相对最近点仅在共同锚点先导的同一窗口三次改选，节省合计 6.61 ms，后续未复现。前瞻策略在现有回放中未改变恢复边界。现有覆盖限于公开会话内及本机混合模型的成本域，跨会话到达、生产驻留和更大模型的实际比值仍缺实测证据。

### 复算

在仓库根目录运行。Weka 来自 Dataset Viewer 的单行响应，文件名、偏移、会话 ID、请求数和 SHA-256 见归档内的 `source_manifest`。重新下载时核对这些字段。以下命令将解包和复算产物写入 `outputs/`。[六份回放证据](evidence/r11-weka-replays.zip)

```powershell
$env:PYTHONPATH = 'src'
python -m zipfile -e docs/evidence/r11-weka-replays.zip outputs/r11-archive
$r11Out = 'outputs/r11-repro'
New-Item -ItemType Directory -Force $r11Out | Out-Null
python -m scripts.data.audit_r11_tracelab data/raw/tracelab/syfi_coding_trace.jsonl.gz "$r11Out/r11-tracelab-audit.json"
$r11Pilot = @(Get-Item data/raw/weka/062126-preview.json) + @(Get-ChildItem data/raw/weka/*-row-[0-5].json)
python -m scripts.data.study_r11_weka "$r11Out/r11-weka-pilot.json" @($r11Pilot.FullName)
$r11Confirmation = @(Get-ChildItem data/raw/weka/*-row-*.json | Where-Object Name -Match '-row-(6|7|8|9|10|11)\.json$')
python -m scripts.data.study_r11_weka "$r11Out/r11-weka-confirmation.json" @($r11Confirmation.FullName)
$r11June = @(Get-Item data/raw/weka/062126-preview.json) + @(Get-ChildItem data/raw/weka/062126-row-*.json | Where-Object { [int]($_.BaseName -split '-')[-1] -le 11 } | Sort-Object { [int]($_.BaseName -split '-')[-1] })
python -m scripts.data.study_r11_subagents "$r11Out/r11-weka-subagents.json" --target-calibration docs/evidence/r11-resident-pair.json @($r11June.FullName)
python -m scripts.data.study_r11_subagents "$r11Out/r11-weka-subagents-resident-peer.json" --deepest-peer-anchor @($r11June.FullName)
python -m scripts.data.study_r11_subagents "$r11Out/r11-weka-subagents-followers.json" --followers @($r11June.FullName)
python -m scripts.data.study_r11_subagents "$r11Out/r11-weka-subagents-all-gaps.json" --any-gap --followers @($r11June.FullName)
```

GPU 配对复跑依赖 R10 保留的 Qwen 权重、Vulkan DLL 和匹配头文件。重复测量会产生新的时延和 SHA-256，建议写入独立输出目录。

```powershell
$env:PYTHONPATH = 'src'
New-Item -ItemType Directory -Force outputs/r11-repro | Out-Null
$r11Source = (Resolve-Path outputs/r10/upstream/llama.cpp-8a56aedd6).Path
$r11Runtime = (Resolve-Path outputs/r10/llama-native-vulkan-build/bin).Path
g++ -std=c++17 -O1 -fno-thread-jumps -Wall -Wextra -I "$r11Source/include" -I "$r11Source/ggml/include" scripts/gpu/experiments/r11_resident_pair_probe.cpp "$r11Runtime/libllama.dll" "$r11Runtime/ggml.dll" "$r11Runtime/ggml-base.dll" -o outputs/r11-repro/r11_resident_pair_probe.exe
$r11GpuArgs = @('outputs/r11-repro/r11_resident_pair_probe.exe', 'data/raw/prefill/qwen35-4b-q4km.gguf', $r11Runtime, 'outputs/r11-repro', 'outputs/r11-repro/resident-pair.json')
python -m scripts.gpu.experiments.run_r11_resident_pair_probe @r11GpuArgs --group short --repeats 8
python -m scripts.gpu.experiments.run_r11_resident_pair_probe @r11GpuArgs --group trajectory --repeats 6
```
