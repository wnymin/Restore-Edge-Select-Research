# R10-GPU-02 | 本机原生路径校准与先导计时

2026-11-25 起在 RTX 4060 Laptop 8 GB、Qwen3.5-4B Q4_K_M 和同源 llama.cpp `8a56aedd6`／Vulkan 上校准。下文保留秒级停顿的定位、统一预热后的资源与噪声测量，以及正式主留出之外的五对补测。冻结条件见[实验方案](r10-real-execution-experiment-plan.md#5-执行冻结与变更记录)。

预填充使用与在线执行器相同的 3,072 token 上下文和 256 token 切分。训练形状为已缓存位置 0／512／1,024／1,536／2,048／2,560 分别续算 1／16／64／128／256 token。另设 6 个不参与拟合的形状。前缀准备和载入放在计时外，记录提交前至模型同步完成的时间；预热一轮，12 轮随机排列每轮全部 36 个形状，共 432 次测量。模型按位置与单块长度插值，将实际 256-token 切分上的成本相加。6 个预留形状上最大绝对误差为 9.60 ms，最大相对误差 12.0%，15 个形状对的中位时长排序无反转。排序核对的是单块形状，不是策略的候选收益；最坏 9.60 ms 已大于方案约定的 5 ms 建议效果线，不能作为忽略预测误差的理由。测量时温度起止快照为 70°C 和 79°C，不是温控实验。

单对象搬运沿用可变目录执行器的搬运代码。RAM 8／16／32 MiB 源从已验收的 KV 对象摘取或拼接；SSD 使用真实 52,690,944 字节的循环与卷积检查点。测量开始前准备文件、常驻 RAM、锁页缓冲与显存暂存池。每种大小预热后 12 次；测量各阶段及设备内拷贝后，再读回设备结果确认字节完全一致。12 次 SSD 读均使用 Windows 无缓冲异步文件接口，进程 I/O 计数每次增加 52,690,944 字节，12 次合计 632,291,328 字节。该计数核实了进程实际读取的数据量，不报告 SSD 控制器内缓存或 NAND 访问。

| 路径与大小 | 发起至设备就绪中位数（ms） | 磁盘至主机观察中位数（ms） | 主机准备和提交、不含磁盘（ms） | 设备上传信号（ms） | 设备内 16 段拷贝（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAM 8 MiB | 2.21 | — | 1.35 | 0.84 | 0.31 |
| RAM 16 MiB | 3.98 | — | 2.25 | 1.71 | 0.40 |
| RAM 32 MiB | 7.68 | — | 4.60 | 3.00 | 0.52 |
| SSD，50.25 MiB | 24.72 | 13.78 | 6.41 | 4.80 | 0.87 |

表内各阶段分别取中位数，未要求其和等于总时间的中位数。SSD 发起到就绪的范围为 22.77–31.75 ms。最后一列使用独立显存目标的 16 个连续分段，不能当作在线循环状态写前保留、碎片页散拷或固定工作区峰值的精确替代。早期 Vulkan 探针的重叠证据见[异步执行资格](r10-stage-pool.md#r10-async-qualification)，不与本批数据混合拟合。

[预填充摘要](evidence/r10-prefill-grid-summary.json) · [预填充原始记录](evidence/r10-prefill-grid.zip) · [搬运摘要](evidence/r10-transfer-grid-summary.json) · [搬运原始记录](evidence/r10-transfer-grid.zip)。从仓库根目录用 `scripts.gpu.experiments.run_r10_prefill_grid` 与 `scripts.gpu.experiments.run_r10_transfer_grid` 复跑，两者均接收 `--help`；编译采用[同源运行时](r10-stage-pool.md#r10-direct-rerun)的头文件和库。两批归档保留种子、设备、运行库哈希、退出码及起止 GPU 快照。

## 计算与搬运同行

使用真实的计算完成信号和搬运就绪信号。分别对 4 种预填充形状与 2 种搬运路径，随机执行计算两遍、单独搬运、计算与搬运同时进行；每格 12 组，共 384 条记录。96 条同时进行的记录中，在真实上传提交后及上传就绪后均查到计算尚未完成。该观测证明同一窗口内两种工作未串行等待，但不能据此计算两种设备引擎同时活动的精确时长。同行与两条独占计算臂的 logits 逐值一致。

先用 512 位置的两种后缀为 RAM 和 SSD 各拟合一个固定时间修正，再用 768 位置的两种后缀独立核对。相对考虑实际搬运提交延迟的理想重叠模型，修正后各验证格的绝对误差中位数仍为 5.83–9.04 ms，单次最大 63.19 ms；6 个形状对的排序错误为 1，修正前后相同。计算单跑的同条件重复差本身很大，负残差不能当作搬运带来的加速。冻结成本未加入固定修正。本组未做同状态候选消融或闭环遗漏损失验证。首轮因预热期限超时退出，未产生正式样本，保留原始记录。

[同行摘要](evidence/r10-native-joint-grid-summary.json) · [同行原始记录](evidence/r10-native-joint-grid.zip) · [首轮预热失败记录](evidence/r10-native-joint-warmup-v1.zip)。

## 首个同策略完整块及计时诊断

在 OASST 校准块 `oasst-cal-03` 上，用同一原生目录、82.25 MiB 暂存、同一预热和先导成本，两次运行 `local-cost` 各 64 个请求。两次均通过原生事件与容量台账核对，64 个边界完全相同。以实际客户端到达到预填充完成计，均值分别为 28,030.35 和 27,971.12 ms，相差 59.23 ms；P95 分别为 29,862.20 和 29,889.26 ms，相差 27.07 ms。这只有一个源块，不能估计跨块 A/A 的第 90 百分位，也不用于确定主负载或策略收益阈值。

两遍的第 2、3 个请求从第一个预填充块提交完成至整个请求完成，耗时均在 12.60–13.50 s；两请求均选择全重算，输入分别为 520 和 608 token。第 4 个请求同区间约 1.59 s。其余请求也因积压变慢：开放环客户端相对计划最大迟到 16.38 ms，而到达后至控制器观察的 P95 分别为 6.26 和 5.89 s。控制端和原生端的完成时间序列去常数偏移后，差值跨度在每遍内均小于 0.30 ms，因此仅挪动起止时钟无法解释长等待。此区间包含后续块的提交、设备信号和轮询，现有日志不足以认定 GPU 连续计算了十几秒。已加入仅用于诊断的 `--chunk-events`，分别记录块提交开始、标记返回和首次就绪观测；先用同一校准块复现并定位，再变更预热或负载档，保留改动前后的整块以免事后挑样。

两遍共记录 34 次实际张量散拷：16 MiB 各 9 次，中位数 0.553／0.501 ms；50.25 MiB 各 8 次，中位数 0.629／0.679 ms。区间从真实设备拷贝调用至返回，含主机端提交和等待。没有出现共享循环状态写前保留或旧活动 KV 保留，不能从散拷推出这两条路径的成本。两遍运行进程的工作集峰值约 3.40 GB；末尾私有提交约 4.04 GB，Vulkan 最低可用预算快照约 4.27 GB。后两项是快照，且不含系统页缓存峰值；暂存计费峰值 82.25 MiB 也不能替代显存峰值。还需在三档分开量预算及活跃执行的计数，避免将监测开销混入 A/A。

[A/A 摘要](evidence/r10-native-pilot-aa-summary.json) · [A/A 原始记录](evidence/r10-native-pilot-aa.zip) · [长等待与实际散拷审计](evidence/r10-native-pilot-timing-audit.json)。记录由 `scripts.gpu.analysis.summarize_r10_pilot_aa` 和 `scripts.gpu.analysis.audit_r10_pilot_timing` 复算。当前 `configs/r10-pilot-cost-v1.json` 只用于校准，不是主留出的冻结配置。

## 秒级停顿的定位

用同一 `oasst-cal-03` 块、相同对象和原有 256-token 预热运行分段诊断。第一轮逐块诊断核对 64 个请求、127 个预填充块，同样的长等待复现。细分后，前三处长等待落在主机开始调用到返回标记的区间（6.48、6.39、11.84 s）；对应标记返回到设备就绪观测的区间仅 48–182 ms。

第二轮在原有时间线层上增设可选的 `vkCreateComputePipelines` 调用记录，将每次调用的开始和结束按同一进程的 QPC 时钟与预填充块对齐。4 个提交时间超过 1 s 的块，`llama_decode` 主机调用占每个区间几乎全部；将可能重叠的管线调用裁切并取时间并集后，分别覆盖 6,230.03／6,246.73、6,748.84／6,768.61、12,223.05／12,239.67、1,467.76／1,485.22 ms，即各区间超过 98.8%。这说明该校准运行的秒级停顿主要发生在 Vulkan 创建计算管线时；后续仍要核验同一预热能否覆盖另一类校准输入。新增探针只在诊断臂开启，不把 API 调用耗时当作 GPU 运算耗时。

[逐块摘要](evidence/r10-native-pilot-chunks-summary.json) · [逐块原始记录](evidence/r10-native-pilot-chunks.zip) · [管线诊断摘要](evidence/r10-native-pilot-pipelines-summary.json) · [管线诊断原始记录](evidence/r10-native-pilot-pipelines.zip)。相对 DLL 路径那次没有打开 Vulkan 设备，未进入计时；换用既有的绝对路径清单格式后才开始诊断，[失败记录](evidence/r10-native-pipeline-relative-layer-failure.zip)单列。

采用固定的 18 个合成输入长度预热（1–2,560 token，按 256 token 切分；长度在校准阶段确定），每个进程在开放环计时前执行，并清除活动序列。沿用同一个管线探针和真实目录：

| 校准输入 | 请求／预填充块 | 形状预热 | 计时段创建管线 | 最长主机提交 |
| --- | ---: | ---: | ---: | ---: |
| OASST `cal-03`，固定形状 | 64／127 | 61.45 s | 0 | 18.95 ms |
| 百炼 coding 重建块 `coder-28`，相同固定形状 | 96／574 | 57.94 s | 0 | 21.39 ms |

两块共 701 次提交，均核对了选点、设备事件和额度。常驻实例的预热开销单列；冷启动仍然有这部分延迟。不同运行使用同源后端和固定选点，以上是定位与跨校准块覆盖检查，不能用来估计算法收益或断言留出输入不会触发新的编译。下一步使用统一预热，分开跑计数和 A/A，正式留出继续封存。

负载初筛：上述 OASST 先导的实际客户端提交跨度 12.61 s、最后请求完成于约 16.55 s，高峰有 17 个已到达未完成请求；百炼先导的提交跨度 15.47 s、最后请求完成于约 85.56 s，高峰有 79 个已到达未完成请求。两者从实际到达至控制器观察的平均延迟约 2 ms。因而百炼当前档位的积压仍存在，但不该归给已消失的计时内管线创建；不能把这两个先导到达比例直接冻结为主实验争用档。先仅用基线扫描低、近饱和和过载锚点，以实际排空和队列判定。

[OASST 预热后摘要](evidence/r10-native-warmup-grid-oasst-summary.json) · [OASST 原始记录](evidence/r10-native-warmup-grid-oasst.zip) · [百炼预热后摘要](evidence/r10-native-warmup-grid-bailian-summary.json) · [百炼原始记录](evidence/r10-native-warmup-grid-bailian.zip)。逐块与管线原始记录可用 `scripts.gpu.analysis.audit_r10_pilot_chunks` 核对。

## 三档暂存的驻留开销

用同一百炼 coding 校准目录（28 个常驻 RAM KV 对象，共 448 MiB；主机锁页区实际分配 52,690,976 字节）、完整模型和活动上下文，分进程初始化三档暂存池。执行一次公共短预热后保持分配，等待计数采集。本试验不开放客户端，不作为吞吐或完整活动峰值。每档取得 17 组完整 WDDM 和系统内存计数。

| 暂存 | 对象覆盖 | WDDM 独占显存采样最大值（GB） | 进程工作集峰值（GB） | 静置时私有提交（GB） |
| ---: | --- | ---: | ---: | ---: |
| 64 MiB | 放不下一套 512-token 冷候选（66.25 MiB） | 3.254 | 3.707 | 4.331 |
| 82.25 MiB | 恰能放 1,024-token 冷候选；两套独立浅候选放不下 | 3.273 | 3.708 | 4.343 |
| 144 MiB | 两套独立浅候选可同时放入（132.50 MiB） | 3.338 | 3.708 | 4.415 |

表中的 GB 为十进制。系统缓存计数包含其它进程，三档采样总范围约 0.587–0.614 GB，不能全部归因于检查点。真正的活动显存与页缓存峰值需另跑忙时采样；这里确认的是三档原生分配和驻留尺度。早期监测脚本在读到无实例名或进程退出引发的无效样本时提前结束，留档后修正，正式驻留表只使用三条监测退出码为 0 的记录。指标由 `scripts.gpu.analysis.audit_r10_pilot_resources` 复算。

[三档驻留摘要](evidence/r10-native-resource-allocations-summary.json) · [计数、资源快照与回执](evidence/r10-native-resource-allocations.zip) · [监测失败记录](evidence/r10-native-resource-allocations-monitor-failures.zip)。

在同一 coding 校准目录上以 82.25 MiB 运行完整 96 请求、形状预热和资源采样。通过事件核对的 147 次上传中有 39 次实际 SSD 读，暂存计费峰值为 82.25 MiB。进程从预热到清空共采得 87 组 WDDM 和主存计数；独占显存采样最大 3.278 GB，共享显存采样最大 0.085 GB，进程工作集峰值 3.726 GB，排空时私有提交 4.377 GB，同一期间系统可用主存最低采样 16.355 GB。系统缓存计数包含其他进程；观测到的离散最大值也不等于连续峰值。本轮未记录客户端计时窗口的绝对起止，故表述为从预热到请求排空的资源观测；后续监测单列开放环窗口。资源监测轮不纳入 A/A。该轮到达用源时间 ×6，峰值 9 个已到达请求未完成，末次提交后仍需 6.72 s 排空。正式到达倍率依据下节的无额外监测 A/A。

[coding 忙时资源与事件摘要](evidence/r10-native-coder-busy-resources-summary.json) · [计数与原始记录](evidence/r10-native-coder-busy-resources.zip) · [该轮负载观测](evidence/r10-native-coder-busy-load.json)。

补做 OASST 同块三档忙时扫描。每档固定 260 ms/节拍和 64 请求，从第一批开放环到达到排空单独截取 WDDM 与系统计数，每档有 9 组，均完成 8 次实际 SSD 读。三档依次为 64、82.25、144 MiB：独占显存采样最大 3.26、3.28、3.34 GB，系统可用内存采样最小约 15.89、15.80、15.87 GB。峰值未完成请求均为 2；未完成请求的时间占比为 0.661、0.698、0.668。这是同一有限负载在各预算的实测资源量级，离散峰值只给下界，也不据此比较策略或把监测轮计入 A/A。[三档忙时摘要](evidence/r10-native-active-capacity-sweep-summary.json) · [窗口与负载核对](evidence/r10-native-active-capacity-sweep-load.json) · [原始计数与回执](evidence/r10-native-active-capacity-sweep.zip)。

## 统一预热后的 A/A

相同成本、82.25 MiB、开放环、固定形状预热、基础事件记录。没有启用逐块管线或额外资源监测。先在两个源块上各做一次完整 A/A；两对均保持实际选点逐请求一致。

| 校准块与到达比例 | 每次请求数 | 两次平均响应（ms） | 块均值差的绝对值（ms） | P95 差的绝对值（ms） |
| --- | ---: | ---: | ---: | ---: |
| OASST `cal-03`，260 ms/源节拍 | 64 | 401.09／360.15 | 40.94 | 75.26 |
| 百炼 `TraceA-9`，源时间 ×2.5 | 96 | 5,980.24／5,733.62 | 246.62 | 623.69 |

两对用于发现噪声量级，不能用来冻结跨块第 90 百分位。TraceA 首遍最后一次提交后仍需 8.43 s 排空，有 14 个已到达请求未完成；这个负载的均值差不可外推给较空闲档位。最终门槛使用下表的同档 A/A。

仅用基线扫描时，OASST 从 180 改为 260 ms/节拍后，最后提交后排空从 5.79 s 降为 0.41–0.52 s，峰值未完成请求从 24 降至 4。TraceA 从 ×2.5 改为 ×4 后，排空从 7.72–8.43 s 降为 0.56 s，峰值从 24–25 降至 8，提交结束时从 14 个未完成降至 1 个。较低到达率很大程度消除了这两条校准流的末尾积压；其他百炼源仍需按各自校准块确定倍率，不把 TraceA 的倍率直接外推到全部源。[本轮负载审计](evidence/r10-native-warmed-load-v1.json) · [两条额外负载原始记录](evidence/r10-native-warmed-load-controls.zip)。

另在上述四次运行的 320 个实际快照上，遮蔽未到达请求时间，同时重算 `local-cost` 和 `entry-replay`。第一种策略复现了实发边界，两种策略未出现不同边界；其中 88 个快照有非零备选。再对早期两个预热诊断块复算 160 个快照，仍零分歧（47 个快照有非零备选）。这说明目前已观察的状态上缺少可归因于选点改变的机会；该对照不反映择点开销和策略改变状态后的闭环结果，也不替代留出。百炼与 OASST 的重复快照不算新的独立源块。

[OASST A/A](evidence/r10-native-warmed-aa-oasst03-summary.json) · [TraceA A/A](evidence/r10-native-warmed-aa-traceA09-summary.json) · [A/A 快照反事实](evidence/r10-native-warmed-aa-policy-crosscheck.json) · [早期预热快照复算](evidence/r10-native-warmup-policy-crosscheck.json) · [早期负载审计](evidence/r10-native-warmup-load-initial.json)。

冻结时每源仅采用通过争用与排空检查的同档 A/A。下表为同策略两臂的块均值绝对差；每源只有一个校准块。

| 校准源 | 冻结到达尺度 | 块均值绝对差（ms） | 证据 |
| --- | --- | ---: | --- |
| OASST | 260 ms／节拍 | 40.94 | [同档摘要](evidence/r10-native-warmed-aa-oasst03-summary.json) |
| 百炼 TraceA | 源时间 ×4 | 144.65 | [同档摘要](evidence/r10-native-final-aa-traceA09-summary.json) · [原始记录](evidence/r10-native-final-aa-traceA09.zip) |
| 百炼 TraceB | 源时间 ×12 | 115.38 | [同档摘要](evidence/r10-native-final-aa-traceB208-summary.json) · [原始记录](evidence/r10-native-final-aa-traceB208.zip) |
| 百炼 coding | 源时间 ×5.25 | 20.72 | [同档摘要](evidence/r10-native-aa-coder28-525-summary.json) · [原始记录](evidence/r10-native-aa-coder28-525.zip) · [负载摘要](evidence/r10-native-coder28-525-load.json) |
| 百炼 thinking | 源时间 ×0.62 | 9.79 | [同档摘要](evidence/r10-native-aa-thinking29-062-summary.json) · [原始记录](evidence/r10-native-aa-thinking29-062.zip) · [负载摘要](evidence/r10-native-thinking29-062-load.json) |

五源块均值绝对差的最近秩 P90 为 144.65 ms，由 TraceA 决定。逐请求绝对差的各块 P95 中，最大是 TraceB 的 1,883.14 ms，按预登记规则向上取整得到 1,884 ms 等待预算。两项进入[主执行锁](evidence/r10-local-holdout-execution-lock.json)，P95 非劣另行判定。

保留未选档。coding ×6 的第二轮未完成请求时间占比为 0.6483，低于 0.65，块均值相差 1,367.98 ms；thinking ×0.85 的第二轮占比为 0.508，两轮块均值相差 1,061.90 ms；thinking ×1.4 的占比为 0.437，用作较空闲诊断。[coding 未选档](evidence/r10-native-final-aa-coder28-summary.json) · [原始记录](evidence/r10-native-final-aa-coder28.zip) · [thinking 未选档](evidence/r10-native-final-aa-thinking29-summary.json) · [原始记录](evidence/r10-native-final-aa-thinking29.zip) · [thinking 初扫](evidence/r10-native-thinking-load-initial.json) · [初扫归档](evidence/r10-native-thinking-load-initial.zip) · [各源基线 v1](evidence/r10-native-source-load-v1.json) · [原始记录](evidence/r10-native-source-load-v1.zip) · [各源基线 v2](evidence/r10-native-source-load-v2.json) · [原始记录](evidence/r10-native-source-load-v2.zip)

OASST、TraceA、TraceB、coding 四个源的八臂校准共复算 704 个可见快照，230 个有非零候选、82 个有至少两个，两策略均无改选。[逐臂核对](evidence/r10-native-contention-policy-crosscheck-v1.json)

另外，以五个校准源的两遍同档运行作宿主机择点微基准。复算 896 个实际快照，310 个有非零备选、154 个有至少两个非零备选，仍零改选。每个快照先核对动作，再对两种 Python 择点函数各热跑 16 次，交替先后，单独计函数时间。在十个运行块中，逐快照配对差的块内中位数显示 entry-replay 每次少用 20.1–59.1 微秒。有非零备选的子集里，差值的块内中位数少用 1.1–63.5 微秒；每块的 P95 则多用 17.0–98.7 微秒。该计时不含采集快照、校验、发出命令，也不反映原生响应或闭环净收益。[逐臂计时和候选计数](evidence/r10-native-host-chooser-cost-v1.json)。

## 预填充诊断锚点复算

从归档中选取同一 OASST 块的三档容量、较空闲的 thinking、到达集中的 TraceA，以及早期排队最重的 coding。按原先导成本复现已发动作，再对每个实际可见快照用两种策略选点。先核对归档、日程、回执和原生设备事件。六臂来自四个源块，共 480 个快照，其中 139 个有非零备选，78 个有至少两个，未观察到改选。

| 原生诊断臂 | MiB | 实际提交率（请求/s） | 未清空时间占比 | 未完成峰值 | 末次提交后排空（s） | 100 ms 内提交峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASST 三档 | 64／82.25／144 | 3.85／3.84／3.84 | 0.661／0.698／0.668 | 各 2 | 0.25／0.27／0.24 | 各 1 |
| 较空闲 thinking | 82.25 | 0.73 | 0.437 | 5 | 0.18 | 4 |
| 到达集中的 TraceA | 82.25 | 2.11 | 0.979 | 25 | 8.43 | 4 |
| 早期积压 coding | 82.25 | 6.14 | 1.000 | 79 | 70.09 | 4 |

所有臂都有真实 RAM 搬运和 SSD 读取。OASST 三臂均只走按需搬运、暂存计费峰值为零，因此其中 144 MiB 点不能回答活跃预取时宽容量是否有益。100 ms 提交峰值来自原到达记录；早期 coding 带分块和管线诊断，不能把它的积压大小归因于请求突发。这里没有做新配对，也未覆盖低负载 × 宽容量的联合工况、独立 RAM 主导放置或受控突发。独立无缓冲 SSD 读见前文单对象测量，不能判定设备内部是否从 NAND 读取。[逐臂复算与原始输入身份](evidence/r10-native-prefill-diagnostic-anchors.json)。

<a id="r10-prefill-diagnostics-result"></a>

## R10-GPU-04 原生补测（2026-11-26，JST）

五个已登记配对全部完成，10 臂无失败。两条运行路径的 384 个配对恢复边界一致；两选择器在全部 768 个实际快照上复算也没有分歧。该补测与主留出分开，每个条件只有一对。

| 条件 | 实际提交率（请求/s） | 100 ms 提交峰值 | 暂存计费峰值（MiB） | 两策略平均响应差（基线减 entry，ms） |
| --- | ---: | ---: | ---: | ---: |
| thinking 较空闲，144 MiB | 0.73 | 4 | 98.25 | −1.82 |
| thinking 较忙，144 MiB | 1.65 | 4 | 98.25 | +18.97 |
| OASST 均匀提交，RAM/SSD 混合 | 3.84 | 1 | 66.25 | −10.95 |
| OASST 突发提交，RAM/SSD 混合 | 3.84–3.85 | 4 | 82.25 | +5.72 |
| OASST 突发提交，全部 RAM | 3.84–3.85 | 4 | 82.25 | −3.58 |

两条 thinking 实验的暂存实际超出默认 82.25 MiB，因此补到了宽容量确实被使用的工况；这两对没有同到达的窄容量对照，不能估计扩容效果。三条 OASST 使用相同 64 请求和 16.38 s 计划跨度。混合放置从均匀到突发，实际 100 ms 提交峰值从 1 到 4、整段未完成峰值从 2 到 4，暂存计费由 66.25 增到 82.25 MiB；改为 RAM 放置后未记录 SSD 读取。差异在提交形状和数据路径上都实际发生过，仍未使两策略作出不同恢复决策。计时符号有正有负，不能归因于选点。

[登记矩阵](../configs/r10-prefill-diagnostics-v1.json) · [逐臂统计及动作复算](evidence/r10-native-prefill-diagnostics-v1-summary.json) · [原始归档](evidence/r10-native-prefill-diagnostics-v1.zip)

## 基础事件日志扰动

在同一 OASST 校准块、同一预热、82.25 MiB、260 ms/节拍下，交叉运行关／开与开／关两对；详细块管线事件和 WDDM 采样均关闭。基础事件开启减关闭的块均值差为 −1.15、−1.16 ms，绝对值最大 1.16 ms，低于事先选用的同块 A/A 参照 40.94 ms；两对都没有选点变化，P95 差为 +2.23 和 −10.94 ms。这是该块上未见达到参照量级的测量，符号不表示记录能加速。该组的预热比稍早同配置 A/A 更短；未同时记录足以归因的系统状态，所以只用组内反序配对估计日志扰动，不用组间差异估计日志效果。[扰动审计](evidence/r10-native-logging-control-summary.json) · [两对原始记录](evidence/r10-native-logging-control.zip)。
