# R10-GPU-01 | 本机恢复、暂存池与在线执行资格

本机使用 RTX 4060 Laptop、Qwen3.5-4B Q4_K_M 和固定 llama.cpp `8a56aedd6`／Vulkan。本文汇总真实状态恢复、对象暂存及在线 FIFO 的正确性证据。可变对象目录及定向夹具见[原生对象目录](r10-native-catalog.md)，策略留出见[本地验收](r10-local-acceptance.md)。

## 同步预检与原 Vulkan 完成接口

归档运行时的同步夹具覆盖 512、1,024、1,536 token。48 个正式恢复动作在消费前重取检查点，均与保存状态逐字节相同，且与同切分重算的 8 个续算位置 logits 一致。不同切分的最大差 0.230412 与两种切分直接重算之间的差相同。预检包也保留 CUDA 独占搬运、Vulkan 预填充先导和同步强制边界对照。[预检原始包](evidence/r10-local-preflight.zip) · [摘要](evidence/r10-local-preflight-summary.json) · [恢复探针](../scripts/gpu/experiments/r10_restore_probe.cpp)

独立 Vulkan 暂存探针的 120 次搬运均逐字节回读通过。后放行 RAM 与 SSD 上传的 API 调用约 0.02–0.03 ms 返回，随后原 ggml 同步等待的中位时间约为 44.61–195.02 ms。仅推迟 RAM 放行也复现了现象。原 `ggml_vk_synchronize` 在计算队列提交 fence，主机同步返回受此前计算约束，后续改用独立搬运时间线确认上传就绪。[搬运探针](../scripts/gpu/experiments/r10_overlap_probe.cpp) · [配置](../configs/r10-overlap-probe.json) · [原始表](evidence/r10-overlap.csv) · [摘要](evidence/r10-overlap-summary.json)

<a id="r10-preflight-rerun"></a>

### 预检复算

从实验仓库根目录执行。先导原始包包含恢复日志的 CSV、12 个预填充形状、独占搬运记录与暂存对照。摘要校验各组完整性、逐字节状态、匹配续算值和容量分支。

~~~powershell
python -m zipfile -e docs/evidence/r10-local-preflight.zip outputs/r10-preflight-recheck
python -m scripts.gpu.analysis.summarize_r10_preflight outputs/r10-preflight-recheck --out outputs/r10-preflight-recheck/summary.json
python -m scripts.gpu.analysis.summarize_r10_overlap docs/evidence/r10-overlap.csv --out outputs/r10-preflight-recheck/overlap.json
~~~

重跑搬运诊断需本机归档编译件及固定源码头文件。以下设置只作用于当前 PowerShell 进程；如果换主机，先按日志确认选定独显。

~~~powershell
$runtime = (Resolve-Path data/raw/prefill/llama-vulkan).Path
$headers = 'outputs/r10/upstream/llama.cpp-8a56aedd6'
g++ -std=c++17 -O1 -fno-thread-jumps -Wall -Wextra -I "$headers/include" -I "$headers/ggml/include" scripts/gpu/experiments/r10_overlap_probe.cpp "$runtime/llama.dll" "$runtime/ggml.dll" "$runtime/ggml-base.dll" -o outputs/r10/r10_overlap_probe.exe
$env:PATH = $runtime + ';' + $env:PATH
$env:GGML_VK_VISIBLE_DEVICES = '1'
$env:GGML_VK_ASYNC_USE_TRANSFER_QUEUE = '1'
& outputs/r10/r10_overlap_probe.exe data/raw/prefill/qwen35-4b-q4km.gguf $runtime Vulkan0 outputs/r10/overlap-rerun.bin 12 10343 > outputs/r10/overlap-rerun.csv 2> outputs/r10/overlap-rerun.log
~~~

<a id="r10-async-qualification"></a>

## 独立搬运信号与已驻留前缀

### 独立观察上传完成

观察器是进程内的显式 Vulkan 层。它在成功提交原有命令后辨认该次上传的时间线信号，由观察线程直接等待信号。同一轮仍保留 ggml 的同步返回时间。观察器不插入设备命令，也不替代模型或 ggml 的同步。时间为同一进程的单调主机计时，报告的是主机最晚确认点，未对两条设备队列的活动区间做硬件时间戳对齐。

仍用 52,691,548 B 的真实循环和卷积状态。独立 Vulkan 暂存的字节回读全部一致。一组关闭观察器，一组开启；用同一提示和臂顺序（种子 10345），每组 128／512 token 各 12 轮。每个长度包含 RAM 同时放行、RAM 等计算提交后放行，以及无缓冲读取 SSD 再上传。观察器测到 120/120 次正式搬运信号，时间线值随搬运递增。下面“提前”要求上传原命令在模型计算提交返回之后提交、信号在模型计算完成之前至少 1 ms 被观察到。

| 后缀 | 实验臂 | 提前观察到信号 | 原 ggml 接口至少提前 1 ms 返回 | 上传提交至信号中位数 | 原接口比信号多等中位数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 | RAM 后放行 | 12/12 | 0/12 | 4.27 ms | 46.41 ms |
| 128 | SSD 无缓冲读后上传 | 12/12 | 0/12 | 4.21 ms | 34.52 ms |
| 512 | RAM 后放行 | 12/12 | 0/12 | 4.23 ms | 167.11 ms |
| 512 | SSD 无缓冲读后上传 | 12/12 | 0/12 | 4.26 ms | 158.90 ms |

RAM 同启对照的信号与原接口几乎同时返回；它早于模型计算提交返回，单列为先发对照。SSD 结果把此前的长同步等待从实际上传成本中分离了出来。对照的计算路径 on 减 off 中位差，在后放行 RAM 为 0.39／1.22 ms、SSD 为 0.29／1.10 ms。两次运行按 off、on 各跑整个进程，可能包含跨次运行漂移；同进程 A/A 绝对差 P90 为 2.38–3.88 ms，暂不据此宣布追踪扰动可忽略。当前上传目标仍为与模型不相连的真实显存缓冲。

### 已驻留共享前缀的实际消费

在同一模型上下文、单个统一 KV 流中，保存前缀的供体用 `llama_memory_seq_cp` 给两个序列添加使用权，然后释放供体。固定版本源码在相同 KV 流只添加单元的序列身份；循环状态也先保留共享尾部。两个序列之后分开续算，每次续算后读取另一分支的完整状态，验证写入没有串到另一个分支。比较时核验序列所属并只归一化序列头和各 KV 单元中的序列编号，位置、M-RoPE 附加字段、注意力及循环张量保留原字节。

每个 512／1,024／1,536-token 前缀 8 个正式随机提示（种子 10346）。24 个分支对均在供体释放后保有正确的完整状态，第一、第二分支写入后另一分支的状态也均逐字节相同。两条分支均与各自独立重算对照 8 个固定后续位置。384/384 个贪心 token 相同；第一分支的 logits 完全相同，第二分支最大绝对差 0.134512，第二分支此时另有一条已续算序列驻留，同布局数值对照尚未做。这些样本支持已驻留前缀的消费与写入隔离；它们没有通过真实 I/O 填充共享对象、检验容量边界或者测量策略收益。

### 复算与观察层构建

[原始记录](evidence/r10-async-qualification.zip) · [复算摘要](evidence/r10-async-qualification-summary.json) · [观察器](../scripts/gpu/experiments/r10_vk_trace_layer.cpp) · [共享消费探针](../scripts/gpu/experiments/r10_shared_consume_probe.cpp)

~~~powershell
python -m zipfile -e docs/evidence/r10-async-qualification.zip outputs/r10-async-recheck
python -m scripts.gpu.analysis.summarize_r10_async_qualification --off outputs/r10-async-recheck/timeline-off.csv --on outputs/r10-async-recheck/timeline-on.csv --resident outputs/r10-async-recheck/resident-share-v2.csv --out outputs/r10-async-recheck/summary.json
~~~

观察器由 MinGW 14.1 编译，头文件包为 Khronos Vulkan-Headers `vulkan-sdk-1.4.321.0`，ZIP SHA-256 为 `cff5e149953ccea6b5a8787150acfe6bc4d7cadb5f54c1e3cb6289407741c28b`。把[层清单](../scripts/gpu/experiments/r10_vk_trace_layer.json)复制到输出目录，在副本中将 `library_path` 改成编译所得 DLL 的绝对路径。复用[搬运探针编译和运行步骤](#r10-preflight-rerun)，正式对照先运行无层的一组（12 轮，种子 10345），再给第二组进程设置 `VK_ADD_LAYER_PATH` 为清单目录、`VK_INSTANCE_LAYERS` 为 `VK_LAYER_R10_timeline_observe`，并在探针的最后一个可选参数传入 DLL 绝对路径。观察器 DLL 编译命令：

~~~powershell
$vkInclude = 'outputs/r10/upstream/Vulkan-Headers-vulkan-sdk-1.4.321.0/include'
g++ -shared -static -static-libgcc -static-libstdc++ -std=c++17 -O2 -Wall -Wextra -I $vkInclude scripts/gpu/experiments/r10_vk_trace_layer.cpp -o outputs/r10/vulkan-observer/r10_vk_trace_layer.dll
~~~

<a id="r10-direct-restore"></a>

## 直接写入活动模型张量

本机曾把不同构建的核心库与 Vulkan 后端混用，发生 Windows 内存访问冲突。此处的测量只使用同源构建。无实验钩子的同步对照在三个深度、四种恢复臂上全部通过：共 12 个正式恢复臂的检查点字节与参考一致，且同切分重算的 8 个后续位置全部 logits 一致。这里的「同切分」是指检查点前的 token 按相同 batch 切分；切分不同的参考仍单列。同步冒烟的原始 CSV 和 DLL 摘要在[诊断归档](evidence/r10-direct-threaded-failure.zip)。

同源实验核心在空序列调用[计划钩子](../scripts/gpu/experiments/prepare_r10_restore_hook.py)。钩子复用原来的精确恢复解析器布置序列元数据，返回每笔真实张量的目的区间，本身不上传。实验将纯 RAM 快照或 RAM 注意力／SSD 循环与卷积的合成快照解析后，把有对齐要求的张量有效载荷打包到锁页内存。SSD 使用 Windows 无缓冲读取，保留实际对齐读量和进程 I/O 计数；这个计数证实请求经过文件 I/O，不能排除设备控制器自己的缓存。专用上传后端将载荷直接传到后继将使用的 KV 和循环目标，并在搬运队列上发出时间线信号；后继的计算队列在提交续算前显式等待同一信号。没有中间显存暂存对象或恢复后的预读。前驱和后继共用一次加载的模型权重，活动 KV 和循环缓冲各自独立。这样可以排除在计算尚未结束时向前驱可能读取的活动缓存写入。

[直接恢复探针](../scripts/gpu/experiments/r10_direct_restore_probe.cpp)覆盖 512、1,024、1,536 token，每个深度轮换隔离 RAM、前驱同行 RAM、前驱同行混合 SSD 三臂。每臂开始前，在计时区间外清空后继的旧设备状态。每个臂用相同提示和同切分重算作参考，续算喂入相同的 8 个 token，比较完整 logits、贪心位置及续算结束后的完整检查点。前驱同行臂启动一个 512 token 的前驱，跟在前驱提交之后的原生 ggml 计算队列标记由观察层读取；在搬运队列信号完成后，以零超时独立查询计算标记是否仍未就绪。同一主机线程操作模型与搬运后端，两个 GPU 队列仍能运行。设备标记的顺序结果不是经校准的设备区间计时，不能给出两个引擎重叠运行的精确时长。

首次长跑用第二个主机线程调用前驱同步。在第 12 轮、1,536 token 组第一臂写出前，程序以 `0xC0000005` 退出；105 个已写出的正式臂全部通过，余下 3 臂缺失。这次没有混用二进制，原因尚未确认。原始结果和进程返回码已保存，[失败轮](evidence/r10-direct-threaded-failure.zip)不并入单线程正式数据。

去掉并发的主机后端调用后，完整短测通过；修复前的第二次长跑仍在 1,536 token 的重算参考阶段退出，写出了 51 个准确的正式臂。独立 GDB 运行在加载模型时截到同一 `ntdll.dll+0x33aca` 位置，栈经过 MinGW 的 `__mingw_vsnprintf`。无需加载模型或启动 Vulkan 的小程序也复现了 `0xC0000005`；MSVCRT 的 `_vsnprintf` 和 UCRT 替代接口均通过 10 万次诊断循环。[故障诊断包](evidence/r10-direct-format-diagnosis.zip)包含栈、最小复现和过程记录。[兼容修正](../scripts/gpu/experiments/prepare_r10_windows_format.py)只调整本地 MinGW 构建的格式化调用，以 `_vscprintf` 补足旧接口的截断和长度语义，保留原格式字符串和模型操作。已定位的是这种同源运行中的格式化故障；早期未取栈的 DLL 混用故障没有归因到此处。

修正后完整运行 12 轮，另有 1 轮预热。三个恢复深度各 36 个正式臂；108 个臂的 864 个续算位置与同切分重算的完整 logits 一致，续算后的完整检查点均逐字节一致。两个同行臂合计 72/72 次在实际 DMA 信号完成后仍观察到前驱设备端标记未就绪。混合路径每次通过无缓冲文件 I/O 返回 52,695,040 B，进程读计数增量相同。原始 CSV、日志、二进制指纹和复算产物见[完整证据包](evidence/r10-direct-restore.zip)及[摘要](evidence/r10-direct-restore-summary.json)。

| 恢复深度（token） | 64 笔 DMA 有效载荷合计（B） | RAM 独占：提交至信号中位数（ms） | RAM 同行（ms） | 混合 SSD 同行的上传段（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 69,468,160 | 5.669 | 5.585 | 5.584 |
| 1,024 | 86,245,376 | 6.963 | 6.957 | 6.948 |
| 1,536 | 103,022,592 | 8.213 | 8.299 | 8.265 |

上传计时从提交返回至主机观测到设备信号，不包含读盘、解析和锁页缓冲准备。混合路径从前驱提交结束到文件读取完成的中位数分别为 28.00、25.74、25.95 ms，包含文件打开和进程 I/O 记录。驱动在双活动上下文和锁页缓冲就绪后报告约 3.24 GiB 可用预算，这是运行期间的离散快照，不是瞬时峰值或物理暂存额度。

本夹具只证明后继实际消费了搬入的状态。对象去重、在途交接、容量临界和释放见[暂存池验证](#r10-pool-qualification)。本夹具两个独立活动上下文的内存占用仅作可运行性参考，不能作为主 A/B 接口通过的依据。

<a id="r10-direct-rerun"></a>

### 同源构建与直接恢复复跑

从实验仓库根目录运行。准备固定版本的源码与 SDK，并编出同源基础生成物后，再应用两个恢复钩子。不要把归档的 DLL 混入下列路径。Windows 格式化修正需要先具备编好的直接恢复插件；已有同源修正版时从探针编译开始。观察层构建和清单生成见[异步资格复算](#r10-async-qualification)。

~~~powershell
python -m scripts.gpu.experiments.build_r10_native_vulkan --jobs 2
python -m scripts.gpu.experiments.prepare_r10_restore_hook --apply
python -m scripts.gpu.experiments.prepare_r10_vulkan_upload --apply
python -m scripts.gpu.experiments.build_r10_native_vulkan --direct
python -m scripts.gpu.experiments.prepare_r10_windows_format --apply
python -m scripts.gpu.experiments.build_r10_native_vulkan --direct --format-fix
~~~

~~~powershell
$src = (Resolve-Path outputs/r10/upstream/llama.cpp-8a56aedd6).Path
$rt = (Resolve-Path outputs/r10/llama-native-vulkan-build/bin).Path
g++ -std=c++17 -O1 -fno-thread-jumps -Wall -Wextra -I "$src/include" -I "$src/ggml/include" scripts/gpu/experiments/r10_direct_restore_probe.cpp "$rt/libllama.dll" "$rt/ggml.dll" "$rt/ggml-base.dll" -o outputs/r10/r10_direct_restore_serial.exe
$env:PATH = "$rt;$env:PATH"
$env:GGML_VK_VISIBLE_DEVICES = '1'
$env:GGML_VK_ASYNC_USE_TRANSFER_QUEUE = '1'
$env:GGML_VK_R10_CONCURRENT_BUFFERS = '1'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_R10_timeline_observe'
$env:VK_ADD_LAYER_PATH = (Resolve-Path outputs/r10/vulkan-observer).Path
$model = (Resolve-Path data/raw/prefill/qwen35-4b-q4km.gguf).Path
$layer = (Resolve-Path outputs/r10/vulkan-observer/r10_vk_trace_layer.dll).Path
& outputs/r10/r10_direct_restore_serial.exe $model $rt Vulkan0 outputs/r10/new-scratch.bin 12 1536 $layer > outputs/r10/new-direct.csv 2> outputs/r10/new-direct.log
~~~

~~~powershell
python -m scripts.gpu.analysis.summarize_r10_direct_restore --raw outputs/r10/new-direct.csv --log outputs/r10/new-direct.log --baseline outputs/r10/native-sync-baseline/native-smoke.csv --out outputs/r10/new-direct-summary.json
~~~

<a id="r10-pool-qualification"></a>

## 暂存池计费、共享及消费

浅和深候选分别恢复到 512 和 1,024 token。浅候选依赖前 512 token KV 和自身的循环／卷积状态；深候选额外依赖第二块 KV 和它自己的检查点。每轮核对两个边界共用的前 512 token KV 逐字节一致，并检查两个检查点不同。KV 来自 RAM，两个循环／卷积状态来自 Windows 无缓冲文件读。设备池固定使用 65,536 B 页，预留时按对象并集计费。KV 两块各 16,777,216 B，检查点两份各 52,690,944 B，四个对象均整页对齐。活跃模型权重、上下文和工作区另计；表中的容量是 Vulkan 设备缓冲的请求大小，不是驱动报告的全机显存峰值。

| 工况 | 设备池容量（B） | 后继在首次尝试时 | 共享 KV 上传次数 | 全场主机至设备字节（B） |
| --- | ---: | --- | ---: | ---: |
| 对象并集刚好装下 | 138,936,320 | 获准，引用同一笔在途上传 | 1 | 138,936,320 |
| 容量少一页 | 138,870,784 | 预取拒绝；队首按需恢复 | 2 | 155,713,536 |
| 首个租约在上传中取消 | 138,936,320 | 获准；继续持有同一笔上传 | 1 | 86,245,376 |

12 轮独立合成 token 提示，另有一轮预热，三个条件在每轮内换序。36 个正式工况在发起后以 Vulkan 等待接口零超时查询，36/36 次确认共享 KV 上传此刻未完成；随后才处理后继预留。三类工况均按单一时间线信号串行搬运，记录充当容量边界的真实设备分配、预留、拒绝、DMA 完成、复制、释放和模型消费。在途对象拒绝提前消费，缺 KV 或要求错误检查点的候选依赖被拒绝。接受的对象最终全部释放，未出现超额度。12/12 次取消对照中，原租约在上传完成前释放，后继仍以原时间线信号消费该对象。

60 次真实模型恢复共 480 个后续位置的完整 logits 和最终检查点均与同切分重算一致。后继的主机侧恢复元数据只保留结构，所有张量载荷事先清零；续算前的真实数据由池内对象复制到模型张量，没有把文本正确性建立在缓存旧状态上。浅／深的设备复制分别为 69,468,160／86,245,376 B；跨度描述、提交及完成的主机时间中位数约 0.75–0.94 ms，当前包括检查和封装开销。这是可复算的真实交接成本，不能当作 GPU-02 的独立无仪表延迟校准。

**活动 KV 交接补充**：对相同前缀的浅／深请求，在相同暂存容量、来源、顺序和输入下，只切换后继是否复用上一活动序列的前 512 token KV。复用臂只清除活动元数据，检查每段 KV 对应同一个 Vulkan 缓冲、相同地址区间和对象内偏移，然后跳过这些区间的设备复制，仍真实复制 KV 尾块与后继检查点。浅请求的 8 个续算 token 不属于深请求前缀的 KV 尾块，因此检验了该试验中的前缀留存与后缀替换。分 12 轮配对，共 24 个工况和 384 个续算位置，完整 logits 与最后的检查点全部通过。每次少复制 16,777,216 B；设备复制这一段的配对耗时差中位数为 0.126 ms（12 次中有 1 次未变快）。两臂的主机至设备流量和容量相同。见[原始数据与指纹](evidence/r10-active-kv-handoff.zip)和[摘要](evidence/r10-active-kv-handoff-summary.json)。

**本轮资格范围**：真实页池的计费、共享、在途接管与消费，以及同一活动上下文、偏移可重用的前缀交接通过。这个补充对照只免去第二次 active-KV 复制。后续[连续 FIFO 资格](#r10-fifo-qualification)另验证了活动 KV 的低层上传可以省略。冷对象仍可能由页池复制到活动序列；可变目录的异址保留和队首混合需求见[定向真机夹具](r10-native-catalog.md#可变目录定向真机正确性)。取消首租约是对象生命周期的故障注入，只作资格验证。

**暂存装不下单个候选时的队首路径**：补了独立的直写动作。按两个深度各对比 12 个同输入工况，控制组将候选放入刚好容纳它的暂存池；直写组把额度减 65,536 B，使预取被拒，空闲队首随后真实逐对象搬到其已有的活动张量。两个组读盘和上传字节相同，每臂分别读盘 52,690,944 B、上传 69,468,160 B 或 86,245,376 B。直写组的暂存计费始终为零，没有设备内复制；暂存组要将同等字节复制到活动张量。48 臂共 384 个后续位置的完整 logits 和最终检查点完全一致。这验证了无其他租约、搬运通道空闲的按需恢复；同一调度器中带后继租约的混合路径见[定向真机夹具](r10-native-catalog.md#可变目录定向真机正确性)。见[原始事件和指纹](evidence/r10-head-demand.zip)及[核对摘要](evidence/r10-head-demand-summary.json)。

[选点快照入口](../src/restore_edge_select/live_snapshot.py)单独通过 CPU 校验。它把已到达请求、承诺边界、预留、就绪对象和已发生的设备阶段起点交给原选点函数；剩余工作由已冻结的成本推算，接口没有成功完成时刻或未来到达字段。60 个 R07 短负载上，两策略各 420 次选择与原执行器逐次相同，回放汇总与提交边界也一致。404 个仍有未到达请求的快照中，只把未来到达整体后移 437 ms，当前选择均未改变。`entry-replay` 路径上的 420 个快照有 122 个与在同一快照评估的 `local-cost` 选择不同。被检查的原 `choose` 函数 AST 与冻结归档一致。见[快照审计](evidence/r10-live-snapshot-audit.json)，用 `python -m scripts.validation.check_r10_live_snapshot` 复算。真实快照驱动的后续验证另见[在线动作诊断](#r10-online-fixture)。

[数据、日志与运行指纹](evidence/r10-stage-pool.zip) · [逐事件校验摘要](evidence/r10-stage-pool-summary.json) · [探针](../scripts/gpu/experiments/r10_stage_pool_probe.cpp) · [设备页账](../scripts/gpu/experiments/r10_stage_pool.h)。复跑使用和[异步直写](#r10-direct-rerun)相同的同源 DLL、MinGW 参数、模型、Vulkan 设备选择及运行设置。常规暂存池夹具使用以下探针。若此前运行过直写探针，先清掉其显式观察层设置。

~~~powershell
Remove-Item Env:VK_INSTANCE_LAYERS, Env:VK_ADD_LAYER_PATH -ErrorAction SilentlyContinue
g++ -std=c++17 -O1 -fno-thread-jumps -Wall -Wextra -I "$src/include" -I "$src/ggml/include" scripts/gpu/experiments/r10_stage_pool_probe.cpp "$rt/libllama.dll" "$rt/ggml.dll" "$rt/ggml-base.dll" -o outputs/r10/r10_stage_pool_probe.exe
& outputs/r10/r10_stage_pool_probe.exe $model $rt Vulkan0 outputs/r10/new-stage-scratch 12 > outputs/r10/new-stage.csv 2> outputs/r10/new-stage.log
python -m scripts.gpu.analysis.summarize_r10_stage_pool --raw outputs/r10/new-stage.csv --out outputs/r10/new-stage-summary.json
~~~

仅重复活动 KV 对照时，探针在重复数之后追加 `active-kv`，复算脚本另加 `--active-kv`，使用新的日志和临时路径。

补充的队首路径和计算信号使用同源运行库，探针改用 `-O0 -Wall -Wextra` 编译（本机 MinGW 优化探针时内部编译失败，不是 GPU 运行失败）。队首路径在重复数后追加 `head-demand`，再用 `python -m scripts.gpu.analysis.check_r10_head_demand CSV` 独立核对。在线控制入口追加 `--observer outputs/r10/vulkan-observer/r10_vk_trace_layer.dll` 会启用带独立计算信号的六臂模式。

<a id="r10-online-fixture"></a>

## 固定三请求的在线动作及设备复核

在 RTX 4060 Laptop GPU 上，`entry-replay` 和 `local-cost` 已各自根据设备上传查询时的快照发出边界，实际控制 Qwen3.5-4B 恢复和续算。正式 12 轮，每轮对两档暂存、两条策略换序执行，共 48 个工况。与同切分全重算相比，48 个工况里浅请求和目标请求的 768 个后续位置完整 logits 全部一致，最终状态也逐字节相同。按原始设备事件离线重建快照，两策略在各自同一快照上的发出动作均可复算。

前驱先提交 512 token 的全重算；第二个请求固定从 512 token 恢复，其前缀 KV 与循环／卷积检查点已预留。上传前缀 KV 后用 Vulkan 零超时查询，48/48 次尚未完成；把这个事件作为第三个请求的人工到达点。快照包含三条已知请求、两条承诺边界、实际预留、就绪状态和已提交阶段的时间。前驱计算标为已提交且未被主机同步；本探针未单独查询它的设备完成标记。两条策略用同一份按本机预检测量取整的成本表。探针通过本地管道收到边界之后才尝试为目标预留；设备消费结束后再用同一快照复算两条策略。

两策略在 48 个快照上都选了 1,024 token，估算由 512 token 的续算成本差主导。前两条边界和第三条的发布事件由夹具固定，计时含管道和回读。容量刚好覆盖并集时接受目标预留；容量少 64 KiB 的 24 个工况中，全部拒绝目标预取、保留边界，等目标成为队首后才按需恢复。另做 2 轮强制动作，分别证实目标在 0、512、1,024 处确实重算 1,024、512、0 个前缀 token，并分别在设备上恢复所需的精确状态；合计 6 个工况和 96 个后续位置通过。强制轮只用于验证动作实现。

**设备状态复核**：在同一接口加独立的计算队列信号与零超时查询。另跑 12 轮 6 臂，每轮随机换序，72/72 次选点时上传未完成，其中 48 次还观察到前驱计算未就绪，24 次在受控的前驱完成工况观察到计算已就绪。后一类快照把已完成前驱从队首移除，未把它的历史完成时间传给选点。两类状态均能复算发出的动作，1,152 个后续 logits 位置与最终状态完全一致。两策略仍都选 1,024；没有主实验收益证据，也没有把这两个独立查询宣称为同刻事件的完整验证。见[带设备信号的证据](evidence/r10-device-witness.zip)和[摘要](evidence/r10-device-witness-summary.json)。

[完整原始事件、快照、指纹和代码](evidence/r10-online-action.zip) · [摘要](evidence/r10-online-action-summary.json)。用[真机控制入口](../scripts/gpu/experiments/run_r10_online_diagnostic.py)复跑，`--forced` 切换为三条强制动作。探针与[暂存池](#r10-pool-qualification)使用同一编译命令和 DLL；不加载 Vulkan 观察层。归档在本地解开后，可不启动模型独立复算两个汇总：

~~~powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m scripts.gpu.analysis.check_r10_online_evidence outputs/r10/online-forced-smoke outputs/r10/online-native
~~~

<a id="r10-fifo-qualification"></a>

## 连续 FIFO 与受控改选

原有单次选点探针只有目标请求由策略选择。本次把冻结选点入口接到可连续处理请求的同一个控制器：每条请求第一次入窗即提交边界，计算槽和搬运通道分别以完成通知推进，预取超出额度时保留边界，让队首按需取数。控制器接收带身份的完成通知，不用预测成本判断 GPU 或 SSD 是否完成。成本估计仍由原策略在选点时使用。代码见[控制器](../src/restore_edge_select/live_fifo.py)、[真机驱动](../scripts/gpu/experiments/run_r10_fifo_link.py)和[原生桥](../scripts/gpu/experiments/r10_stage_pool_probe.cpp)。

同刻按冻结执行器顺序承认搬运完成、推进已完成计算，再使新到达可见；空闲队首先于预取请求使用搬运通道。CPU 资格核对覆盖 60 个冻结 R07 短负载，每条策略各 420 次承诺。快照、候选边界、暂存预留时刻、搬运发出和完成、请求完成时间及暂存峰值分别一致。定向夹具另覆盖三事件同刻、共享循环检查点需复制、单个候选也装不进预取额度、零续算在同刻推进。此处的同刻通过指离散驱动结果；真机只能报告同一采样批内看见哪些信号。见[核对结果](evidence/r10-live-fifo-audit.json)及[复算](../scripts/validation/check_r10_live_fifo.py)。

本机原生桥已使用两种独立的 Vulkan 设备信号。一种表示串行上传可以消费，另一种表示当前计算队列完成。真实提示已在计时外准备，三条请求由独立线程按绝对偏移提交。驱动分别记录计划时刻、提交时刻和选点能看见的到达时刻。三条请求都由同一个控制器决定，并用同一活动上下文连续执行；核对包含两个恢复候选完整的后续 logits 和所有请求最终状态。没有用事后的完成时刻参与选点。共享 KV 借用要求对象身份一致且活动张量缓冲和偏移相同；续算接管时不再上传或复制该块。

原生 I/O 条件，两策略各在下列三组跑 8 个短块，每个短块 3 个请求，共 48 块、144 次在线选择与计算。768 个续算位置逐项一致，48 块的原始事件均能复算实发动作。两策略都选择 0／512／1,024；这批未观察到原生选点差异。读次数按实际发出的对象上传计算。

| 工况 | 暂存容量（B） | 计划到达（ms） | 两策略合计 |
| --- | ---: | --- | --- |
| 宽容量、早到达 | 138,936,320 | 0／35／70 | 64 次暂存上传，已预留请求共用同一份 KV |
| 队首直写 | 69,402,624 | 0／35／70 | 80 次直接上传，暂存峰值为零 |
| 宽容量、后继晚到 | 138,936,320 | 0／35／300 | 64 次暂存上传，16 次从在用张量接管 KV |

[事件、动作快照、日志与二进制指纹](evidence/r10-live-fifo-native.zip) · [核对摘要](evidence/r10-live-fifo-native-summary.json)。核对桥接结果用 `python -m scripts.gpu.analysis.check_r10_fifo_link <不含后缀的记录路径>`。可直接读取同名 CSV 和 JSONL，无需 GPU。

另以同输入强制后继借用／不借用活动 KV，各做 12 轮配对，并用连续执行器的构建件重跑 12 轮。两批都逐轮少上传 16 MiB、少设备内复制 16 MiB，容量记账相同；384＋384 个后续位置及最终状态一致。每批 24 个工况的采样中，前驱的第一个续算位置仍未完成。这个对照只覆盖紧接的同址接管。[两批原始事件](evidence/r10-native-kv-borrow.zip) · [首批摘要](evidence/r10-native-kv-borrow-summary.json)。

**受控搬运条件下，两策略首次在这个连续真机夹具发出了不同边界。** 按[试验前登记](../configs/r10-controlled-fifo-diagnostic.json)，在真实无缓冲 SSD 读取前增加 250 ms 软件等待。等待不停止计算状态查询，两策略使用相同的等待和成本表。各跑 8 个同种子短块：ENTRY 每次选择 0／512／512，复用同一个浅循环检查点；`local-cost` 每次选择 0／0／1,024，第二个请求重算，第三个请求读深检查点。两条路径每块分别发出 2、3 次对象上传，均只读一次真实 SSD 检查点。每次传输和选点可复算，16 块共 256 个续算位置和最终状态通过。受控组的客户端计划误差为约 −0.007 至 +14.4 ms，原始记录保留；随后让客户端在提前唤醒时继续等到绝对偏移。两条臂按固定先后执行，前驱同动作的两个组自身用时也不同，管道和状态回读均计入记录。本轮只确认机制在该成本干预下会改变真实动作，不据此估计原生或因果延迟收益。见[受控记录](evidence/r10-live-fifo-controlled.zip)及[复算摘要](evidence/r10-live-fifo-control-summary.json)。

这个桥接夹具固定三请求、四对象。可变目录、队首混合直写与暂存读取，以及非相邻租约的后续验证见[原生对象目录](r10-native-catalog.md#可变目录定向真机正确性)。
