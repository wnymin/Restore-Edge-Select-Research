# R10-GPU-01／DATA-01 | 两类输入的原生对象目录

2026-11-25，最初在本机 RTX 4060 为 4 个百炼先导块和 4 个 OASST 校准块生成原生对象，并通过恢复、续算和归档核验；当时 14 个保留／留出块只检查了输入格式。随后在执行冻结下为其中 12 个主留出块生成目录，并单独核验和归档准备结果，见下文。这里的正确性证据不用于估计算法收益。

| 输入 | 校准／先导块 | 留出／保留块 | 请求数 |
| --- | ---: | ---: | ---: |
| 百炼轨迹重建 | 4 | 8 | 1,152 |
| OASST 独立文本 | 4 | 6 | 640 |
| 合计 | 8 | 14 | 1,792 |

[计划清单](evidence/r10-native-plans.json)记录每个源块、对象数、依赖引用及两份输入文件的 SHA-256。22 个块共 2,819 个候选，其中边界 0、512、1,024、1,536 分别有 1,792、559、256、212 次。按块计有 351 个独立注意力 KV 对象、351 个循环与卷积检查点；同一块里 305 个对象身份出现在多个候选中。此数只描述输入目录里的共享机会。原始到达与请求内容的来源口径见[独立文本输入](r10-independent-input.md)。

每块有两个派生文件：原生程序读入请求 token、候选和依赖序列，调度控制器读入相同的依赖及原始时间字段。KV 对象按 512 token 整块切分，静态放置为 RAM；循环和卷积检查点放在 SSD。对象身份含截至该块的**完整前缀**，并要求 token 逐个相同；只比较当前 512 个 token 不足以复用。不同边界可以共享同一 KV 块，但不能彼此替代循环检查点。调度文件中的成本、容量和窗口仍是诊断占位；OASST 时间字段仍是合成到达刻度。它们没有冻结为真机计时参数。

[主机审计](evidence/r10-native-catalog-host-audit.json)逐项比对了两份文件的 token、候选和依赖，并用不链接 GPU DLL 的 C++ 读取器验收所有块。三种深度的构造元数据均可读取；错误前缀、同一对象拆成两个身份、额外计划内容、对象内偏移错误、KV 顺序颠倒、张量区间重叠、元数据截断共 7 种失败输入均被拒绝。这些检查不执行模型。另用构造文件验证了独立产物校验可识别缺失的二进制、长度不符、缺少检查点行和未写完的尾行。

[原生目录生成器](../scripts/gpu/experiments/r10_prepare_catalog.cpp)在计时外从 0 重新计算每个独立检查点，按布局拆回原生张量。声明可共享的对象必须在不同独立检查点的导出中逐字节一致。每个检查点写出一份去掉张量内容的紧凑描述和各依赖对象文件；验证阶段从原生设备写入接口还原，再与从头重算比较完整状态、8 个固定续算位置的 logits 和续算后的完整状态。目录准备时 KV 用文件补充到锁页缓冲；可变目录执行器启动时才将其载入常驻 RAM。检查点用无缓冲读取经 RAM 中转上传。准备记录中的耗时只用于估算准备工期。

8 个校准块合计通过 106 个原生检查点（512／1,024／1,536 分别是 63／25／18 个），848 个续算位置及续算后的模型状态全部逐字节一致。61 次跨独立检查点的共享 KV 重算字节比较均一致。首次封存校验拒绝了 ggml 张量合法的 ` (view)` 名称；原生进程退出码为 0，放行名称内的可打印空格后，直接复核原件并通过。生成器随后换用 [Windows 无缓冲异步读](../scripts/gpu/experiments/r10_catalog_source.h)并重用[原生设备地址映射](../scripts/gpu/experiments/r10_catalog_target.h)。单独 3 次轮询均真实观察到未完成，完成后各读满 52,690,944 字节并与普通读取逐字节一致。这证明该路径可被轮询，未测算自然负载中的读盘带宽或重叠收益。日志、回执和哈希见[原始记录](evidence/r10-native-calibration.zip)和[复核摘要](evidence/r10-native-calibration-summary.json)。

主留出 12 块的目录准备记录另已核验。175 个原生检查点均通过独立重算的状态核对，1,400 个续算位置的 logits 全部精确一致，续算后的状态也一致；另有 107 次跨检查点的共享对象身份核对。归档包含目录封条、生成回执和数值校验日志，不包含原始对象载荷。正式主实验归档完成后，再将每个计时臂使用的封条哈希与此归档逐一对照。[主留出准备归档](evidence/r10-native-primary-preparation.zip) · [逐块复核摘要](evidence/r10-native-primary-preparation-summary.json) · [准备归档工具](../scripts/gpu/analysis/archive_r10_primary_preparation.py)。

[启动器](../scripts/gpu/experiments/run_r10_catalog_prepare.py)使用新的输出目录，分别保留 CSV、stderr 和含返回码的运行回执；主留出与服务留出还要求各自的执行冻结。子进程使用无窗口启动并抑制系统错误框；退出异常保留在日志中。退出码为 0 后，[产物校验](../scripts/gpu/analysis/check_r10_native_artifacts.py)还要检查对象完整性、分段覆盖、每条数值校验记录，并给文件生成哈希，才能写出校验清单。部分输出即使看起来可读也不能进入计时。

在项目根目录复核 CPU 数据（使用[同源运行时](r10-stage-pool.md#r10-direct-rerun)对应的上游头文件）：

~~~powershell
$src = (Resolve-Path outputs/r10/upstream/llama.cpp-8a56aedd6).Path
g++ -std=c++17 -O0 -Wall -Wextra -Werror -I "$src/include" -I "$src/ggml/include" scripts/gpu/experiments/r10_catalog_inspect.cpp -o outputs/r10/r10_catalog_inspect.exe
python -m scripts.validation.check_r10_native_catalog --summary docs/evidence/r10-native-plans.json --plans outputs/r10/native-plans --inspector outputs/r10/r10_catalog_inspect.exe --out outputs/r10/native-catalog-host-recheck.json
~~~

复跑真机恢复可从百炼小先导块 `qwen_traceB_blksz_16-208-pilot` 和 `oasst-cal-01` 起步。编译目录生成器时沿用同源 `libllama.dll`、`ggml.dll`、`ggml-base.dll` 以及相同头文件，入口 `python -m scripts.gpu.experiments.run_r10_catalog_prepare --help`。可变目录和队首／后继同时持租约的真机验证见下文。

## 可变目录执行器的首个真机短测

[执行器](../scripts/gpu/experiments/r10_catalog_worker.cpp)接入同一个 `LiveFIFO` 控制器，真实分配一块固定显存暂存池、常驻 RAM KV 及锁页缓冲。一个 SSD 对象从发读到完成上传均占用串行搬运通道；轮询读盘和上传期间仍可查询计算完成。进程按物理地址判断旧 KV 能否原地交接，地址不一致时若仍有租约则先复制到已收费的页。共享循环状态可能被续算写入，所以也必须先保留可复用副本。[物理账本](../scripts/gpu/experiments/r10_catalog_ledger.h)及其[主机夹具](../scripts/gpu/experiments/r10_catalog_ledger_probe.cpp)已覆盖在途加入租约、队首直写与暂存拷贝合用、循环状态写前保留、跨非相邻请求保留 KV 等状态转换。设备是否真的保留正确字节，需另有真机状态对照。

2026-11-25，百炼 `TraceB-208` 先导块用最长合法边界、80 MiB 暂存和源到达间隔的百分之一运行了 96 条请求。边界 0／512／1,024 分别为 64／31／1 次。设备确认 35 次传输，含 19 次无缓冲 SSD 读；其中 3 个对象直接写入执行缓冲，32 个先写暂存池。观察到 62 次暂存拷贝、4 次覆盖旧活动区域前的 KV 保留；固定暂存分配中的已收费页峰值为 69,468,160 字节，结束时归零。[逐事件核对](evidence/r10-variable-fifo-smoke-summary.json)确认了策略提交边界、请求完成顺序，以及控制器与实际发出的传输对象逐项相符。[原始证据](evidence/r10-variable-fifo-smoke.zip)保留三份记录。

这个先导块未产生同一请求同时直写且从暂存拷贝的事件，也未在在线执行中逐请求比较最终状态。后续用下列定向真机夹具验证这些分支。复跑入口：`python -m scripts.gpu.experiments.run_r10_catalog_fifo --help`。

## 可变目录定向真机正确性

在同一 RTX 4060 和执行器上，用 66.25 MiB 暂存跑了三个合成夹具。每条请求先单独从头重算，保存原始完整状态字节和最后位置的全部 logits；在线运行后逐项比较。参考生成、完整状态回读和额外核对只在验证模式打开，不计入策略计时。三个夹具合计 8 条请求，状态和末位 logits 均逐字节一致。

| 夹具 | 必须观察到的设备路径 | 完整状态／末位 logits 一致 |
| --- | --- | ---: |
| `mixed-private-staged` | 1,024-token 队首无法占用整个候选的预取页，从暂存池取得后继预取的共享 KV，另外两个对象直接写入执行缓冲 | 3／3 |
| `direct-handoff` | 后继在 SSD 已发起、控制器尚未承认完成时取得租约；循环状态在续算前保留到暂存池，后继按同地址接管 KV | 2／2 |
| `nonadjacent-loan` | 中间的全重算请求覆盖执行区域前把仍有租约的旧 KV 保存到暂存池，最后请求从池中取得 KV 与循环状态 | 3／3 |

每组的后继按首个计算或 SSD 事件发布，只检验状态和依赖，不代表自然到达，也不要求发布时物理读盘必然还未结束。三组都达到配置的页预算、没有超额，逐项核对了实际传输与控制器顺序。完整源块的逐请求正确性未与从头重算比较。自然负载下的策略对照见[主实验](r10-local-acceptance.md#r10-gpu-04-primary)。[定向夹具](../scripts/data/build_r10_mixed_qualification.py) · [归档核验](../scripts/gpu/analysis/archive_r10_mixed_qualification.py) · [结果摘要](evidence/r10-native-mixed-qualification-summary.json) · [原始记录](evidence/r10-native-mixed-qualification.zip)。成本分线见[原生校准与计时诊断](r10-native-cost.md)。
