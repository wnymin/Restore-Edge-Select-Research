# 同槽提前恢复：任务排队、冷读和目标入批

本机双槽结果把三个时间段分开了：恢复任务可能在服务端积压数百毫秒；读冷文件对恢复回调的净影响约 10–17 ms；目标获得槽以后仍可能因同行的空闲槽保存或运行中检查点创建而迟迟不进入首批。关掉空闲槽保存消除了对应的入批尖峰，但没有使提前深恢复稳定优于浅驻留。

## 恢复调用时间花在哪里

复用 [提前恢复](dynamic-gpu-early-restore-plan.md) 的 88 条实测，依据服务端任务 ID 关联同行首批、恢复请求和目标首批。恢复的最后一次分派确实执行了该任务：之后没有解码拒绝，分派到结果发送的时长与独立 HTTP 回报相差不超过 2 ms。恢复排队从服务端收件到最后一次分派；回调从最后一次分派到结果发送；客户端调用扣去这两段是外部余量。目标的收件到获槽、获槽到首批另计，各段只在同一时钟内相减。

| 同行输入 token | 提前 ms | 冷恢复：队列 / 回调 / 外部余量中位数 ms | 至少一次被解码拒绝 | 释放前已就绪 |
| ---: | ---: | ---: | ---: | ---: |
| 2,816 | 400 | 414.94 / 147.11 / 6.12 | 6/6 | 0/6 |
| 3,072 | 400 | 386.94 / 232.27 / 1.43 | 5/5 | 0/5 |

队列占了主要部分。拒绝日志证明恢复任务在解码期间至少一次被推迟，没有给出每一轮在排队或 GPU 上的精确归因；回调的总时间也不能用作独立的 SSD 带宽参数。[首轮任务记录](../../evidence/dynamic-gpu-queue-7321.csv) · [留出任务记录](../../evidence/dynamic-gpu-queue-7322.csv)

## 同一检查点的冷文件与 RAM 热页

新种子 7323，固定相同深边界和释放时刻。每个同行长度和提前量做 4 对，配对使用相同提示词，冷路径直接恢复，热路径先正常读取同一新生成文件以加热 Windows 文件缓存。文件约 90.47 MB，冷路径恢复期间卷读取 90,472,960 字节，热路径为 0；16 对均通过缓存长度与首 token 核对。这里比较的是页缓存状态对回调总时长的影响，并未单独计时磁盘指令或 GPU 操作。

| 同行输入 token | 提前 ms | 冷减热的回调时间：配对中位数 ms | 热路径回调中位数 ms |
| ---: | ---: | ---: | ---: |
| 1,920 | 0 | 13.24 | 42.95 |
| 1,920 | 400 | 11.55 | 180.13 |
| 3,072 | 0 | 12.53 | 223.38 |
| 3,072 | 400 | 12.76 | 224.77 |

16 对的热路径回调均更短，差值 9.86–16.91 ms。长组提前 400 ms 时即使不读盘，队列和回调中位数仍分别是 406.15、224.77 ms。每个成功回调内可定位一次目标槽读到 1,152 个缓存单元的元数据日志：长组提前 400 ms 在热路径上，分派至该日志的中位数为 **198.91 ms**，该日志至结果发送为 **25.19 ms**。冷热净差主要出现在日志之后，各组配对中位数约 **11 ms**，日志之前仅约 **0.45–2.32 ms**。因此剩下的大头在读到缓存单元元数据之前；仍不能把这段总时间直接认作设备同步。内存读取、反序列化、设备传输以及等待已提交计算等尚未独立计时。[冷热配对记录](../../evidence/dynamic-gpu-queue-warm-7323.csv)

## 在读取有效负载之前的等待

在 b10923 的本机 DLL 中，导出函数 llama_state_seq_load_file（RVA 0x11B700）先调用 RVA 0x110220，再进入读文件的 RVA 0x117FE0 路径；导出的 llama_synchronize（RVA 0x11A4D0）直接跳至同一 RVA 0x110220。二进制的 SHA-256 为 F644529B58FF1B05DBE5748B5BAE2FE83F0A39182EC7E8058E02F37D12EECD76。用导出表和对应地址反汇编核对这两个调用；未开 Vulkan 算子计时。

种子 7329，逻辑／物理批量为 1,024／512，关闭空闲槽保存和自动检查点。3,200 与 3,328 token 同行各测五个配对块。每块先驻留相同的 1,152 token 深边界，随机交错同行活跃／已经返回两种状态和三种文件：8 字节无效热文件头、有效 RAM 热文件、有效冷文件。活跃组约在同行提交后 150 ms 发起恢复，日志确认回调结束时同行仍活跃，分派前最后一条同行缓存位置记录为 512 token。结束组在收到同行首 token 后立即发起。60 条测量中无效文件均以 HTTP 400 拒绝，日志确认拒绝原因是文件头，未出现状态元数据或恢复字节，卷读数为 0；有效冷文件卷读量约 90.47 MB，有效热文件卷读量不超过 32 KiB。所有同行首 token 块内一致。

| 同行 token | 文件条件 | 活跃／已结束：分派到日志中位数 ms | 活跃减已结束：配对中位数 ms |
| ---: | --- | ---: | ---: |
| 3,200 | 无效热文件头 | 210.15 / 0.23 | +210.00 |
| 3,200 | 有效热文件 | 237.14 / 23.44 | +212.20 |
| 3,200 | 有效冷文件 | 242.65 / 31.77 | +213.74 |
| 3,328 | 无效热文件头 | 212.07 / 0.26 | +211.84 |
| 3,328 | 有效热文件 | 233.46 / 26.98 | +205.64 |
| 3,328 | 有效冷文件 | 241.14 / 31.08 | +212.67 |

无效文件计到拒绝日志，有效文件计到首条目标槽状态元数据日志；两类终点不同，比较的是各自相对同行已结束的增量。无效文件在活跃组仍先等约 210–212 ms；拒绝日志到回包仅约 0.26–0.34 ms，分派前另有约 64–68 ms 排队。有效文件的对应增量相近，说明主要额外回调时间在有效负载读取之前。结合二进制调用顺序和同行已结束的阴性对照，结果支持等待已提交后端工作是主因。这不是精确的 GPU fence 计时；文件打开、头部检查和少量 CPU 开销仍在日志跨度内，也没有把有效路径的解析与上传分开。约 210 ms 属于特定批次状态，不能作为每个检查点或每次同步的常数。[60 条对照记录](../../evidence/dynamic-gpu-header-7329.csv)

## 空闲槽保存干预与性能复核

原留出短组两条最大的延迟反例中，日志显示服务端获槽后、首批前同步保存同行的空闲槽到提示词缓存。用原种子和工况重新启动服务，加上 `--no-cache-idle-slots`。40 条同工况测量的首 token 均与默认运行一致；关闭后获槽至首批没有观察到保存标记。两次服务是独立启动且默认先跑，不能把端到端时差全部归因于开关；下表中的保存时段由日志直接计时，开关复测显示相应尖峰消失。

| 1,920 token，种子 7322 | 原始保存时段 ms | 目标获槽至首批：默认 → 关闭 ms |
| --- | ---: | ---: |
| 提前 400 ms，第 3 号块（从 0 数） | 213.189 | 213.263 → 0.056 |
| 提前 150 ms，第 4 号块（从 0 数） | 163.294 | 163.372 → 0.059 |

关掉保存后，长组提前 400 ms 仍是 **0/5** 提前就绪，恢复排队中位数 386.11 ms，目标获槽到首批中位数 183.55 ms。短组提前 400 ms 是 **5/5** 提前就绪，目标比按需深恢复平均快 25.53 ms、同行慢 54.98 ms；目标仍比浅驻留平均慢 26.44 ms。四个关闭后的提前工况中，目标／共同完工平均均不及浅驻留。保留同槽提前恢复的负结果，不据此推断隔离的恢复执行器也无效。[关闭空闲槽保存的记录](../../evidence/dynamic-gpu-queue-no-idle-7322.csv)

## 长组获槽后在等什么

用同行任务 ID 配对进入检查点创建分支和创建完成的日志，只计完全落在目标获槽至首批之间的时段。种子 7322、长组提前 400 ms，默认设置下 5/5 次在此区间创建了同行的运行中检查点，中位数 **183.77 ms**，而整个区间中位数为 184.11 ms。关闭空闲槽保存后，4/5 次仍出现此分支，两个中位数分别为 **183.24 ms** 和 183.55 ms。种子 7321 的长组也复现：提前 400 ms 时 6/6 次观察到创建，区间中位数 192.94 ms，其中创建时段的中位数为 192.65 ms。这两种缓存操作有各自的日志标记，不计为恢复任务自身的数据搬运。

在已关闭空闲槽保存的基础上，进一步关闭自动运行中检查点（`--ctx-checkpoints 0`），用种子 7322 重跑完整 40 条。日志没有自动检查点完成事件，显式深恢复仍通过检查点字节、使用边界和首 token 核对。对于基线出现同行检查点的 15 条匹配测量，目标获槽至首批由约 183–186 ms 降至不足 0.3 ms。恢复队列仍然存在：3,072 token 提前 400 ms 仍是 0/5 提前就绪，恢复排队中位数 380.70 ms；目标收件至获槽的中位数从 24.79 ms 升到 203.67 ms。关闭该功能消除了入批阻塞，但前段等待增加，不能把约 183 ms 算作端到端收益。[关闭自动检查点的记录](../../evidence/dynamic-gpu-queue-no-checkpoints-7322.csv)

对照中出现一条值得留出检验的窄路径：关闭两项缓存操作后，1,920 token 同行下，提前 400 ms 的深路径相比浅驻留目标／完工平均快 **22.40 ms**（5/5），两项响应和平均改善 **19.97 ms**，同行平均慢 2.43 ms；3,072 token 同组反而让目标慢 12.37 ms、同行慢 53.88 ms。短组也由 5/5 提前就绪变成 0/5，说明提前就绪率不能单独用来选择边界。此开关改变真实服务缓存行为，以上只比较该设置下的路径，未证明默认配置下可部署。

留出使用种子 7325 和未测过的同行长度 1,856、2,048、3,200，固定释放 550 ms 和按需、提前 150 / 400 ms、浅驻留四条路径。主对照是提前 400 ms 对浅驻留：两个邻近短工况均需改善目标及响应和的平均值，且各至少 3/4 对改善目标，才视为短负载收益在邻近长度留出复现。3,200 token 是反例对照；任何提前量的同行损失与完工效果均独立报告。

12 个留出块、48 条有效测量的块内首 token 一致，目标总是最后完成。表中正值表示比浅驻留快，收益是配对平均值。

| 同行 token | 提前 ms | 释放前就绪 | 目标／完工收益 ms | 同行收益 ms | 响应和收益 ms | 目标更快 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,856 | 150 | 4/4 | +48.55 | +4.18 | +52.73 | 4/4 |
| 1,856 | 400 | 3/4 | +38.07 | +1.53 | +39.60 | 4/4 |
| 2,048 | 150 | 0/4 | −14.94 | +2.98 | −11.96 | 1/4 |
| 2,048 | 400 | 0/4 | −24.90 | −6.70 | −31.59 | 0/4 |
| 3,200 | 150 | 0/4 | −9.99 | +67.91 | +57.92 | 1/4 |
| 3,200 | 400 | 0/4 | −15.07 | +64.22 | +49.16 | 1/4 |

短负载收益未通过预定的跨长度条件：1,856 仍受益，2,048 两档均翻转。两个工况提前 400 ms 的冷回调中位数同为约 53 ms，恢复队列却从 339.92 升到 410.49 ms，释放时的就绪也从 3/4 降到 0/4。3,200 两档领先的平均同行收益由第 0 块主导：该块浅驻留同行耗时 1,452.78 ms，另外三块约 967–980 ms；提前 400 ms 在另外三块均使同行变慢。该异常留在平均值中。当前不能从这些长度推导可部署的同槽提前规则。[跨长度留出记录](../../evidence/dynamic-gpu-queue-no-checkpoints-7325.csv) · [缩小逻辑批量的后续对照](dynamic-gpu-batch.md)

试过二进制内置的 Vulkan 算子计时。单个配对块（种子 7324、3,072 token、关闭空闲槽保存）中，开启逐图计时后，提前 400 ms 路径的恢复回调由 229.39 ms 变为 57.19 ms、队列由 380.79 ms 变为 597.01 ms，同时同行检查点创建由 183.93 ms 变为 21.48 ms。两次独立服务启动、只测一块；这个差异说明开启计时的运行不足以代表正常路径，不能拿差值换算上传或 GPU 等待。Windows GPU 事件追踪的本机试启也未能启用系统性能记录策略（错误 `0xc5585011`）。回调内部的拆分需要不改变调度行为的设备同步和传输计时。[计时器试跑记录](../../evidence/dynamic-gpu-queue-vulkan-probe-7324.csv)

## 复算

已归档的 CSV 可直接复算，不需要 GPU。原始多行服务日志留在本机；从原始 CSV 和日志重建时向分析脚本传入 `--log` 与 `--out`。使用 `.local/hardware` 的命令是本机重跑入口。

~~~powershell
python -m scripts.gpu.experiments.study_dynamic_gpu_restore_queue docs/evidence/dynamic-gpu-queue-7322.csv --idle-save-off docs/evidence/dynamic-gpu-queue-no-idle-7322.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_restore_queue docs/evidence/dynamic-gpu-queue-warm-7323.csv --warm-pairs
python -m scripts.gpu.experiments.study_dynamic_gpu_header docs/evidence/dynamic-gpu-header-7329.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_early docs/evidence/dynamic-gpu-queue-no-idle-7322.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_restore_queue docs/evidence/dynamic-gpu-queue-no-idle-7322.csv --checkpoints-off docs/evidence/dynamic-gpu-queue-no-checkpoints-7322.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_early docs/evidence/dynamic-gpu-queue-no-checkpoints-7322.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_early docs/evidence/dynamic-gpu-queue-no-checkpoints-7325.csv
python .local/hardware/bench_early_restore.py --seed 7323 --peer 1920 3072 --repeats 4 --lead 400 --warm-control --out outputs/dynamic-gpu-queue-warm-7323.csv
python .local/hardware/bench_early_restore.py --seed 7322 --peer 1920 3072 --repeats 5 --no-idle-slot-save --out outputs/dynamic-gpu-queue-no-idle-7322.csv
python .local/hardware/bench_early_restore.py --seed 7322 --peer 1920 3072 --repeats 5 --no-idle-slot-save --no-context-checkpoints --out outputs/dynamic-gpu-queue-no-checkpoints-7322.csv
python .local/hardware/bench_early_restore.py --seed 7325 --peer 1856 2048 3200 --repeats 4 --no-idle-slot-save --no-context-checkpoints --out outputs/dynamic-gpu-queue-no-checkpoints-7325.csv
python .local/hardware/bench_early_restore.py --header-probe --seed 7329 --peer 3200 3328 --repeats 5 --lead 400 --no-idle-slot-save --no-context-checkpoints --batch-tokens 1024 --ubatch-tokens 512 --out outputs/dynamic-gpu-header-repeat-7329.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_header outputs/dynamic-gpu-header-repeat-7329.csv --log outputs/dynamic-gpu-header-repeat-7329.log --out outputs/dynamic-gpu-header-repeat-7329-evidence.csv
~~~
