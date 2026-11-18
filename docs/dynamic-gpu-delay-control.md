# GPU 配对：延迟提交能否重现深恢复的同行收益

2026-11-17，[A1](work-plan.md) 机制对照，接续[同行启动时差实验](dynamic-gpu-peer-progress.md)。待区分的是：深恢复是否必须靠减少目标续算才能让同行更快，还是只推迟目标入批就能产生这类收益。若浅驻留延迟后也能让同行更快，则该收益不需要冷恢复；只有两种方案同时匹配入批时序和同行进度时，才尝试定量拆分两者影响。

## 配对设计

沿用本机 RTX 4060 Laptop、Vulkan、llama.cpp b10923（8a56aedd6）、Qwen3.5-4B Q4_K_M 和两槽连续批处理，目标及同行输入均为 2,560 token，同行已有 512 token 驻留。三种方案为立即提交浅驻留（1,024 token）、等待后提交同一浅驻留、SSD 冷恢复（1,152 token）后提交。每个提示词块交叉 25 / 350 ms 的同行提前量和三方案，随机交错执行；每次启动排除一个预热块。等待时长分别固定为 572 / 249 ms，取独立种子 7201 的[先前测量](evidence/dynamic-gpu-lag-trials.csv)中各档有同行冷恢复的客户端恢复耗时中位数（四舍五入），本轮不重新拟合。

目标响应从选定路径开始计时，包含人为等待或恢复调用；同行响应从同行请求提交计时；共同完工从同行提交计至两者较晚的完成时刻。首次入批间隔使用同一服务端时钟：取两项首次组装 prompt 后首个 decode 日志事件的时间差；这是引擎提交标记，并非 GPU kernel 的起始时间。另标注目标首批时同行槽是否已释放，这是事后诊断。采集器检查实际缓存和续算长度，分析器检查三方案返回同一约束贪心 token、深恢复读量与文件一致、卷读量达到文件大小，以及日志槽位和初始缓存量与客户端记录一致。卷读计数覆盖整卷，不能单独细分具体恢复阶段。

本机采集命令（复测将种子和输出文件名中的 7281 换为 7282）：

~~~powershell
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --total 2560 --peer-total 2560 --seed 7281 --delay-control 25 572 --delay-control 350 249 --repeats 8 --port 18255 --out outputs/dynamic-gpu-delay-7281.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_delay outputs/dynamic-gpu-delay-7281.csv --log outputs/dynamic-gpu-delay-7281.log --out outputs/dynamic-gpu-delay-enriched-7281.csv
~~~

## 结果

两个新种子共 96 条有效测量，每档 16 个三方案配对块。种子 7282 的同一提示词在两档里均发生首 token 分歧（浅驻留和延迟浅驻留都是 ID 44，深恢复是 ID 34）；相应两块保留在归档里。下表先报告每档 15 个 token 一致的块，收益均为相对立即浅驻留的配对均值，正值为更快。

| 同行提前量 | 方案 | 首批间隔中位数 ms | 目标首批时同行仍在槽 | 目标收益 ms | 同行收益 ms | 响应和收益 ms | 共同完工收益 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 ms | 立即浅驻留 | 548.66 | 15/15 | — | — | — | — |
| 25 ms | 延迟浅驻留 | 797.43 | 1/15 | +16.44 | +548.89 | +565.33 | +16.46 |
| 25 ms | SSD 深恢复 | 809.34 | 15/15 | −39.60 | +158.92 | +119.32 | −39.52 |
| 350 ms | 立即浅驻留 | 562.70 | 15/15 | — | — | — | — |
| 350 ms | 延迟浅驻留 | 792.06 | 0/15 | +50.81 | +544.76 | +595.57 | +50.82 |
| 350 ms | SSD 深恢复 | 814.01 | 14/15 | −5.89 | +155.35 | +149.45 | −5.86 |

延迟浅驻留在两档均让 15/15 个同行更快。纳入分歧块后，延迟浅驻留的同行平均收益为 551.76 / 547.38 ms，深恢复为 158.59 / 152.92 ms，主要比较的方向不变。分歧是两档重复使用同一提示词的结果，不是两个独立提示词出现分歧；首 token 一致也不能证明内部状态逐元素一致。

**延迟目标提交本身足以产生同行收益。** 浅前缀、续算长度和模型路径保持不变，只改提交时机，同行仍能更快。然而深恢复与延迟浅驻留的首批间隔即使接近（两个档位逐对差的绝对值中位数是 29.14 / 18.92 ms），同行在目标首批时的状态仍几乎相反。这个对照不能量化深恢复收益中有多少来自晚入批，也不能把差值直接记成 SSD、同步或排队成本。

后续若要拆分贡献，应补一个驻留深边界对照区分深浅计算路径，并在不恢复的方案里按可观测的同行进度校验入批状态。调度规则需在实验前冻结；槽释放和响应完成只能用于事后标注。本轮是同机两槽机制实验，不能推出通用等待阈值或线上吞吐收益。

## 首 token 分歧复核

已补做[同一深边界的驻留与冷恢复对照](dynamic-gpu-same-boundary.md)，记录分数和入批时序。

另用同种子 7282、25 ms 档重跑前两个提示词块，补上无同行及全重算对照。两种采集模式每块均在生成提示词后对六个方案做一次随机排列，因而复用相同的提示词序列。第 0 块六条均返回 ID 82；第 1 块复现分歧：

| 第 1 块 | 无同行 | 有同行 |
| --- | ---: | ---: |
| 全重算 | 34 | 34 |
| 1,024 驻留 | 44 | 44 |
| 1,152 SSD 冷恢复 | 44 | 34 |

这说明分歧也出现在全重算与浅驻留之间，不能归为 SSD 恢复独有的问题；深恢复的首 token 也随同行条件变化。现有读数只核验缓存位置和一个约束贪心 token，不能区分浮点近似、批次计算路径与状态差异。下一步优先比较同一深边界的驻留与冷恢复，并记录两个竞争 token 的分数，以判断是否接近选取边界。

[12 条诊断读数](evidence/dynamic-gpu-token-check-7282.csv)对应命令：

~~~powershell
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --total 2560 --peer-total 2560 --seed 7282 --lag-ms 25 --repeats 2 --port 18255 --out outputs/dynamic-gpu-token-check-7282.csv
~~~

## 数据复算

[96 条逐次证据](evidence/dynamic-gpu-delay-trials.csv)包含校验标志、实测首次入批间隔和同行释放相对时间。从仓库根目录复算，第二条包含有分歧的计时块：

~~~powershell
python -m scripts.gpu.analysis.summarize_dynamic_gpu_delay docs/evidence/dynamic-gpu-delay-trials.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_delay docs/evidence/dynamic-gpu-delay-trials.csv --include-mismatch
~~~
