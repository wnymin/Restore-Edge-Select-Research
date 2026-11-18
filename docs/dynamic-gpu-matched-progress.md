# GPU 配对：同一恢复边界与同行阶段

接续[同边界首 token 对照](dynamic-gpu-same-boundary.md)。上一轮在 1,152-token 边界观测到：有同行时，显存驻留和 SSD 冷恢复都选 token 34，但候选分数不同。本轮检验延迟驻留到可比同行阶段能否重现冷恢复分数。

沿用 RTX 4060、Vulkan、Qwen3.5-4B Q4_K_M 与双槽配置。同一个目标提示词重复六轮，同行固定提前 25 ms。每轮随机交错：冷恢复、立即深驻留、深驻留多等 300/450/564 ms。564 ms 由上一轮有同行的冷恢复调用中位数 563.61 ms 四舍五入得出；300/450 ms 是事先选定的较早对照。等待和恢复计入目标响应；预热块排除。

同行阶段定义为目标首批前，同一同行任务最近一条 cached n_tokens 日志的值，包含已驻留的 512 token；槽已释放则记为总长 2,560。该值是调度器记录的前缀位置，用于事后区分阶段，不是 GPU kernel 计时或在线选点时可知的未来状态。本轮全部 30 条实测中，目标首批时同行仍在槽内。

| 路径 | 同行前缀 | 目标首批间隔中位数 ms | 对数概率差 $\log P(34)-\log P(44)$ | 目标响应均值 ms | 同行响应均值 ms | 共同完工均值 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 立即驻留 | 2,044 | 603.87 | +0.005763 | 1,511.59 | 1,460.72 | 1,537.40 |
| 驻留等 300 ms | 2,044 | 601.97 | +0.005763 | 1,504.45 | 1,457.34 | 1,530.41 |
| 驻留等 450 ms | 2,556 | 821.34 | +0.000826 | 1,498.75 | 1,245.29 | 1,524.56 |
| 驻留等 564 ms | 2,556 | 815.04 | +0.000826 | 1,510.94 | 1,255.73 | 1,536.68 |
| SSD 冷恢复 | 2,556 | 873.13 | +0.000826 | 1,560.25 | 1,304.16 | 1,586.06 |

**对齐同行阶段后，深驻留重现了冷恢复的两个候选分数。** 两档较晚等待各有六对与冷恢复阶段相同，两个原始分数分别完全相等；较早等待和立即驻留也彼此相等。所有运行都选 token 34。观测到的分数差不要求 SSD 路径，也不要求同行已经结束。12 对共享同一个提示词、每轮共用冷恢复参照，不能视为 12 个独立请求，也不能证明其他提示词或路径逐元素等价。

**对齐阶段后，性能差仍在。** 相比两条阶段匹配的驻留路径，冷恢复仍让同行平均慢 58.87 / 48.43 ms，目标和共同完工也平均更晚。阶段匹配没有配平其他等待和恢复开销，无法将差值拆为 SSD 读盘、同步和设备复制成本。延迟深驻留要求深检查点已在显存，仅为机制对照；在原问题只有浅驻留和 SSD 深检查点时，深驻留不构成可用候选。

对该分歧提示词的追查给出了可验证的阶段对照。后续已完成[A2 选点前信号的留出](dynamic-gpu-a2-signals.md)、[A3 实时选点](dynamic-gpu-a3-live.md)及[同槽提前恢复](dynamic-gpu-early-restore-plan.md)。实际首批阶段只作诊断标签，没有进入选点规则。

[30 条逐次测量](evidence/dynamic-gpu-score-wait-7282.csv)和复现命令：

~~~powershell
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --total 2560 --peer-total 2560 --seed 7282 --score-prompt-rep 1 --score-wait-ms 300 450 564 --lag-ms 25 --repeats 6 --port 18255 --out outputs/dynamic-gpu-score-wait-7282.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_scores outputs/dynamic-gpu-score-wait-7282.csv --log outputs/dynamic-gpu-score-wait-7282.log --out outputs/dynamic-gpu-score-wait-evidence.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_scores docs/evidence/dynamic-gpu-score-wait-7282.csv
~~~
