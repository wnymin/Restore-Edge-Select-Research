# GPU 配对：同一深边界的驻留与冷恢复

后续已完成[延迟深驻留与同行阶段匹配](dynamic-gpu-matched-progress.md)。

2026-11-17，接续 [A1 的首 token 分歧](dynamic-gpu-delay-control.md#首-token-分歧复核)。本轮追问两个问题：换成同样长的显存驻留前缀，分歧还会不会出现？发生分歧时，两个竞争 token 的分数相差多大？

沿用 RTX 4060 Laptop、Qwen3.5-4B Q4_K_M、llama.cpp b10923 的 Vulkan 双槽配置。重放种子 7282 的第 1 个实测提示词块；目标与同行输入均为 2,560 token。同行无／有（提前 25 ms，已驻留 512 token）与四种路径交叉：全重算、1,024 驻留、1,152 驻留、1,152 SSD 冷恢复。八个组合随机排序，重复四轮，丢弃独立预热块；四轮都是同一个提示词，衡量复现性。驻留路径在同行提交前用相同前缀填充槽；冷恢复在目标计时开始后读取事先保存的 1,152-token 状态，延后续算提交。

保留原先的约束贪心采样；请求服务端返回前 256 个采样前分数。下表用两个竞争 token 的原始对数概率差 $\log P(34)-\log P(44)$，正值表示 34 分数较高。这些分数未按输出语法重新归一化；它们不是约束采样后的概率，也不度量回答质量。所有行均核对缓存长度和续算长度，冷恢复另核对恢复字节、状态文件和整卷读量。日志核对目标和同行的首次批次提交及同行槽释放时间。

| 路径 | 无同行：首 token / 分数差 | 同行提前 25 ms：首 token / 分数差 | 有同行：首批间隔中位数 ms |
| --- | ---: | ---: | ---: |
| 全重算 | 34 / +0.012066 | 34 / +0.016030 | 535.27 |
| 1,024 驻留 | 44 / −0.003604 | 44 / −0.003654 | 535.98 |
| 1,152 驻留 | 44 / −0.004216 | 34 / +0.005763 | 534.72 |
| 1,152 SSD 冷恢复 | 44 / −0.004216 | 34 / +0.000826 | 784.87 |

**无需读 SSD，也能复现深边界的首 token 翻转。** 深驻留和深冷恢复在无同行时均选 44，有同行时均选 34；八个组合各自四轮的两个分数均相同。对这一个提示词，两个竞争 token 的原始对数概率差绝对值仅 0.000826–0.016030。计算路径改变时，较小的分数变化即可交换两者的顺序。这支持把本例视为近分候选对计算路径敏感，不支持从首 token 分歧判断 SSD 恢复状态有误。

同一深边界，单跑时驻留和冷恢复的两个分数四对均完全相同；有同行时四对均不同，尽管所选 token 相同。两条有同行路径的目标首批时间中位数相差约 250 ms，且目标首批时同行均未释放槽。因此尚不能单独归因并发下的分数差。后续应在同一深边界加入受控延迟的驻留路径，比较实际首批时刻和同行所处阶段，再检验分数差是否随之改变。该用例可作为分歧诊断，不提供多提示词的精确一致性保证或性能结论。

[32 条测量](evidence/dynamic-gpu-same-boundary-7282.csv)包含 prompt 校验和、分数、排序和入批诊断。先运行本机采集和转换命令；最后一条可直接复算归档，无需 GPU。原始前 256 候选与服务端日志留在本机 outputs。

~~~powershell
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --total 2560 --peer-total 2560 --seed 7282 --score-prompt-rep 1 --lag-ms 25 --repeats 4 --port 18255 --out outputs/dynamic-gpu-same-boundary-7282.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_scores outputs/dynamic-gpu-same-boundary-7282.csv --log outputs/dynamic-gpu-same-boundary-7282.log --out outputs/dynamic-gpu-same-boundary-evidence.csv
python -m scripts.gpu.analysis.summarize_dynamic_gpu_scores docs/evidence/dynamic-gpu-same-boundary-7282.csv
~~~
