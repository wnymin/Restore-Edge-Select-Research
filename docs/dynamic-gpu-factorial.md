# 提前恢复为什么让同行变慢？

[逻辑批量对照](dynamic-gpu-batch.md)中，缩小批量能让提前恢复更早就绪，却伴随明显的同行延迟。本轮用四路配对拆开两种操作。结果是：**只做恢复使同行平均慢约 52–54 ms；当恢复使目标与同行进入同一个预填充批次时，两项操作合计使同行慢约 142–171 ms。** 两种同负载下都看到了这个交互。

## 对照设置

RTX 4060 Laptop、Vulkan、Qwen3.5-4B Q4_K_M、双槽，逻辑批量 1,024、物理微批量 512。关闭空闲槽保存和自动检查点；保留同槽的显式保存和恢复。每条测量开始前，同行槽驻留 512 token，目标槽驻留同一个 1,152 token 深边界；四路都做一次相同的准备保存。目标若提交，总输入 2,560 token，且必须复用 1,152 token。恢复若执行，于同行提交约 150 ms 后发出，从新生成的冷文件恢复到同一深边界。目标若执行，最早在同行提交后 550 ms 发出，组合路另等待恢复结果；实测所有恢复都在 550 ms 前完成，未推迟目标客户端的释放。

四路分别为同行独跑、同行加恢复、同行加目标、同行加恢复和目标。同一块的提示词相同，四路顺序随机。种子 7326 测 3,200 token、8 块；种子 7327 用未测过的 3,328 token 工况留出、6 块。各组另有一个预热块，不计结果。两组共 56 条测量，块内同行首 token 一致，提交目标的两路目标首 token 也一致。用服务端任务 ID 连接收件、解码、槽释放、冷恢复回调与批内位置；冷读量由客户端卷计数核对。

## 各项延迟与交互

数值是同块配对平均差，正值为多等了多久。交互是组合路相对同行独跑的延迟减去两项单独操作的延迟之和。同行响应从客户端提交到首 token；服务端结果另用收件到槽释放核对。

| 同行输入 token | 只恢复 ms | 只启动目标 ms | 恢复并启动目标 ms | 交互 ms |
| ---: | ---: | ---: | ---: | ---: |
| 3,200 | +53.56 | +0.46 | +171.48 | +117.47 |
| 3,328 | +52.03 | −3.17 | +141.72 | +92.86 |

只恢复或有交互时，两组的同行延迟在各自全部块中都是正值。服务端计时的对应结果为：只恢复 +54.21 / +52.71 ms，交互 +113.91 / +84.63 ms。只启动目标的服务端均值只差 +0.56 / +0.35 ms；该路径不能用来估计同批执行时的争用代价，因为目标虽按时提交，每次都在同行槽释放后才进入首批。

| 同行输入 token | 路径 | 目标首批距同行收件中位数 ms | 目标首批到同行释放中位数 ms | 首批中有同行预填充 |
| ---: | --- | ---: | ---: | ---: |
| 3,200 | 只启动目标 | 959.33 | −0.32 | 0/8 |
| 3,200 | 恢复并启动目标 | 584.34 | +542.79 | 8/8 |
| 3,328 | 只启动目标 | 998.80 | −0.36 | 0/6 |
| 3,328 | 恢复并启动目标 | 586.49 | +548.53 | 6/6 |

只恢复时，在同行的第一个和第二个逻辑批次之间插入了约 223–224 ms 的回调，但同行最终只慢约 52–54 ms。两项一起做时，第二批的位置标记与只恢复相比只差约 1 ms，恢复回调也近似相同；显著的额外延迟出现在后续阶段。此时两个槽的预填充位置更新属于同一次服务端解码：恢复改变了同行后续批次的时间，使原本等到同行完成才处理的目标得以在同行仍活跃时入批。组合路中，目标虽先入批，却比仅启动目标平均晚 **37.48 / 61.52 ms** 返回首 token，各组所有配对同方向。提前就绪和更早入批都不足以判断最终收益。

## 不做恢复，同批是否仍有代价？

新种子 7328，两个同行长度各做六个配对块。在原四路之外，加入「不恢复、目标在 350 ms 提交」；350 ms 由前一轮的批次日志选定，使其有机会在同行最后一轮预填充前入队。这是改变提交时间的机制控制，不是 550 ms 释放约束下可以选用的在线策略。60 条测量通过块内首 token 和实际缓存核对。包括组合路在内，两种同批路径在所有 12 对均对齐到同行缓存位置 2,560 token；测量该位置到同行释放，也避免了上游排队和客户端线程干扰。

| 同行输入 token | 不做恢复，只让目标同批：同行最后预填充到释放增加 ms | 已经恢复，再让目标同批：同一阶段增加 ms | 两个效应之差 ms |
| ---: | ---: | ---: | ---: |
| 3,200 | +114.96 | +113.59 | −1.37 |
| 3,328 | +83.24 | +83.19 | −0.05 |

各同行长度的六对全部出现同批代价，两个效应之差在逐块比较中均落在 −3.43 至 +3.13 ms。本轮按 550 ms 提交的普通目标路有一次也恰好入了同批，同行服务端耗时增加 117.45 ms；它保留在数据中，没有把普通目标提交等同于固定不共批。控制了同行位置后，不执行恢复也能复现组合路中约 83–115 ms 的后段额外延迟。可将这部分解释为该位置上加入目标预填充的代价；没有证据要求用额外的 SSD 成本解释。

这是对现有执行路径的因果对照，基线事先生成了同一深边界，用于保持目标的续算工作量相同。它不表示服务能免费持有这个边界，也不直接估计浅驻留和深恢复的收益差。设备同步和上传各自耗时仍不从日志区间反推。

[首轮 32 条](evidence/dynamic-gpu-factorial-7326.csv) · [留出 24 条](evidence/dynamic-gpu-factorial-7327.csv) · [同批对照 60 条](evidence/dynamic-gpu-factorial-phase-7328.csv)

~~~powershell
python -m scripts.gpu.experiments.study_dynamic_gpu_factorial docs/evidence/dynamic-gpu-factorial-7326.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_factorial docs/evidence/dynamic-gpu-factorial-7327.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_factorial docs/evidence/dynamic-gpu-factorial-phase-7328.csv
python .local/hardware/bench_early_restore.py --factorial --seed 7326 --peer 3200 --repeats 8 --lead 400 --no-idle-slot-save --no-context-checkpoints --batch-tokens 1024 --ubatch-tokens 512 --out outputs/dynamic-gpu-factorial-repeat-7326.csv
python .local/hardware/bench_early_restore.py --factorial --seed 7327 --peer 3328 --repeats 6 --lead 400 --no-idle-slot-save --no-context-checkpoints --batch-tokens 1024 --ubatch-tokens 512 --out outputs/dynamic-gpu-factorial-repeat-7327.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_factorial outputs/dynamic-gpu-factorial-repeat-7326.csv --log outputs/dynamic-gpu-factorial-repeat-7326.log --out outputs/dynamic-gpu-factorial-repeat-7326-evidence.csv
python .local/hardware/bench_early_restore.py --factorial --phase-target-ms 350 --seed 7328 --peer 3200 3328 --repeats 6 --lead 400 --no-idle-slot-save --no-context-checkpoints --batch-tokens 1024 --ubatch-tokens 512 --out outputs/dynamic-gpu-factorial-phase-repeat-7328.csv
~~~
