# GPU A2：选点前的同行信号

目标是在真实双槽服务检验：已知浅前缀驻留、深状态在 SSD，只有提交时间和同行工作量等可见信号时，能否比单跑成本更有把握地判断何时改选。接续[同边界阶段对照](dynamic-gpu-matched-progress.md)，但此次的两条实际可用路径为 1,024-token 浅驻留与 1,152-token SSD 冷恢复。

沿用 RTX 4060、Vulkan、Qwen3.5-4B Q4_K_M、llama.cpp b10923 和两个 4,096-token 槽。目标输入 2,560 token，同行总输入按配置改变，一律已有 512 token 前缀。每个随机提示词块内部交叉所有工况和两条路径，包含无同行对照。每次启动丢弃一个预热块。长短同行是同一随机同行序列的前缀；目标实际缓存长度、续算长度和卷读量逐条核验。

| 划分 | 种子 / 实测提示词块 | 同行总长 token | 同行提前量 ms |
| --- | --- | --- | --- |
| 校准 | 7311 / 5 | 1,536、3,584 | 25、350、1,050 |
| 留出 | 7312 / 5 | 2,048、3,072 | 100、550 |

只比较决策前信息。无同行时比较校准单跑成本；有同行时增加客户端可见的“同行响应尚未返回”标志、同行提交到当前选点的时间，或用已知总长减去预先缓存的 512 token 得到初始待处理量，以它归一化提交时长。该比值是粗代理，不把剩余工作量当成实测。对同一配对，两种选择共用浅驻留一侧的选点前快照；另外核对两侧状态是否一致。服务端首批的同行日志进度只用于评价，距末端不超过四 token 或槽已释放标为“接近结束”。没有计算进行中的状态通过客户端接口泄漏到特征。

策略按校准中各区间的平均收益判断是否改选。年龄和归一化年龄的阈值只在小组预定网格中用校准集选择，比较目标响应、两项响应之和、共同完工三个独立目标。规则冻结后才启动留出采集。阶段预测在同一网格校准，用 Brier 分数（概率预测与二值标签的均方差，越低越好）和常量基线比较。选点收益均是独立重复的配对估计，尚未在同一次服务中实时调用策略；计时也不包含 A3 的实际决策开销。

## 留出结果

[校准](evidence/dynamic-gpu-a2-calibration.csv)有 35 对，[留出](evidence/dynamic-gpu-a2-holdout.csv)有 25 对；各配对的首 token 和选点前同行是否待完成的标志均一致。[冻结规则](evidence/dynamic-gpu-a2-rules.json)包含校准输入哈希及所有阈值。留出复算得到：

| 优化两项响应之和的方法 | 改选数 | 方向正确 | 平均绝对误差 ms | 目标收益 ms | 同行收益 ms | 响应和收益 ms | 共同完工收益 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 只看单跑成本，始终浅驻留 | 0 | 8/25 | 172.54 | 0 | 0 | 0 | 0 |
| 固定深恢复 | 25 | — | — | −11.54 | +129.64 | +118.09 | −11.62 |
| 同行待完成标志 | 20 | 22/25 | 105.72 | −7.16 | +129.64 | +122.48 | −7.23 |
| 标志及提交时长 | 10 | 18/25 | 106.76 | −9.72 | +67.12 | +57.39 | −9.72 |
| 标志及按初始工作量归一化的时长 | 20 | 22/25 | 103.85 | −7.16 | +129.64 | +122.48 | −7.23 |

正值为快于固定浅驻留，均值包含单跑对照。待完成标志在五个提示词块都改善了响应和，块均值收益 97.92–156.98 ms；但最慢的目标请求比基线慢 59.28 ms，离每对事后最优仍有 7.33 ms 的平均响应和差距。事后最优只在本实验两条实测路径中选。按提交时长筛选在校准略优于仅用标志（55.79 对 54.94 ms），在新工况反而丢掉一半以上的响应和收益。归一化时长稍降数值预测误差，但没有改变一次实际决策。

**只看目标或共同完工，所有规则都保留浅驻留。** 留出仍有八对深恢复更快，始终保留浅驻留相距对应的事后配对最优约 10.88 ms；这批特征没有给出可靠的改选方式。以同行为主的总响应收益不能解释为目标或整组加速。

实际同行阶段的结果也分路径：用归一化时长预测目标首批时同行是否接近结束，留出里浅驻留的 Brier 从 0.2678 降至 0.1700；冷恢复则从 0.0278 升至 0.0833。说明先前找到的阶段可用于解释机制，而该阶段在不同路径上的可预测性不同。没有证据支持把复杂特征默认带入选点。

进入 A3 的实时对照时，冻结候选为**仅优化响应和时用同行待完成标志**，主要基线为浅驻留和固定深恢复；在此短边界配置里，独占成本规则与浅驻留选择相同。目标与共同完工默认沿用浅驻留。实时对照应让所有策略从同一已驻留浅状态开始，量决策实际耗时，并在第三个种子和新工况上报告目标代价。若要改善目标或共同完工，需单独检验减少恢复关键路径开销的系统方案，不能靠这条规则推断。

后续[实时分派对照](dynamic-gpu-a3-live.md)已完成。

## 复算

从仓库根目录运行。采集使用本机 Windows 包装器，服务端详细日志和包含预热的输入保留在 outputs。归档副本含观测到的阶段、配对状态及 token 一致标志，不需要 GPU 即可复算。

~~~powershell
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --seed 7311 --peer-case 1536 25 --peer-case 1536 350 --peer-case 1536 1050 --peer-case 3584 25 --peer-case 3584 350 --peer-case 3584 1050 --repeats 5 --port 18255 --out outputs/dynamic-gpu-a2-calibration.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_signals calibrate outputs/dynamic-gpu-a2-calibration.csv --log outputs/dynamic-gpu-a2-calibration.log --out outputs/dynamic-gpu-a2-calibration-evidence.csv --rules outputs/dynamic-gpu-a2-rules-copy.json
python outputs/run-slot-retest.py --paired-storage --resident-comparison --boundaries 1024 1152 --seed 7312 --peer-case 2048 100 --peer-case 2048 550 --peer-case 3072 100 --peer-case 3072 550 --repeats 5 --port 18255 --out outputs/dynamic-gpu-a2-holdout.csv
python -m scripts.gpu.experiments.study_dynamic_gpu_signals evaluate docs/evidence/dynamic-gpu-a2-holdout.csv --rules docs/evidence/dynamic-gpu-a2-rules.json
~~~
