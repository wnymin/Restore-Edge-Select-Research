# 共享暂存额度的 GPU 物理对照

固定 GPU 显存实际分配，只改变逻辑额度，能够改变同一对候选的胜负；是否翻转还取决于暂存对象何时释放。**只观察准入等待会高估其代价。** 等待落在仍在执行的计算窗口内，可能不增加关键路径；若拖过窗口末尾，晚搬运会直接推迟目标续算。

RTX 4060 Laptop 上用 CUDA Driver API 预先分配固定 128 MiB 设备缓冲区。目标从锁页 RAM 上传 16 或 48 MiB，同行从文件以 Windows 无缓冲读取 48 MiB 到锁页 RAM，再上传 GPU。先提交同行，逻辑共享额度为 64 或 96 MiB；独立额度对照给两边各 64 MiB。另用独占 32 MiB 演示深路径自身不可行。每次真实传输都核对目标缓冲区末字节；在计时区间外、未被复用的同行缓冲区另核对文件末字节。只测了确定性先到顺序，未做主动丢弃或重复搬运。

GPU 先执行约 30 ms 的合成计算；目标浅路径后续执行 20 ms，深路径分别执行 6 或 18 ms。串行模式先等初始计算完成再传输；重叠模式提前提交，设备事件核对实际重叠区间。同行释放条件由事件控制：两条序列分别在初始窗口结束可释放、或再执行约 12 ms 后释放。两种释放规则分别有独立随机顺序的校准与复测，每格各 8 对。以下为复测的目标设备时间中位差（深减浅）：

| 重叠模式，深路径省 14 ms 续算 | 共享 64 MiB | 共享 96 MiB | 独立 64 + 64 MiB |
| --- | ---: | ---: | ---: |
| 早释放 | -11.32 ms | -11.29 ms | -11.16 ms |
| 晚释放 | +2.38 ms | -14.03 ms | -14.00 ms |

早释放工况在 64 MiB 下，深路径实际等待准入约 18.41 ms，却仍 8/8 获胜。晚释放时深路径仍可执行，但在共享 64 MiB 下等待约 26.91 ms，目标 0/8 获胜；放宽至 96 MiB 或改用独立额度时，约 4 ms 的目标上传落进设备计算窗口，均 8/8 获胜。两种额度使用同一物理分配、字节量、程序和消费规则，因此这组是**可行候选之间的排序翻转**。深路径仅省约 2 ms 的独占对照里，串行时深路径慢 0.67 ms，允许重叠时快 2.00 ms，同样是 8/8 的方向变化。设备事件的区间交集用于这个夹具的相对比较，没有替代推理服务内的 GPU 时间线。

用校准集逐格要求深路径在全部配对上获胜，决定复测时选深还是浅。晚释放复测相对固定浅平均快 4.09 ms，160 对无一比对应浅路径慢；能准入就选深平均快 2.31 ms，但有 49/160 对更慢。该规则需要事先知道额度拓扑、释放条件和合成计算设置；它是夹具的机制对照，不是可以直接部署的模型服务策略。这组数据适合给事件模拟加入「同时占用」「最后有效搬运窗口」的反例，不能当成真实模型的 TTFT 参数。

另外一批 7344 直接记录先行计算结束 $T_0$、目标传输完成 $R_b$、目标续算耗时 $C_b$ 和实际完工。夹具的依赖关系给出 $\max(T_0,R_b)+C_b$。对所有 128 个深浅都可执行的配对，这个分解判断的方向和实测相同，深浅差的最大绝对残差为 0.448 ms。这是用**事后实测设备事件**核对关键路径，不是事先预测：真实服务若因选点改变同批代价，$T_0$ 和 $C_b$ 就不能视为对两边都固定。设备事件单独归档：[晚释放关键路径事件](evidence/dynamic-gpu-staging-7344-events.csv)。

证据：[早释放校准](evidence/dynamic-gpu-staging-7336-cal.csv) · [早释放复测](evidence/dynamic-gpu-staging-7337-holdout.csv) · [晚释放校准](evidence/dynamic-gpu-staging-7338-retain.csv) · [晚释放复测](evidence/dynamic-gpu-staging-7339-retain.csv)。

~~~powershell
python .local/hardware/bench_gpu_staging.py --source outputs/gpu-staging-source.bin --seed 7338 --repeats 8 --retain-ms 12 --out outputs/dynamic-gpu-staging-7338-retain.csv
python .local/hardware/bench_gpu_staging.py --source outputs/gpu-staging-source.bin --seed 7339 --repeats 8 --retain-ms 12 --out outputs/dynamic-gpu-staging-7339-retain.csv
python -m scripts.gpu.experiments.study_gpu_staging docs/evidence/dynamic-gpu-staging-7338-retain.csv --holdout docs/evidence/dynamic-gpu-staging-7339-retain.csv
~~~
