# 恢复与同行是否共用同一执行器

独立进程对照改变了窄边界选点的结果。将同行移出目标的服务进程后，预填充末段的典型深恢复损失消失，即使两份模型仍在同一块 GPU 上。独立进程既改变任务等待，也取消了跨请求合批；结果用来限定整体服务机制，不能单独归因于传输和计算重叠。

同一张 RTX 4060 Laptop，两个同配置的 llama.cpp Vulkan 服务都加载、预热，各有两个槽，模型、文件路径和每槽上下文不变。两轮使用相同种子与提示词：同执行器组由目标服务的槽 0 执行同行，第二份模型保持驻留；分执行器组由第二个服务的槽 0 执行同行。目标和恢复文件始终只在第一服务。两轮分别启动，固定顺序先隔离、后共享；轮内随机化浅／深与阶段顺序。

| 触发时同行的阶段 | 共用执行器：深减浅目标中位数 | 独立执行器：深减浅目标中位数 | 同一输入下两种拓扑的深浅效果之差 |
| --- | ---: | ---: | ---: |
| 预填充早期 | -40.39 ms | -44.45 ms | +10.28 ms，2/4 为正 |
| 预填充末段 | +28.66 ms | -60.40 ms | +96.29 ms，4/4 为正 |
| 解码 | +4.62 ms | -46.70 ms | +49.45 ms，4/4 为正 |

独立组在所有 12 个配对里都保留相同的贪心首 token，且深恢复更快。共用组的末段恢复客户端中位数约 603 ms，独立组约 56 ms；两个服务端使用各自的时钟，跨进程的重叠不能从两个日志时戳相减得到。关键对照是相同的恢复路径和同卡并发，在控制面是否必须等同行进程的执行器：移开该边界之后，真实服务结果可改变。GPU 是否同时执行两个进程的命令以及多卡是否同样受益都没有由这组数据直接测定。CUDA 夹具的设备事件对照单独证实了是否允许计算与传输重叠可以翻转合成负载的路径排名。

证据：[同执行器](evidence/dynamic-gpu-phase-7340-shared.csv) · [独立执行器](evidence/dynamic-gpu-phase-7340-isolated.csv) · [CUDA 设备事件](dynamic-gpu-staging-study.md)。

~~~powershell
python .local/hardware/bench_gpu_phase.py --seed 7340 --repeats 4 --phases early late decode --port 18266 --peer-port 18267 --isolated-peer --out outputs/dynamic-gpu-phase-7340-isolated.csv
python .local/hardware/bench_gpu_phase.py --seed 7340 --repeats 4 --phases early late decode --port 18268 --peer-port 18269 --out outputs/dynamic-gpu-phase-7340-shared.csv
python -m scripts.gpu.experiments.study_gpu_phase docs/evidence/dynamic-gpu-phase-7340-shared.csv --executor-control docs/evidence/dynamic-gpu-phase-7340-isolated.csv
~~~
