# 固定恢复边界的阶段干预

在 RTX 4060 Laptop 上，固定同一对检查点（1,024 驻留 → 1,152 文件恢复），只改变同行的触发阶段，深恢复相对浅驻留的目标延迟就会变。换随机提示词、同行从 3,200 改到 3,328 token、生成从 32 改到 40 token 后，预填充早期和末段的差异仍能在中位数上复现，但早期不再逐对获胜。**阶段是有用的干预变量；现有阶段标签不构成窄边界的稳健选点规则。** 宽边界的收益明显高于窄边界，因此固定边界的优先级仍是需要击败的基线。

环境与原双槽实验相同（Vulkan、llama.cpp b10923、Qwen3.5-4B Q4_K_M）；设逻辑批量 1,024、物理批量 512，关闭自动空闲槽保存和上下文检查点。固定目标为 2,560 token。各块随机化阶段顺序与浅、深、同深度驻留的运行顺序。深边界用同一块刚保存的文件，不要求冷读；实际读取量不能仅由文件大小推断。触发条件是空闲、同行预填充第一个批次入口（已缓存 512）、最后一个初始预填充批次入口（已缓存 2,560），或服务端记录第 8 个生成 token；脚本实时观察服务器日志再发起目标路径。测得的首 token 接收代理是从触发到单 token HTTP 返回，包含选择后的恢复。同行完成是其流式提交到流结束；最长间隔基于客户端实际接收的 token，另有服务器采样时间可复算。共同完工从同一触发点计起。

| 阶段 | 校准目标差值，4 对 | 留出目标差值，6 对 | 留出深恢复获胜 | 留出同行完成差值 | 留出最长间隔差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 空闲 | +24.71 ms | +14.20 ms | 1/6 | — | — |
| 预填充早期 | -26.03 ms | -13.03 ms | 4/6 | -20.56 ms | +2.75 ms |
| 预填充末段 | +61.37 ms | +62.47 ms | 1/6 | +4.93 ms | -5.27 ms |
| 解码 | +16.21 ms | +26.61 ms | 0/6 | -19.83 ms | -1.87 ms |

差值为深减浅的配对中位数。用每块的「末段深浅差减早期深浅差」检验交互，留出中位数为 +87.77 ms，6 块有 5 块为正。窄边界的 40 个校准和留出块中，目标的单个贪心 token 都相同；这不代表长输出一致。同行在触发时的绝对运行年龄因阶段而不同，实验识别的是**改变释放阶段这一整体干预**，没有把其中的批次组成、队列位置和已提交后端工作分别随机化。

校准早期的深恢复客户端历时中位数约 485 ms，目标整体却比浅路径快约 26 ms。服务器日志显示，浅路径的目标也要排队，恢复前半段还允许同行继续推进。末段则约有 529 ms 排在恢复真正派发之前；随后，目标加入的是含同行解码的批次。等深度驻留控制有助于说明光看重算节省不够：留出早期深恢复比等深度驻留慢约 10 ms，末段慢约 116 ms。两个差值包含时间线变化，不是固定的 I/O 成本。

新种子 7334 单独做了等深度延时对照：先把深边界置于驻留状态，再按照校准的三个阶段恢复中位数等待（485 / 585 / 95 ms）后提交。深恢复与驻留延时组在每个阶段的首批同行类型完全一致（早期预填充、末段和解码为同行解码）；但首批时间仍有约 30–53 ms 的阶段中位差。相对驻留延时，真实深恢复在早期和末段目标分别慢约 65、58 ms，解码阶段反而快约 34 ms。因此固定睡眠只控制了提交时刻的近似量级，不能当成等价恢复或纯 I/O 测量。该批深减浅早期中位数也转为 +8.37 ms，末段仍为 +59.77 ms，进一步限定了「早期选深」的适用性。

保护规则的条件为：同边界同阶段至少四个校准块、每块目标快于浅路径 20 ms 以上，且同行完成和最长间隔各不比浅路径多 50 ms。窄边界没有任何阶段通过，留出中回退到浅路径；固定深与同行待完成时选深，分别使目标平均慢 21.63 和 18.27 ms，均有 4/24 块超过同行阈值。回退仍错过部分有利配对：在同行符合上述阈值的留出反事实中，相对逐块最优的平均损失为 4.75 ms。样本不足以给出覆盖保证或 P95。

宽边界 512→1,536 的四个校准阶段全部通过。冻结规则在 20 个随机排序留出块中实际执行了深恢复，目标相对浅路径平均缩短 411.43 ms，没有观察到同行超过阈值，和另一组固定深对照只差约 3 ms。这证明了宽边界作为正对照的稳定性；它没有证明增加阶段信号的收益，因为规则对两类边界可以简化为「宽始终深、窄始终浅」。两种边界合并的离线复算仅是这个配比下的结果。

中间边界 1,024→1,280 提供了更有区分力的对照。冻结规则只在空闲和预填充早期选择深边界。24 个留出块里，固定深平均比浅快 43.68 ms，有 1 块同行完成超过 50 ms；规则回放选深 12 次，平均快 33.71 ms，未记录超限。但另行随机运行冻结规则，实际平均快 35.69 ms，出现两次同行完成超限：一次是早期选深（+151.29 ms），另一次是解码回退浅（+102.42 ms）。后者是同动作之间的运行差，说明不能把实时的每次超限都归给策略改选，也不能拿回放的零超限作为保证。结果支持阶段能用于筛选实验工况，尚不支持用这一标签提供逐请求保护。相对于直接选深，阶段规则也让出了约 8–10 ms 的平均目标收益。

第三槽控制在相同种子下让额外请求在目标放行前持续解码。对窄边界，四个配对的深减浅目标中位差在同行解码时从 +10.17 ms 变为 +69.78 ms，同行末段从 +13.52 ms 变为 +47.47 ms。第三槽下批次边界因此移动，末段触发使用第一个剩余不超过逻辑批量的预填充批次；不能解释为只修改一个批次内的 token 数。两组在同一个早期块都发生贪心首 token 不同，已经保留在原始数据中，所以这些配对只提供时序证据。

证据：[窄边界校准](../../evidence/dynamic-gpu-phase-7330-cal.csv) · [窄边界留出](../../evidence/dynamic-gpu-phase-7331-holdout.csv) · [宽边界校准](../../evidence/dynamic-gpu-phase-7332-cal.csv) · [宽边界留出](../../evidence/dynamic-gpu-phase-7333-holdout.csv) · [冻结规则和输入摘要](../../evidence/dynamic-gpu-phase-guard.json) · [延时对照](../../evidence/dynamic-gpu-phase-7334-delay.csv)。每行含服务器任务号、首批中同行的预填充或解码、恢复排队及文件元数据时刻。原始详细日志保留于本机输出；公开证据不保存日志内的完整提示词。

新增证据：[中间边界校准](../../evidence/dynamic-gpu-phase-7342-middle-cal.csv) · [冻结规则](../../evidence/dynamic-gpu-phase-middle-guard.json) · [中间边界实时留出](../../evidence/dynamic-gpu-phase-7343-middle-holdout.csv) · [双槽对照](../../evidence/dynamic-gpu-phase-7341-base.csv) · [第三槽解码](../../evidence/dynamic-gpu-phase-7341-third.csv)。

~~~powershell
python .local/hardware/bench_gpu_phase.py --seed 7330 --repeats 4 --resident --out outputs/dynamic-gpu-phase-7330-cal.csv
python .local/hardware/bench_gpu_phase.py --seed 7331 --peer 3328 --decode-tokens 40 --repeats 6 --resident --out outputs/dynamic-gpu-phase-7331-holdout.csv
python -m scripts.gpu.analysis.enrich_gpu_phase outputs/dynamic-gpu-phase-7330-cal.csv --out outputs/dynamic-gpu-phase-7330-cal-enriched.csv
python .local/hardware/bench_gpu_phase.py --seed 7332 --boundaries 512 1536 --repeats 4 --out outputs/dynamic-gpu-phase-7332-cal.csv
python .local/hardware/bench_gpu_phase.py --seed 7333 --boundaries 512 1536 --peer 3328 --decode-tokens 40 --repeats 5 --guard docs/evidence/dynamic-gpu-phase-guard.json --out outputs/dynamic-gpu-phase-7333-holdout.csv
python .local/hardware/bench_gpu_phase.py --seed 7334 --repeats 4 --phases early late decode --delay early 485 --delay late 585 --delay decode 95 --out outputs/dynamic-gpu-phase-7334-delay.csv
python -m scripts.gpu.experiments.study_gpu_phase docs/evidence/dynamic-gpu-phase-7330-cal.csv docs/evidence/dynamic-gpu-phase-7332-cal.csv --frozen docs/evidence/dynamic-gpu-phase-guard.json --holdout docs/evidence/dynamic-gpu-phase-7331-holdout.csv docs/evidence/dynamic-gpu-phase-7333-holdout.csv
~~~
