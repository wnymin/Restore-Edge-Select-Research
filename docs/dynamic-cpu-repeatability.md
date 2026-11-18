# CPU 选点计时的重复性与兼容边界

本轮以已经提交到仓库的 `f791ad7` 为基线，复查 `entry-fast` 的时间收益和回退条件。在线选点和执行器代码沿用该提交。此前每配置的计时为三次，本轮使用十二次平衡顺序的重复，并保存逐轮配对差。

## 输入和比较方式

百炼三份轨迹各取第 84、85 窗。第 84 窗沿用至多两个非零候选，第 85 窗保留所有历史匹配的检查点。交叉 SSD 放置加原到达间隔、混合放置加四分之一间隔，两档预取额度（128、384 MiB），仅 SSD 或 RAM/SSD 同时受压，以及 1、12、20、32 倍成本，共 192 个配置。门控参数没有参与本次比较，三种策略为 `local-cost`、`entry` 和 `entry-fast`。

每个配置先执行零开销回放，逐项核对两种 `entry` 的边界、完成时间、等待、传输与容量结果。每种策略再预热两遍。计时阶段和注入决策时间阶段各执行十二轮，两轮完整的六种执行顺序组成一个实验，各策略在第一、第二、第三位置上分别出现四次。相同轮次的策略差作配对，同配置内取配对差的中位数，再对配置等权平均。实际时长注入后，决策等待会推进模拟时钟并影响缓存和排队，故另保存 P95 和最差单请求差。

不同时刻运行的 CPU 时间可能受系统负载和频率影响。重复轮次只描述相同请求及配置的计时波动。按轨迹为一组重采样，另给每档净改善的分位区间，每个汇总档只有三组轨迹。这些区间适合检查本次结论对三条轨迹的依赖程度，样本范围限于本次构造和 CPU 模拟。

## 两次运行的结果

两次运行使用同一批 192 个配置，执行顺序种子分别为 20261118 和 20261119。每次每配置在固定时间线上计时十二轮，再注入实测决策时长十二轮。两次都先核对了零开销逐请求结果，未发现差异。加密候选最多有 7 个边界。

| 统计量 | 第一遍 | 换序复测 |
| --- | ---: | ---: |
| 原 `entry` 的选点总耗时 | 4.623 ms | 4.649 ms |
| `entry-fast` 的选点总耗时 | 3.992 ms | 4.029 ms |
| 选点总耗时下降 | 13.65% | 13.33% |
| 配对中位数显示选点更快的配置 | 192 / 192 | 192 / 192 |
| 注入计时后的平均响应改善 | 0.0532 ms | 0.0529 ms |
| 平均响应改善大于零的配置 | 165 / 192 | 160 / 192 |
| P95 响应改善的配置等权均值 | 0.0918 ms | 0.0926 ms |
| 全部重复轮次中最大的单请求额外等待 | 1.366 ms | 1.743 ms |

选点总耗时表示每个配置中所有请求的选点时间之和，表中先取配置内十二轮中位数，再对配置等权平均。平均响应和 P95 的变化均先在每个重复轮次相对原 `entry` 配对，再取中位数与配置平均。百分比由表中的两种总耗时计算，不能作为端到端响应降幅。

普通候选组两遍分别减少 14.09%、13.91% 的选点耗时，加密候选组减少 13.29%、12.86%。在 144 个受压配置中，两遍的响应净改善为 0.0661、0.0657 ms。1 倍成本组的部分轨迹净差跨过零，受压各组的三条轨迹均值范围在这两遍内大于零。各档只含三条轨迹，重采样区间不能外推为生产分布上的保证。

重复运行支持这项实现降低了本机 CPU 选点成本。注入后仍有部分配置或请求变慢，反映了计时波动和闭环排队的影响。当前证据覆盖固定成本下的事件模拟以及不满足条件时的回退对照，硬件服务端的恢复调用需要另行计时。

第一遍的[逐轮配对差](evidence/dynamic-cpu-repeatability-paired.csv)、[逐配置结果](evidence/dynamic-cpu-repeatability-cases.csv)、[分组汇总](evidence/dynamic-cpu-repeatability-summary.csv)及[输入版本](evidence/dynamic-cpu-repeatability.json)可复算上述数据。换序复测对应[逐轮配对差](evidence/dynamic-cpu-repeatability-recheck-paired.csv)、[逐配置结果](evidence/dynamic-cpu-repeatability-recheck-cases.csv)、[分组汇总](evidence/dynamic-cpu-repeatability-recheck-summary.csv)及[输入版本](evidence/dynamic-cpu-repeatability-recheck.json)。

## 整合版本复核

新增搬运下界与离线状态接口后，辅助下界函数改为仅在启用实验模式时构造，避免普通 `entry` 路径为未使用的对照付出初始化成本。对完整整合版本用种子 20261121 重跑同一批 192 个配置，每阶段仍为十二轮平衡顺序，未改变输入或筛选配置。

| 统计量 | 整合版本 |
| --- | ---: |
| 原 `entry` 选点总耗时的配置均值 | 7.218 ms |
| `entry-fast` 选点总耗时的配置均值 | 6.079 ms |
| 平均选点总耗时下降 | 15.78% |
| 配对中位数显示选点更快的配置 | 182 / 192 |
| 注入后的平均响应改善 | 0.0847 ms |
| 平均响应改善大于零的配置 | 159 / 192 |
| P95 改善的配置等权均值 | 0.1550 ms |
| 全部重复轮次中最大的单请求额外等待 | 22.993 ms |

这遍的绝对耗时与极端损失均大于前两遍；三遍之间同时存在实现版本和运行时刻的变化，不能把差异全部归因于本次代码调整。结果仍支持平均选点开销下降，但逐配置和逐请求收益并不一致，尤其不能从平均的亚毫秒改善推出单请求延迟保证。完整保留[逐轮配对差](evidence/dynamic-cpu-repeatability-integrated-paired.csv)、[逐配置结果](evidence/dynamic-cpu-repeatability-integrated-cases.csv)、[分组结果](evidence/dynamic-cpu-repeatability-integrated-summary.csv)和[输入版本](evidence/dynamic-cpu-repeatability-integrated.json)。

## 兼容性对照

构造对照交叉共享前缀、初始驻留 KV、计算与搬运并发降速、三种容量释放方式和五档预取额度。用固定种子生成 B、C 的到达时刻。分别检查已知固定成本、估计偏低、估计偏高、对象实际传输时长不同、时变传输速率、主机暂存缓存以及双传输通道，共 840 个配置。每个配置都比较原 `entry`、`entry-fast` 和 `entry-progress`。

固定成本下要求执行结果逐项相同，搜索计数可不同。其他场景要求整个输出字典相同，从而检查优化别名实际执行原有路径。两种种子各有 840 项通过，未发现需要改动执行器的情况。[第一遍兼容性配置](evidence/dynamic-cpu-repeatability-compatibility.csv) · [换序复测兼容性配置](evidence/dynamic-cpu-repeatability-recheck-compatibility.csv)

整合版本使用种子 20261121 的 840 项也全部通过，见[整合兼容性配置](evidence/dynamic-cpu-repeatability-integrated-compatibility.csv)。

## 复现

以下命令从仓库根目录运行，Python 3.12 或 3.13。初次运行需要按数据说明准备百炼轨迹。

~~~sh
python scripts/check_fixture.py
python -m scripts.cpu.dynamic.study_entry_repeatability --repeats 12 --seed 20261118 --out outputs/entry-repeatability/entry.csv
python -m scripts.cpu.dynamic.study_entry_repeatability --repeats 12 --seed 20261119 --out outputs/entry-repeatability/recheck.csv
python -m scripts.cpu.dynamic.study_entry_repeatability --repeats 12 --seed 20261121 --out outputs/entry-repeatability/integrated.csv
~~~

原始轨迹放在其他位置时追加 `--trace-dir`。完整逐策略计时输出保存在本地 `outputs/`，用于报告的逐轮配对差、逐配置中位数、分组汇总和输入哈希随仓库提交。CPU 模拟响应时间按请求到达至预填充完成计量，决策耗时由本机 Python 实测。
