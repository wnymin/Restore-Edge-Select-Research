# 已确定边界的推演优化

`entry-fast` 在选点时为各候选重放一段已到达的请求。重放中的边界已经由候选和后续基线序列指定，原实现仍通过通用选点入口逐项提交。本轮保持候选、可见窗口、评分、同分规则和事件约束，检查能否降低这部分 CPU 成本。

## 两项实现与消融

- `entry-commit` 沿用 `entry-fast`，仅在零开销的固定序列推演内部直接提交已指定边界。每个边界仍检查合法性；通用选点、决策计时和费用分支可以省去。实际服务中的在线选点仍经过原有入口。
- `entry-replay` 在上述基础上，省去已知固定成本推演中的单位因子运算。续算量无需再乘以 1；并发系数切换时，若旧系数或新系数为 1，可省去相应的除法或乘法。全部时钟和剩余工作仍使用精确有理数。

两种模式都使用既有快路的条件检查。估计误差、时变传输、主机暂存缓存和多传输通道等情形回到原 `entry` 路径。三种实现的零开销完整输出必须相同，包括逐请求完成时刻、边界、搬运、容量、推演次数和事件数。

## 比较方式

先在原 144 个校准配置上比较 `entry-fast`、`entry-commit` 与 `entry-replay`，随后保持实现和比较规则，在原 480 个留出配置上复核。每配置预热两次，每阶段执行六轮，三种策略的六个排列各出现一次，顺序由种子 20261124 打乱。固定轨迹计时与注入选点时长的闭环回放分别执行。配置内取配对差中位数，再对配置等权平均。

输入沿用[冻结成本与数据范围](evidence/dynamic-cpu-frozen.json)，公开轨迹先核对上游 SHA-256。另用 840 个兼容配置检查两种新模式的支持与回退，以及 240 个精确时钟配置检查初始驻留、共享、并发降速、释放规则和分数到达时间。构造检查分别使用种子 20261124 和 20261125。

## 结果

| 统计量 | 校准配置 | 留出配置 |
| --- | ---: | ---: |
| 配置数 | 144 | 480 |
| `entry-fast` 选点总时间的配置均值 | 3.982 ms | 4.631 ms |
| `entry-commit` 选点总时间的配置均值 | 3.901 ms | 4.477 ms |
| `entry-replay` 选点总时间的配置均值 | 3.718 ms | 4.294 ms |
| `entry-replay` 相对 `entry-fast` 的选点耗时下降 | 6.64% | 7.26% |
| 配对中位数显示选点更快的配置 | 112 / 144 | 352 / 480 |
| 注入选点时长后的平均响应改善 | 0.0048 ms | 0.0141 ms |
| 注入后的 P95 改善的配置均值 | 0.0089 ms | 0.0303 ms |
| 全部重复轮次中最大的单请求额外等待 | 1.119 ms | 6.050 ms |

表中的时间表示每配置所有请求的选点时间之和，百分比由两种实现的配置均值计算。`entry-commit` 单独在留出降低了 3.33% 的选点耗时，完整优化为 7.26%，说明单位因子运算也构成了一部分开销。零开销下三种实现逐项相同，480 个留出配置的平均响应还与此前归档的原 `entry` 结果一致。

本轮支持完整优化进一步降低本机 Python 选点开销。注入计时后，480 个配置中 297 个平均响应改善，仍有配置或单请求变慢。平均响应改善远小于选点耗时的百分比，且计时会受系统调度影响，不能据此推断真实服务的端到端提速或单请求延迟保证。`entry-fast` 继续作为保留对照，新模式通过显式名称启用。

[校准逐配置](evidence/dynamic-cpu-entry-replay-calibration.csv) · [校准逐轮配对](evidence/dynamic-cpu-entry-replay-calibration-paired.csv) · [校准输入版本](evidence/dynamic-cpu-entry-replay-calibration.json)

[留出逐配置](evidence/dynamic-cpu-entry-replay-holdout.csv) · [留出逐轮配对](evidence/dynamic-cpu-entry-replay-holdout-paired.csv) · [留出输入版本](evidence/dynamic-cpu-entry-replay-holdout.json)

[兼容性配置](evidence/dynamic-cpu-entry-replay-compatibility.csv) · [精确时钟构造](evidence/dynamic-cpu-entry-replay-synthetic.csv)

## 复现

~~~sh
python -m scripts.cpu.dynamic.study_entry_replay --synthetic-only --out outputs/entry-replay/check.csv
python -m scripts.cpu.dynamic.study_entry_replay --out outputs/entry-replay/calibration.csv
python -m scripts.cpu.dynamic.study_entry_replay --holdout --out outputs/entry-replay/holdout.csv
~~~

首条命令无需公开轨迹，其余命令需要百炼数据，可追加 `--trace-dir` 指定目录。完整计时可用 `--repeats 12` 增加平衡轮次。新的模式也可以通过 `study_load_routing.py --policies entry-replay` 使用，CLI 仍需填写该脚本要求的共同参数。
