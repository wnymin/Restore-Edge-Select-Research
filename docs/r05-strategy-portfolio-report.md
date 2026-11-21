# R05 CPU 策略组合与切换报告

2026-11-20 完成 CPU 范围验收。Trace B 留出窗口 350–361 共 864 个配置。计入决策开销后，`portfolio-v4` 相对预登记的固定对照 `entry` 平均降低响应 0.0587 ms/config，按 12 个窗口聚类得到的 95% bootstrap 区间为 [0.0360, 0.0806] ms/config。11 个窗口的净收益为正，1 个为负。相对同语义的优化实现 `entry-replay`，`portfolio-v4` 的平均响应高 0.0056 ms/config。后者是判断门控额外收益的实现对照，也是 R06 的性能参照。

## 校准与策略

使用 Trace B 窗口 200–202 的 216 个配置校准。窗口 300 用于试跑，最终留出使用窗口 350–361。下表为校准配置上相对 `local-cost` 的平均含开销收益。

| 策略 | 收益（ms/config） |
| --- | ---: |
| `entry` | 373.0152 |
| `next4` | 372.9302 |
| `entry-replay` | 373.1745 |
| `adaptive` | 173.8017 |

`portfolio-v4` 在 RAM 布局或 SSD 成本倍率不高于 11 时允许进入快速路径。运行时对预留、搬运、可复用块和候选预取条件执行检查，满足条件时保留串行选择，其余情况用 `entry-replay` 完整求值。倍率来自该模拟配置的校准。

## 留出结果

| 策略 | 相对 `local-cost` 的平均含开销收益（ms/config） |
| --- | ---: |
| `entry` | 274.5222 |
| `portfolio-v4` | 274.5809 |
| `next4` | 274.4616 |
| `entry-replay` | 274.5864 |
| `mixed-reuse-estimate` | 232.8539 |
| `adaptive` | 120.8874 |

相对 `entry`，81,072 个请求的零开销恢复边界全部一致。含开销的最大单请求退化为 8.5122 ms，超过 10 ms 的请求数为 0。各配置 P95 响应差值的均值为 −0.1094 ms。这组结果描述本次 CPU 模拟中含选点开销的响应，GPU 和真实服务性能留待独立验证。

对应版本的门控消融中，`portfolio-v3-dynamic-only`、`portfolio-v3-static-only` 的平均含开销收益分别为 270.7859 和 265.7508 ms/config，该批 `entry` 和 `portfolio-v4` 分别为 274.5110 和 274.5715 ms/config。结果说明这两种单独使用门控的版本均未达到对照收益。选点路径的成本归因还需参考同语义 `entry-replay`，以及 [R06 的配对计时与路由审计](r06-completion.md)。

## 复现

归档包含[配置级结果](evidence/r05-cpu04-holdout-final.csv.gz)、[请求级结果](evidence/r05-cpu04-holdout-requests.csv.gz)、[配对汇总](evidence/r05-cpu04-summary.json)及[消融汇总](evidence/r05-cpu03-ablation-summary.json)。用 `python -m scripts.validation.check_r05_delivery` 验收归档。准备 Trace B 后，从仓库根目录重新回放。

```powershell
python -m scripts.cpu.replay.study_load_routing `
  --cohort 350 --windows 12 --trace qwen_traceB_blksz_16.jsonl `
  --path ssd --rates 1 8 11 12 16 32 --max-prompt 3584 `
  --request-metrics --no-snapshots --fingerprints `
  --policies next4 entry entry-replay mixed-reuse-estimate adaptive portfolio-v4 `
  --request-output outputs/r05-cpu04-holdout-requests.csv `
  --out outputs/r05-cpu04-holdout-final.csv
```
