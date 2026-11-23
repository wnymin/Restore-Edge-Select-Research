# R08：`entry-replay` 在不同模型和暂存容量下的适用域

**结论可以推广到一部分模型和容量组合，不能推广成所有组合都近最优。** 4B 在 128–512 MiB 的受测压力点捕获率下界最低约为 98.86%；27B 架构载荷需要更大容量才出现可认证的收益，在 16×／384–512 MiB 为 95.76%–96.13%，16×／256 MiB 的证据仍不足以判定达到 95%。合成交替层在多个容量点是基线已经最优，不是算法带来额外收益。换用两套本机实测成本后仍能观察到跨模型、跨容量的近最优区间，但**平均近最优不保证单请求不退化**。

## 区分两种容量对照

1. **同一块预算：** 三套模型载荷都给 64、128、176、180、256、384、512 MiB。结果回答「预算相同、换模型会怎样」。
2. **同一类竞争形态：** 对每个模型分别测一个冷 512-token 候选、两个互不共享的该候选的参考线两侧（±1 MiB）。它不是保证每个源窗都有两个同样的候选；用实际预留次数和可装下的候选数核对。完整候选字节为对应前缀 KV 加边界的循环／卷积检查点；共享时按对象 ID 取并集。

| 模型载荷 | 一个 512-token 候选 | 两个不共享 512-token 候选 | 续算输入 |
| --- | ---: | ---: | --- |
| 4B | 65.125 MiB | 130.250 MiB | R07 归档网格 |
| 合成交替层 | 64.750 MiB | 129.500 MiB | R07 1.25× 代理 |
| 27B 官方架构载荷 | 178.813 MiB | 357.625 MiB | R07 2× 代理 |
| 实测的 16 层衍生模型 | 32.563 MiB | 65.125 MiB | 本轮 RTX 4060 Laptop GPU 实测网格 |

所有配对冻结 `entry-replay`、两请求预取窗、四步策略窗、单计算槽和单串行搬运。128 MiB 装不下不等于队首不能恢复；预取不成功仍可到队首按需加载。三份轨迹同属 Bailian，不是三个独立生产部署。

## 容量临界点的因果核对

72 个短用例均以全组合穷举认证最优。固定 `(0,576,576)` 只用于观察实际预留：忙时第一个恢复请求在额度恰等于所需字节时能预留，差 1 字节就不能；两个请求同刻或稍错开时，第二个请求在二者并集恰好装下时才立即预留。共享一个 KV 块降低并集门槛，不降低单候选门槛；待首请求结束才来的第二个恢复请求即使容量更大也不会走同一条预取路径。

| 模型 | 576-token 单候选 | 两个不共享的并集 | 共享一个 576-token KV 块时的并集 |
| --- | ---: | ---: | ---: |
| 4B | 67.125 MiB | 134.250 MiB | 116.250 MiB |
| 合成交替层 | 68.750 MiB | 137.500 MiB | 101.500 MiB |
| 27B 架构载荷 | 182.813 MiB | 365.625 MiB | 329.625 MiB |

72 个短用例中，16 个有可省空间，冻结算法都选到最优；其余 56 个基线已经最优。这个结果只是用于确认机制的短轨迹，不代替更长的留出。

## 新轨迹留出

接近率是各配对窗先取均值，再计算「比 `local-cost` 基线省下的时间 / 离线最优比该基线最多能省的时间」。比值不与各窗百分比的算术平均混用；没有可省空间不记为 100%。下表中的区间由离线搜索上下界决定，不是统计置信区间。时间单位为毫秒，模拟指标为请求恢复响应，非真实 TTFT。

检查了 **12 个新源窗 × 72 个配置 = 864 点**：782 点认证了精确离线最优，767 点认证冻结算法选到最优；504 点证实基线已最优，281 点证实有可省空间，另外 79 点不能确定基线是否最优。72 组中 25 组通过预登记的平均容差，22 组无空间，25 组未定。没有任何一组在两个源窗复现预登记的实质缺口；这不否认单窗反例。另有 144 个设计点（125 个精确），不计入上面的留出结果。

**同 MiB 容量：**

| 模型 | SSD 压力 | 64 MiB | 128 MiB | 176 MiB | 180 MiB | 256 MiB | 384 MiB | 512 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | 12× | 未认证 | 100.00% | 99.56% | 99.56% | 99.56% | 99.56% | 99.56% |
| 4B | 16× | 未认证 | [98.86%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 合成交替层 | 12× | 未认证 | 无空间 | 无空间 | 无空间 | 无空间 | 无空间 | 无空间 |
| 合成交替层 | 16× | 未认证 | 无空间 | 无空间 | 无空间 | 无空间 | 无空间 | 无空间 |
| 27B 架构载荷 | 12× | 未认证 | 未认证 | 未认证 | 未认证 | 100.00% | 100.00% | 100.00% |
| 27B 架构载荷 | 16× | 未认证 | 未认证 | 未认证 | 未认证 | [94.38%, 100.00%] * | 95.76% | 96.13% |

**按候选字节对齐：**

| 模型 | SSD 压力 | 一个 −1 MiB | 一个 +1 MiB | 两个 −1 MiB | 两个 +1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 12× | 未认证 | 未认证 | 99.52% | 99.53% |
| 4B | 16× | 未认证 | 未认证 | 100.00% | 100.00% |
| 合成交替层 | 12× | 未认证 | 未认证 | 无空间 | 无空间 |
| 合成交替层 | 16× | 未认证 | 未认证 | 无空间 | 无空间 |
| 27B 架构载荷 | 12× | 未认证 | 未认证 | 100.00% | 100.00% |
| 27B 架构载荷 | 16× | 未认证 | 未认证 | 95.76% | 95.76% |

「未认证」表示组内可省空间下界为零，不能给有意义的接近率；不代表所有点都没收益。`*` 表示可报区间，但组未通过预登记容差。六组 1×（三个模型 × 128/256 MiB，72 点）均认证基线已最优；本轮没有扫描 1× 至 12× 之间，不能据此给出 11× 的普适转折点。

**含选点耗时。** 72 组中，`entry-replay` 相比 `entry` 的每窗累计选点耗时，组均值均较短，差为 0.26–1.34 ms，零开销的边界序列相同。作为量级参照，4B／12×／176 MiB 相对 `local-cost` 的含开销平均响应与窗内 P95 平均分别缩短 249.30 / 498.14 ms；27B 架构载荷／16×／384 MiB 分别缩短 532.38 / 1023.89 ms。这些汇总是对每窗六次配对差取中位数后再跨窗取均值。**165 个配置**出现同一请求在至少四轮多等超过 10 ms（覆盖 11 个源窗），最坏单轮个体损失 **36.25 ms**；同一个窗在多个容量下出现慢请求不算独立复现。

**已单独重放的容量与选点反例（Trace B 481）。** 27B 架构载荷的 16× 点从 176 加到 180 MiB 时，能独立装下的冷候选从 0 个涨到 14 个，算法实际预留从 0 次涨到 13 次，但基线仍最优；384 MiB 时 21 个冷候选都能独立装下，实际预留 14 次，才出现该窗 681.075 ms 的离线改善空间，冻结策略拿到全部。相反，4B／176 MiB／12× 的 21 个冷候选已全部能独立装下，但四步看见的收益让策略给目标请求选 1024 token；把比较扩大到决策时已到达的请求，改选 512 token，固定这一个动作、继续在线执行，就少等 13.251 ms，补回该窗约 40.1% 的可省空间。180、384 MiB 仍有同源缺口，不算独立复现。前者说明「可预取」不等于存在选点收益，后者说明容量充足也修不了可见范围的失误。

**另一种失误（Trace B 482）。** 27B 架构载荷／16×／384 MiB 的该窗与精确离线最优差 **252.67 ms**，损失该窗可省空间的 **13.91%**。只把比较范围扩大到所有已到达请求，仍选 1024 token；同时重算两个已可见的后续请求候选才翻成 512 token。固定这个动作、继续在线执行，可消去约 222.19 ms，不能解释全部离线差。主留出共记录 11 个可见范围与 4 个联合选点诊断（按配置计，不是独立窗，也没有把诊断策略当线上候选）。

## 原生成本交叉复核

本机用同一 `llama-bench`（`8a56aedd6`）和 Vulkan1，分别跑现有 32 层 4B，以及由相同 GGUF 前 16 层构造的 2.42B 参数衍生配置。每个配置测 40 个前缀×后缀拟合形状和 4 个网格外对照形状，每形状五次；四个形状的插值绝对相对误差最大分别为 4.62% 与 3.80%。例如无缓存的 512-token 测试，中位数约为 **199.769 / 97.960 ms**。这是实际运行该计算结构的预填充成本，不验证衍生模型的生成质量，也不是另一个发布模型。

另外在 8/16/32 MiB KV 和三种检查点实际载荷尺寸测了同步搬运，每路径每尺寸 11 次。拟合斜率约为 RAM→HBM **11.987 MiB/ms**、SSD 无缓冲读＋上传 **2.744 MiB/ms**、设备内复制 **107.998 MiB/ms**；各路径已测尺寸的最大相对拟合残差分别为 0.49%、6.21%、16.97%。这以独立同步路径校准模拟，不替代服务异步搬运；重叠仍沿用既有模型。27B 只核对了载荷尺度的搬运，未实测 27B 模型续算。

实测分线在另六个源窗回放 204 个配置（168 个全实测成本、36 个成本来源配对），199 个精确认证。1× 全实测的 84 个配置均证明基线已最优；这不是额外的算法优势。16× 仍只是把实测 SSD 路径人为放慢，以下均非真实服务 TTFT：

| 16×，两项都用实测成本 | 64 MiB | 128 MiB | 256 MiB |
| --- | --- | --- | --- |
| 32 层 4B | 无可证改善；1 个窗未认证 | **100%** | **100%** |
| 16 层衍生 | **100%** | **100%** | **100%** |

两模型按各自候选大小对齐后，在「两个互不共享候选」参考线两侧也均认证 100%；在「第一个候选」参考线两侧尚无可证的组内改善空间，各有未认证窗。也就是说，打开第一个预取机会不等于恢复选点就出现改进空间。按六窗的零开销平均，4B／128 MiB／16× 与 16 层同条件分别拿到 174.590 / 68.496 ms 的可省时间。只换一类成本时，4B 仅换搬运出现 99.462%（组均差 0.760 ms）；16 层仅换搬运仍测不出改善空间、仅换计算则出现可证收益。用一个统一倍数同时代替两个成本路径会抹掉这种差异。

全实测成本的 168 个配置仍有 16 个出现逐请求重复超阈，均来自 4B 压力条件；完整分线（含 36 个成本对照）共 27 个，最大个体损失 14.668 ms。16 层这批全实测点未发现该阈值上的个体退化，不等于逐请求保证。

具体回归靶点及其调优优先级见 [R08 反例登记](archive/r08-process/r08-anomaly-register.md)；下一轮只讨论、未启动的验证见 [真实调度实验讨论](archive/r08-process/r08-live-scheduling-discussion.md)。

## 能推广的结论

- **可推广：** 在保留精确检查点、固定放置、单计算槽和单串行搬运的前提下，冻结 `entry-replay` 不只在原 4B、原额度有效。对 27B 架构载荷取得了有条件的跨模型近最优证据；对 16 层衍生配置又用实测成本交叉复核。容量应同时以 MiB 和候选并集字节描述，不能把某个 MiB 点直接移植给换层的模型。
- **不能推广：** 「越大越好」「一旦可预取就一定有算法收益」「平均近最优就没有慢请求」均不成立或无足够证据。25 个未定组仍应报告区间或预算不足，不作为支持证据。特别是 27B／16×／256 MiB 还不能排除超过 5% 的缺口。
- **未覆盖：** 27B 原生续算、其他发布模型的真实成本、真实端到端 TTFT、跨部署、不同检查点契约、并行搬运、动态合批和抢占。本轮区分了载荷、暂存和成本变化，没有外推到这些执行语义。

## 复核与出处

[预登记、矩阵与勘误](archive/r08-process/r08-preregistration.md) · [汇总](evidence/r08-summary.json) · [哈希清单](evidence/r08-provenance.json) · [临界测试](evidence/r08-micro.jsonl.gz) · [设计窗](evidence/r08-design.jsonl.gz) · [主留出原始批次](evidence/r08-holdout.jsonl.gz) · [原始批次计时](evidence/r08-timed.jsonl.gz) · [Coder 101 替换窗](evidence/r08-replacement.jsonl.gz) · [替换窗计时](evidence/r08-replacement-timed.jsonl.gz) · [GPU 预填充原始分布](evidence/r08-native-prefill.json.gz) · [实际搬运原始分布](evidence/r08-native-io.json.gz) · [实测成本留出](evidence/r08-native.jsonl.gz) · [实测成本留出计时](evidence/r08-native-timed.jsonl.gz)。

```powershell
python scripts/check_fixture.py
python -m scripts.validation.check_r07_extension
python -m scripts.validation.audit_r08_source_windows
python -m scripts.validation.check_r08_capacity
```

原始主批 Coder 96 与更早的留出见证重合，最终由 Coder 101 的 72 个结果及其配对计时替换，不重复计数。原始公开轨迹按仓库数据清单核验；复算入口为 `scripts.cpu.replay.study_r08_capacity` 和 `scripts.cpu.replay.study_r08_native`。GPU 形状测量用 `scripts.gpu.bench.measure_r08_prefill` 和 `scripts.gpu.bench.measure_r08_paths`。16 层配置由 `scripts.data.make_r08_calibration_gguf` 从本地开源 GGUF 派生，权重不随证据提交。时间测量依赖本机运行环境；GPU 校准与不计开销的 CPU 回放有重叠，并非空机测量。每个 CPU 计时点的六轮顺序执行；复核从归档重算原报告的计时汇总。

<details>
<summary>从固定轨迹与归档实测成本重新回放</summary>

从本仓库根目录运行。先按[数据清单](experiment-index.md)准备轨迹；计时随机器负载变化，不要求与旧记录逐项相等。

```powershell
python -m scripts.data.fetch_bailian --verify-only
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python -m scripts.cpu.replay.study_r08_capacity micro --out outputs/r08-pre-micro.jsonl
python -m scripts.cpu.replay.study_r08_capacity design --out outputs/r08-pre-design.jsonl
python -m scripts.cpu.replay.study_r08_capacity holdout --out outputs/r08-pre-holdout.jsonl
python -m scripts.cpu.replay.study_r08_capacity timed --untimed outputs/r08-pre-holdout.jsonl --out outputs/r08-pre-timed.jsonl
python -m scripts.cpu.replay.study_r08_replacement replacement --out outputs/r08-replacement.jsonl
python -m scripts.cpu.replay.study_r08_replacement replacement-timed --untimed outputs/r08-replacement.jsonl --out outputs/r08-replacement-timed.jsonl
python -c "import gzip; from pathlib import Path; [Path(f'outputs/r08-native-{p}.json').write_bytes(gzip.decompress(Path(f'docs/evidence/r08-native-{p}.json.gz').read_bytes())) for p in ['prefill','io']]"
python -m scripts.cpu.replay.study_r08_native --prefill outputs/r08-native-prefill.json --io outputs/r08-native-io.json --out outputs/r08-native.jsonl
python -m scripts.cpu.replay.study_r08_native --prefill outputs/r08-native-prefill.json --io outputs/r08-native-io.json --timed-from outputs/r08-native.jsonl --out outputs/r08-native-timed.jsonl
```

保留已验收归档时不要运行 `--archive`，该选项会用新回放覆盖报告的原始归档。原生测量命令另需兼容的本地权重和 `llama-bench`；搬运基准显式传 `--rounds 11`。

</details>
