# 相同执行状态的离线分支合并

某些边界历史虽然不同，但在后续选点时会汇合到完全相同的执行状态。如果已经完成的请求累计耗时更大，这条历史可被另一条相同状态、累计耗时更小的历史覆盖，后续动作再相同也无法反超。

状态键包含当前选点请求、时刻、队首、已到达数量、仍未完成请求的已提交边界、预取预留、就绪对象、延迟释放与复制阶段，以及当前计算和传输的剩余执行状态。已经结束的请求只通过其累计完成时间参与比较。这里对状态使用精确的有理数和对象标识相等，未合并只是近似相似的状态。

搜索使用该状态保存目前最小的已完成时间之和。再遇到相同状态且累计耗时不小于已存值时，舍弃这一分支。先前较好的状态已展开的子分支仍在搜索前沿中，因此其可选后续方案继续覆盖被舍弃分支。

## 对照

300 个三请求穷举配置及额外 18 个五请求配置均与精确离线最优一致。五请求对照覆盖密集与稀疏到达、三档预算及单一/交替前缀，确实合并了 38 个被支配状态。用 `--dominance --resource-floor` 也通过了三请求穷举对照。补充的 [240 个精确时钟配置](dynamic-cpu-exact-clock.md)覆盖驻留、降速和额度释放，状态合并也与穷举结果一致。

在此前区间最宽的 12 个公开轨迹配置上，每个配置最多展开 2,048 个节点，未遇到可合并的相同状态，最终区间与原搜索完全相同。高压力、请求持续排队时，不同边界历史往往仍有不同的在途工作、时间或预留状态；这组数据尚未显示状态合并能缩短此类证明。

[逐阶段状态数与区间](../../evidence/dynamic-cpu-oracle-dominance.csv) · [输入与参数](../../evidence/dynamic-cpu-oracle-dominance.json)

## 复现

~~~sh
python -m scripts.cpu.replay.study_adaptive_oracle --dominance --out outputs/dominance-exhaustive.csv
python -m scripts.cpu.replay.study_adaptive_oracle --dominance --resource-floor --out outputs/dominance-resource-exhaustive.csv
python -m scripts.cpu.dynamic.refine_dynamic_cpu_oracle --dominance --top 12 --nodes 32 256 2048 --out outputs/oracle-dominance.csv
~~~

离线原接口仍默认关闭状态合并，通过 `dominance=True` 可启用；`diagnostics` 字典返回合并数、不同状态数和剩余前沿数。这个优化的证据目前来自构造配置，公开轨迹上的负结果也保留。
