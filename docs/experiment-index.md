# 实验口径与复现

[CPU 研究](boundary-findings.md)链接到直接支持结论的配对数据。`docs/evidence/` 保存证据与校准输入，`outputs/` 是可重建的输出。命令从仓库根目录执行，先运行 `python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"`。基本用法见[运行与验收](phase1.md)。

硬件证据：[驻留前缀与冷恢复实测及离线外推](gpu-boundary-restore.md)。以下回放只消费已归档的数字参数；下载公开轨迹后，仅用 CPU 即可运行。默认成本保留历史实验口径，可通过支持的 `--grid` 和 `--io-profile` 替换。下文循环用 POSIX shell 书写，也可展开为逐条 Python 命令。

模拟与实现：[负载分类、策略切换和队列剪枝](boundary-findings.md)。

共享对象估计、跨层保护及成本误差：[成果与适用范围](cpu-policy.md) · [运行命令与精简证据](cpu-policy-reproduce.md)。

- [百炼轨迹及版本校验](../data/README.md)：到达时间、输入长度和前缀哈希来自原始轨迹；放置、检查点和预算由实验构造。主要比较留出窗口 18–23、24–29；16 倍和 32 倍 I/O 为人为压力点。模拟到达至预填充完成时间不等于实测 TTFT。
- [4K](evidence/prefill-vulkan-4k.json) 和 [8K](evidence/prefill-vulkan-8k.json) 续算网格来自 RTX 4060 Laptop、完整 Qwen3.5-4B Q4_K_M、llama.cpp b10923 Vulkan。按已恢复前缀和续算后缀做中位数二维插值，网格外不外推。
- [分段搬运配置](evidence/load-io-profile.json)、[SSD 暂存配置](evidence/load-staged-ssd-profile.json)、[SSD 实测](evidence/load-staged-ssd.csv)、[双路径并发](evidence/boundary-physical-channels.csv)及[与 prefill 并发](evidence/boundary-joint-contention.csv)提供输入；[8K 拓扑对照](evidence/boundary-lane-objective-24.csv)检验本机结论；[无搬运降速的两槽对照](evidence/boundary-batch-recovery-no-io.csv)用于隔离排序翻转机制。[双槽服务配对](evidence/boundary-batched-prefill-warm.csv)用于估算两槽进度；它不能用于高精度预测。
- 物理搬运配对将真实搬运与独立预热 KV 的续算串起来，搬运不写入模型槽。[搬运耗时](evidence/boundary-physical-recovery-trials.csv)和[等长等待耗时](evidence/boundary-physical-wait-trials.csv)保留分阶段读数。实际槽恢复与读量校验见[驻留与冷恢复对照](gpu-boundary-restore.md)。

## 百炼回放

```sh
python scripts/fetch_bailian.py
python scripts/fetch_bailian.py --verify-only
python scripts/check_fixture.py
for cohort in 18 24; do
  python scripts/study_reservation_placement.py --cohort $cohort --out "outputs/reservation-placement-$cohort.csv"
done
python scripts/study_reservation_placement.py --cohort 24 --timer-ms 100 --zero-only --out outputs/reservation-placement-timer-24.csv
python scripts/summarize_reservation_placement.py outputs/reservation-placement-18.csv outputs/reservation-placement-24.csv outputs/reservation-placement-timer-24.csv
python scripts/study_reservation_placement.py --cohort 24 --windows 1 --trace qwen_thinking_blksz_16.jsonl --timer-ms 100 --zero-only --out outputs/reservation-timer-witness.csv
for cohort in 18 24; do
  python scripts/study_batch_recovery.py --cohort $cohort --out "outputs/batch-recovery-$cohort.csv"
  python scripts/study_batch_recovery.py --cohort $cohort --no-io-contention --out "outputs/batch-no-io-$cohort.csv"
done
python scripts/summarize_batch_recovery.py outputs/batch-recovery-18.csv outputs/batch-recovery-24.csv
python scripts/summarize_batch_recovery.py outputs/batch-no-io-18.csv outputs/batch-no-io-24.csv
python scripts/study_batch_recovery.py --cohort 18 --start-window 21 --windows 1 --trace qwen_coder_blksz_16.jsonl --out outputs/batch-attribution-local.csv
python scripts/study_batch_recovery.py --cohort 18 --windows 1 --trace qwen_coder_blksz_16.jsonl --out outputs/batch-attribution-stress.csv
```

复核样本另见[24–29 窗的成本复核](evidence/boundary-robust-24.csv)和[8K 队列目标](evidence/boundary-queue-objective-8k.csv)。

```sh
for tier in split ram ssd; do
  file="outputs/first-change-$tier"
  python scripts/study_load_decomposition.py --skip-windows 18 --windows 6 --max-prompt 4096 --placement $tier --slow-path both --io-sweep 16 --grid docs/evidence/prefill-vulkan-4k.json --io-profile docs/evidence/load-staged-ssd-profile.json --zero-only --overlap-model dynamic --overlap-profile configs/overlap-observed-upper.json --closed-loop --attribute-first-change --output "$file.csv" --details "$file.jsonl"
done
python scripts/summarize_first_change.py --losses-only outputs/first-change-split.jsonl outputs/first-change-ram.jsonl outputs/first-change-ssd.jsonl
for skip in 18 24; do
  if [ "$skip" = 18 ]; then rates="16 32"; else rates="1 16 32"; fi
  for tier in split ram ssd; do
    python scripts/study_load_decomposition.py --skip-windows $skip --windows 6 --max-prompt 4096 --placement $tier --slow-path both --io-sweep $rates --grid docs/evidence/prefill-vulkan-8k.json --stability-grid docs/evidence/prefill-vulkan-4k.json --io-profile docs/evidence/load-staged-ssd-profile.json --overlap-model dynamic --overlap-profile configs/overlap-observed-upper.json --screened --robust-screen --output "outputs/robust-loop-$skip-$tier.csv"
  done
  python scripts/summarize_screened.py outputs/robust-loop-"$skip"-*.csv
done
```

```sh
python scripts/study_queue_objective.py --skip 18 24 --rates 1 32 --out outputs/boundary-queue-objective.csv
python scripts/summarize_queue_objective.py outputs/boundary-queue-objective.csv --expected-cases 288
python scripts/study_queue_objective.py --skip 24 --long 8192 --rates 1 32 --out outputs/boundary-queue-objective-8k.csv
python scripts/summarize_queue_objective.py outputs/boundary-queue-objective-8k.csv --expected-cases 288
python scripts/study_queue_objective.py --skip 18 --rates 32 --loss-source docs/evidence/boundary-queue-loss-input.csv --diagnostics outputs/queue-counterfactual.jsonl --repeats 0 --out outputs/queue-selected.csv
```

```sh
for lanes in shared independent; do
  python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes $lanes --lane-core --rates 1 32 --out "outputs/lane-objective-24-$lanes.csv"
done
python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes coupled --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-coupled.csv
python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes coupled --joint-prefill --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-joint.csv
python scripts/summarize_lane_objective.py outputs/lane-objective-24-shared.csv outputs/lane-objective-24-independent.csv outputs/lane-objective-24-coupled.csv outputs/lane-objective-24-joint.csv
```

## 双槽计时留出验证

只读已存的三组原始配对：同时提交、150 ms 校准和 300 ms 留出。该实验用于证明简单进度估计的误差，和真实深恢复评估是不同对照。

```text
python scripts/validate_batch_join.py docs/evidence/batch-warm.json docs/evidence/batch-stagger-150.json docs/evidence/batch-stagger-300.json
```

## 负载分类与队列搜索

同快照与零开销结果可复核；计时注入使用运行机器上的 Python 决策耗时，绝对值会随设备变化。

```text
python scripts/study_load_routing.py --cohort 18 24 --windows 3 --out outputs/load-routing-mixed.csv
python scripts/study_load_routing.py --cohort 24 --windows 3 --max-prompt 8192 --path ssd --rates 8 16 32 --out outputs/load-routing-ssd.csv
python scripts/study_load_routing.py --cohort 18 24 --windows 3 --no-snapshots --policies next4 rank_only rank_near --out outputs/load-routing-rank.csv
python scripts/study_load_routing.py --cohort 24 --windows 3 --max-prompt 8192 --path ssd --rates 8 16 32 --no-snapshots --policies next4 rank_only rank_near --out outputs/load-routing-rank-ssd.csv
python scripts/summarize_load_routing.py --cases 144 outputs/load-routing-rank.csv outputs/load-routing-rank-ssd.csv
python scripts/summarize_load_routing.py --snapshots --cases 144 outputs/load-routing-mixed.snapshots.csv outputs/load-routing-ssd.snapshots.csv
python scripts/summarize_load_routing.py --breakdown --cases 24 outputs/load-routing-ssd.csv
python scripts/study_load_routing.py --cohort 18 24 --windows 3 --rates 1 16 32 --no-snapshots --policies next4 lean compute cutoff rank_near --out outputs/search-mixed.csv
python scripts/study_load_routing.py --cohort 24 --windows 3 --max-prompt 8192 --path ssd --rates 16 32 --no-snapshots --policies next4 lean compute cutoff rank_near --out outputs/search-ssd.csv
python scripts/study_load_routing.py --cohort 30 --windows 3 --trace qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl --rates 1 16 32 --no-snapshots --policies next4 lean compute cutoff rank_near --out outputs/search-fresh.csv
python scripts/summarize_load_routing.py outputs/search-mixed.csv outputs/search-ssd.csv outputs/search-fresh.csv
python scripts/summarize_search_pairs.py outputs/search-mixed.csv outputs/search-ssd.csv outputs/search-fresh.csv
python scripts/summarize_search_factors.py outputs/search-mixed.csv outputs/search-ssd.csv outputs/search-fresh.csv
python scripts/study_load_routing.py --cohort 33 --windows 3 --trace qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl --rates 1 16 32 --no-snapshots --policies next4 lean cutoff entry rank_near --out outputs/search-entry-holdout.csv
python scripts/study_load_routing.py --cohort 24 --windows 3 --max-prompt 8192 --path ssd --rates 16 32 --no-snapshots --policies next4 lean cutoff entry rank_near --out outputs/search-entry-ssd.csv
python scripts/summarize_load_routing.py outputs/search-entry-holdout.csv outputs/search-entry-ssd.csv
python scripts/summarize_search_pairs.py outputs/search-entry-holdout.csv outputs/search-entry-ssd.csv
```
