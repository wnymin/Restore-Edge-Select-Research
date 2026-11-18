# 实验输入与复现

本轮连续研究已合并为[有限暂存容量下的前缀恢复边界选择](dynamic-gpu-lookahead-report.md)。报告统一列出结论、证据、适用边界与后续方向，第 6 节索引各轮详细推导和复现命令。

前瞻算法的分轮输入、反例和复现命令集中在[主报告第 6 节](dynamic-gpu-lookahead-report.md#6-合并与复现索引)和[证据附录](dynamic-gpu-lookahead-appendices.md)。

本项目所有动态因素研究均遵循[研究约束与因素取舍协议](recovery-research-protocol.md)：先按影响程度和可量化性排序，再用消融、留出与损失预算决定是否简化。

本项目动态因素研究统一遵循[恢复边界研究约束与因素取舍协议](recovery-research-protocol.md)。

## 从公开轨迹重新运行主对照

~~~sh
python scripts/fetch_bailian.py
python scripts/fetch_bailian.py --verify-only
python scripts/check_fixture.py
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
traces="qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl"
common="--trace $traces"  # 下列命令按 POSIX shell 展开变量；PowerShell 中可直接填写文件名。
python scripts/study_load_routing.py $common --cohort 109 --windows 3 --path ssd --rates 1 8 11 12 16 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --policies next4 entry mixed-reuse-estimate reuse-cross-tier-fallback adaptive adaptive-screened adaptive-strict --out outputs/adaptive-holdout-109.csv
python scripts/study_load_routing.py $common --cohort 120 --windows 2 --path all --rates 1 8 11 12 16 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --zero-only --policies next4 entry mixed-reuse-estimate adaptive-mean adaptive-mean-audit adaptive adaptive-strict --out outputs/adaptive-objectives-120.csv
~~~

手算和零开销回放是确定性的。注入决策耗时的结果取决于运行机器；要复算原表，使用压缩包里的原始 CSV。其余主对照参数见[自适应实验清单](evidence/adaptive-manifest.json)和[本机成本实验清单](evidence/local-cost-study-manifest.json)。

逐请求反例可用固定其余边界的干预重建：

~~~sh
python scripts/study_tail_witness.py --trace qwen_traceA_blksz_16.jsonl --window 77 --placement ssd --spacing 1 --budget 128 --rate 32 --path all --reference-policy mixed-estimate --intervene-policy mixed-projected-entry
python scripts/study_tail_witness.py --trace qwen_coder_blksz_16.jsonl --window 89 --placement split --spacing 1/4 --budget 128 --rate 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --overlap-profile configs/overlap-ssd-compute-half-control.json
python scripts/study_tail_witness.py --trace qwen_coder_blksz_16.jsonl --window 96 --placement ssd --spacing 1 --budget 128 --rate 32 --overlap-profile configs/overlap-disabled-control.json --ssd-estimate-factor 2 --intervene-policy reuse-cross-tier-guard
~~~

## 调度与传输拓扑的独立对照

以下实验用于检查结论能否跨暂存释放规则、预取放置、双槽及传输通道成立。它们与单槽选点的离线最优表使用不同的事件约束。

~~~sh
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
~~~

~~~sh
python scripts/study_queue_objective.py --skip 24 --long 8192 --rates 1 32 --out outputs/boundary-queue-objective-8k.csv
python scripts/summarize_queue_objective.py outputs/boundary-queue-objective-8k.csv --expected-cases 288
python scripts/study_queue_objective.py --skip 18 --rates 32 --loss-source docs/evidence/boundary-queue-loss-input.csv --diagnostics outputs/queue-counterfactual.jsonl --repeats 0 --out outputs/queue-selected.csv
~~~

~~~sh
for lanes in shared independent; do
  python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes $lanes --lane-core --rates 1 32 --out "outputs/lane-objective-24-$lanes.csv"
done
python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes coupled --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-coupled.csv
python scripts/study_queue_objective.py --skip 24 --long 8192 --lanes coupled --joint-prefill --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-joint.csv
python scripts/summarize_lane_objective.py outputs/lane-objective-24-shared.csv outputs/lane-objective-24-independent.csv outputs/lane-objective-24-coupled.csv outputs/lane-objective-24-joint.csv
~~~

## 双槽计时留出验证

只读已存的三组原始配对：同时提交、150 ms 校准和 300 ms 留出。该实验用于证明简单进度估计的误差，和真实深恢复评估是不同对照。

~~~sh
python scripts/validate_batch_join.py docs/evidence/batch-warm.json docs/evidence/batch-stagger-150.json docs/evidence/batch-stagger-300.json
~~~

## GPU 归档区间审计

主报告第 3.23 节的区间审计使用已有 CSV。运行 `python scripts/audit_gpu_interval_coverage.py --check docs/evidence/dynamic-gpu-interval-audit-7870.json` 可复核冻结区间、来源哈希与逐批计数。
