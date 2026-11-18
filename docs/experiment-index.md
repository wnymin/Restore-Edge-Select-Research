# 实验输入与复现

运行需要 Python 3.12 或 3.13。手算验收用仓库内的[轨迹与检查点](phase1.md)，无需下载。公开轨迹回放见[数据版本与校验](../data/README.md)；轨迹提供到达、输入长度和前缀哈希，实验另构造缓存放置、候选边界和预算。

- **续算：** [4K](evidence/prefill-vulkan-4k.json) 与 [8K](evidence/prefill-vulkan-8k.json) 成本网格来自 RTX 4060 Laptop 上的 Qwen3.5-4B Q4_K_M、llama.cpp b10923 Vulkan。按已恢复前缀与后缀插值，网格外不外推。
- **搬运：** [基础路径](evidence/load-io-profile.json)、[SSD 经锁页内存上传](evidence/load-staged-ssd-profile.json)；原始[分段读数](evidence/load-staged-ssd.csv)、[双路径并发](evidence/boundary-physical-channels.csv)、[与 prefill 并发](evidence/boundary-joint-contention.csv)。
- **主对照：** [算法、离线最优、表格及复算](trace-oracle.md)。[压缩包](evidence/replay-inputs.zip)保留原始回放和参数，[真实两槽恢复与外推](gpu-boundary-restore.md)可独立核对。
- **第三阶段 CPU 对照：** [动态状态干预、门控与留出回放](dynamic-cpu.md)。
- **CPU 最新增量与离线验收：** [成果入口](dynamic-cpu-latest.md)。可先运行 `python scripts/check_dynamic_cpu_delivery.py`，无需网络或原始公开轨迹。

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
