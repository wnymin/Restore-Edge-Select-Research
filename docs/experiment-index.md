# 实验输入与复现

脚本按任务分类，见[脚本索引](../scripts/README.md)。下列命令均从实验仓库根目录运行。

本轮连续研究已合并为[有限暂存容量下的前缀恢复边界选择](dynamic-gpu-lookahead-report.md)。报告统一列出结论、证据、适用边界与后续方向，第 6 节索引各轮详细推导和复现命令。

前瞻算法的分轮输入、反例和复现命令集中在[主报告第 6 节](dynamic-gpu-lookahead-report.md#6-合并与复现索引)和[证据附录](dynamic-gpu-lookahead-appendices.md)。

本项目所有动态因素研究均遵循[研究约束与因素取舍协议](recovery-research-protocol.md)：先按影响程度和可量化性排序，再用消融、留出与损失预算决定是否简化。

## CPU 前瞻审计数据归档

主报告与附录中以文件名标注的 `dynamic-lookahead-*.csv` 收在[审计数据 ZIP](evidence/lookahead-audit-csv.zip)，[SHA-256 清单](evidence/lookahead-audit-csv.sha256)列出归档成员。CSV 内容不变；JSON 摘要、反例输入和 GPU 实测仍单独保留。`scripts/validation/` 中的归档检查会按需读取，缓存写入 Git 忽略的 `outputs/lookahead-evidence/`，无需先手动解包。

~~~sh
python -m scripts.validation.lookahead_evidence verify
python -m scripts.validation.lookahead_evidence list
python -m scripts.validation.lookahead_evidence extract dynamic-lookahead-snapshot-gap-7760.csv
~~~

`extract` 不带文件名时提取全包。重新生成的新审计结果先写入 `outputs/`；不要将解包副本放回 `docs/evidence/`。

## 从公开轨迹重新运行主对照

~~~sh
python -m scripts.data.fetch_bailian
python -m scripts.data.fetch_bailian --verify-only
python scripts/check_fixture.py
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
traces="qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl"
common="--trace $traces"  # 下列命令按 POSIX shell 展开变量；PowerShell 中可直接填写文件名。
python -m scripts.cpu.replay.study_load_routing $common --cohort 109 --windows 3 --path ssd --rates 1 8 11 12 16 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --policies next4 entry mixed-reuse-estimate reuse-cross-tier-fallback adaptive adaptive-screened adaptive-strict --out outputs/adaptive-holdout-109.csv
python -m scripts.cpu.replay.study_load_routing $common --cohort 120 --windows 2 --path all --rates 1 8 11 12 16 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --zero-only --policies next4 entry mixed-reuse-estimate adaptive-mean adaptive-mean-audit adaptive adaptive-strict --out outputs/adaptive-objectives-120.csv
~~~

手算和零开销回放是确定性的。注入决策耗时的结果取决于运行机器；要复算原表，使用压缩包里的原始 CSV。其余主对照参数见[自适应实验清单](evidence/adaptive-manifest.json)和[本机成本实验清单](evidence/local-cost-study-manifest.json)。

逐请求反例可用固定其余边界的干预重建：

~~~sh
python -m scripts.cpu.replay.study_tail_witness --trace qwen_traceA_blksz_16.jsonl --window 77 --placement ssd --spacing 1 --budget 128 --rate 32 --path all --reference-policy mixed-estimate --intervene-policy mixed-projected-entry
python -m scripts.cpu.replay.study_tail_witness --trace qwen_coder_blksz_16.jsonl --window 89 --placement split --spacing 1/4 --budget 128 --rate 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --overlap-profile configs/overlap-ssd-compute-half-control.json
python -m scripts.cpu.replay.study_tail_witness --trace qwen_coder_blksz_16.jsonl --window 96 --placement ssd --spacing 1 --budget 128 --rate 32 --overlap-profile configs/overlap-disabled-control.json --ssd-estimate-factor 2 --intervene-policy reuse-cross-tier-guard
~~~

## 调度与传输拓扑的独立对照

以下实验用于检查结论能否跨暂存释放规则、预取放置、双槽及传输通道成立。它们与单槽选点的离线最优表使用不同的事件约束。

~~~sh
for cohort in 18 24; do
  python -m scripts.cpu.replay.study_reservation_placement --cohort $cohort --out "outputs/reservation-placement-$cohort.csv"
done
python -m scripts.cpu.replay.study_reservation_placement --cohort 24 --timer-ms 100 --zero-only --out outputs/reservation-placement-timer-24.csv
python -m scripts.cpu.analysis.summarize_reservation_placement outputs/reservation-placement-18.csv outputs/reservation-placement-24.csv outputs/reservation-placement-timer-24.csv
python -m scripts.cpu.replay.study_reservation_placement --cohort 24 --windows 1 --trace qwen_thinking_blksz_16.jsonl --timer-ms 100 --zero-only --out outputs/reservation-timer-witness.csv
for cohort in 18 24; do
  python -m scripts.cpu.replay.study_batch_recovery --cohort $cohort --out "outputs/batch-recovery-$cohort.csv"
  python -m scripts.cpu.replay.study_batch_recovery --cohort $cohort --no-io-contention --out "outputs/batch-no-io-$cohort.csv"
done
python -m scripts.cpu.analysis.summarize_batch_recovery outputs/batch-recovery-18.csv outputs/batch-recovery-24.csv
python -m scripts.cpu.analysis.summarize_batch_recovery outputs/batch-no-io-18.csv outputs/batch-no-io-24.csv
python -m scripts.cpu.replay.study_batch_recovery --cohort 18 --start-window 21 --windows 1 --trace qwen_coder_blksz_16.jsonl --out outputs/batch-attribution-local.csv
python -m scripts.cpu.replay.study_batch_recovery --cohort 18 --windows 1 --trace qwen_coder_blksz_16.jsonl --out outputs/batch-attribution-stress.csv
~~~

~~~sh
python -m scripts.cpu.replay.study_queue_objective --skip 24 --long 8192 --rates 1 32 --out outputs/boundary-queue-objective-8k.csv
python -m scripts.cpu.analysis.summarize_queue_objective outputs/boundary-queue-objective-8k.csv --expected-cases 288
python -m scripts.cpu.replay.study_queue_objective --skip 18 --rates 32 --loss-source docs/evidence/boundary-queue-loss-input.csv --diagnostics outputs/queue-counterfactual.jsonl --repeats 0 --out outputs/queue-selected.csv
~~~

~~~sh
for lanes in shared independent; do
  python -m scripts.cpu.replay.study_queue_objective --skip 24 --long 8192 --lanes $lanes --lane-core --rates 1 32 --out "outputs/lane-objective-24-$lanes.csv"
done
python -m scripts.cpu.replay.study_queue_objective --skip 24 --long 8192 --lanes coupled --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-coupled.csv
python -m scripts.cpu.replay.study_queue_objective --skip 24 --long 8192 --lanes coupled --joint-prefill --lane-core --lane-probes --rates 1 32 --out outputs/lane-objective-24-joint.csv
python -m scripts.cpu.analysis.summarize_lane_objective outputs/lane-objective-24-shared.csv outputs/lane-objective-24-independent.csv outputs/lane-objective-24-coupled.csv outputs/lane-objective-24-joint.csv
~~~

## 双槽计时留出验证

只读已存的三组原始配对：同时提交、150 ms 校准和 300 ms 留出。该实验用于证明简单进度估计的误差，和真实深恢复评估是不同对照。

~~~sh
python -m scripts.gpu.analysis.validate_batch_join docs/evidence/batch-warm.json docs/evidence/batch-stagger-150.json docs/evidence/batch-stagger-300.json
~~~

## GPU 归档区间审计

主报告第 3.23 节的区间审计使用已有 CSV。运行 `python -m scripts.validation.audit_gpu_interval_coverage --check docs/evidence/dynamic-gpu-interval-audit-7870.json` 可复核冻结区间、来源哈希与逐批计数。
