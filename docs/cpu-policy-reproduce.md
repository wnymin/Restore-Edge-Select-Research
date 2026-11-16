# CPU 策略复现与 evidence 归档

用 Python 3.12 或 3.13 从仓库根目录运行。手算验收无需轨迹。百炼回放用锁定版本和校验和，下载方法见[轨迹说明](../data/README.md)。下面使用 POSIX shell，在 PowerShell 中可以直接展开轨迹文件名和策略名。

```sh
python scripts/check_fixture.py
python scripts/fetch_bailian.py
python scripts/fetch_bailian.py --verify-only
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python scripts/study_cpu_oracle.py --out outputs/cpu-oracle-grid.csv

traces="qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl"
estimates="next4 entry mixed-projected-entry mixed-estimate mixed-reuse-estimate"
guards="next4 entry mixed-reuse-estimate reuse-cross-tier-fallback reuse-cross-tier-guard"
python scripts/study_load_routing.py --cohort 84 --windows 3 --trace $traces --max-prompt 6656 --path all --rates 1 8 16 32 --no-snapshots --policies $estimates --out outputs/cpu-estimate-reuse-84.csv
python scripts/study_load_routing.py --cohort 87 --windows 3 --trace $traces --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --path ssd --rates 1 8 16 32 --no-snapshots --overlap-profile configs/overlap-ssd-compute-half-control.json --policies $estimates --out outputs/cpu-estimate-reuse-87-4k.csv
python scripts/study_load_routing.py --cohort 90 --windows 3 --trace $traces --max-prompt 6656 --path all --rates 1 8 16 32 --no-snapshots --policies $guards --out outputs/cpu-reuse-guard-90.csv
python scripts/study_load_routing.py --cohort 93 --windows 3 --trace $traces --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --path ssd --rates 1 8 16 32 --no-snapshots --overlap-profile configs/overlap-ssd-compute-half-control.json --policies $guards --out outputs/cpu-reuse-guard-93-4k.csv
for factor in 1/2 1 2; do
  label=$(printf '%s' "$factor" | tr '/' '-')
  python scripts/study_load_routing.py --cohort 96 --windows 3 --trace $traces --max-prompt 6656 --path ssd --rates 8 16 32 --zero-only --no-snapshots --overlap-profile configs/overlap-disabled-control.json --ssd-estimate-factor "$factor" --policies $guards --out "outputs/cpu-reuse-stale-96-$label.csv"
done
for factor in 1 2; do
  python scripts/study_load_routing.py --cohort 99 --windows 3 --trace $traces --max-prompt 6656 --path ssd --rates 16 32 --zero-only --no-snapshots --overlap-profile configs/overlap-disabled-control.json --ssd-estimate-factor "$factor" --policies next4 entry mixed-reuse-estimate reuse-cross-tier-fallback reuse-prefetch-fallback --out "outputs/cpu-reuse-prefetch-99-$factor.csv"
done

python scripts/study_tail_witness.py --trace qwen_traceA_blksz_16.jsonl --window 77 --placement ssd --spacing 1 --budget 128 --rate 32 --path all --reference-policy mixed-estimate --intervene-policy mixed-projected-entry
python scripts/study_tail_witness.py --trace qwen_coder_blksz_16.jsonl --window 89 --placement split --spacing 1/4 --budget 128 --rate 32 --max-prompt 3584 --grid docs/evidence/prefill-vulkan-4k.json --overlap-profile configs/overlap-ssd-compute-half-control.json
python scripts/study_tail_witness.py --trace qwen_coder_blksz_16.jsonl --window 96 --placement ssd --spacing 1 --budget 128 --rate 32 --overlap-profile configs/overlap-disabled-control.json --ssd-estimate-factor 2 --intervene-policy reuse-cross-tier-guard
```

各输出分别用 `python scripts/summarize_load_routing.py --cases 108 <csv>` 汇总。可使用 `--zero-only` 跳过不确定的本机决策计时。归档的硬件读数和成本配置说明见[实验口径](experiment-index.md)。

| 数据 | 归档 |
| --- | --- |
| 共享直接估计 | [84–86](evidence/cpu-estimate-reuse-84-summary.csv) · [87–89](evidence/cpu-estimate-reuse-87-4k-summary.csv) |
| 选择性跨层检查 | [90–92](evidence/cpu-reuse-guard-90-summary.csv) · [93–95](evidence/cpu-reuse-guard-93-4k-summary.csv) |
| 固定实际值、改变策略估计 | [0.5 倍](evidence/cpu-reuse-stale-96-1-2-summary.csv) · [1 倍](evidence/cpu-reuse-stale-96-1-summary.csv) · [2 倍](evidence/cpu-reuse-stale-96-2-summary.csv) |
| 同层回退留出 | [成本正确](evidence/cpu-reuse-prefetch-99-1-summary.csv) · [成本估高](evidence/cpu-reuse-prefetch-99-2-summary.csv) |
| 小规模穷举 | [48 个控制配置](evidence/cpu-oracle-grid.csv) |
