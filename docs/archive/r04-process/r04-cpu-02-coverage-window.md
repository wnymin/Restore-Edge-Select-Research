# R04-CPU-02: F08/F10 Coverage-Window-Aware Online Boundary Selection

- **Status:** Implemented after user takeover on 2026-11-20. No collaborator branch or reproducible handoff artifact was available for review; independent replication was not performed.
- **Question:** Under the fixed single-compute-slot, shared-prefetch-capacity event model, how should observable compute coverage affect candidate exploration, and what remains guaranteed when the search budget is limited?
- **Scope:** F08/F10 inputs from the decision snapshot: head remaining compute `head_slack_ms` and committed-ahead compute `committed_ahead_compute_ms`. No future arrivals, post-hoc completion times, or trace labels enter the policy.

## Algorithm

For candidate c, let T(c)=T_RAM(c)+T_SSD(c) be incremental transfer time, I(c) in-flight residual, Q(c) queued transfer debt, W(c)=H(c)+A(c) observable compute coverage, M(c) indicate missing dependency bytes, F(c) indicate prefetch admission, and C(c), K(c) be suffix compute and required device-copy costs. The search priority is:

$$
q_W(c)=C(c)+K(c)+\mathbf{1}_{M(c)>0}
\begin{cases}
\max(0,T(c)+I(c)+Q(c)-W(c)), & F(c),\\
T(c)+\max(0,I(c)-W(c)), & \neg F(c).
\end{cases}
$$

Candidates are ordered by (q_W(c), L(c), -c), where L(c) is the executor-provided certified compute floor. The score only orders a bounded set of full rollouts; it is not a pruning certificate. Candidates that cannot prefetch remain feasible and preserve head demand-loading. The original executor handles object-identity deduplication, ready/shared/in-flight dependencies, handoff, release, and event ties during exact candidate forecasts.

## Exact coverage criterion for the two-request subdomain

Suppose the running head has $W$ ms of compute left at the decision, the target is next with no followers, and there is no queued or in-flight work, shared handoff, delayed admission retry, or overlap slowdown. Assume the modeled transfer and copy of each candidate match the executor costs. If the target can reserve its dependencies now, they finish after the serial transfer duration $T(c)$ and the target starts at $\max(W,T(c))$ relative to the decision. If it cannot reserve before becoming head, it must wait $W+T(c)$. Adding suffix compute and copy gives target completion equal to head completion plus $q_W(c)$. Head completion is constant across the candidates, so sorting by $(q_W(c),-c)$ gives the exact window objective order, including ties. The fixed-snapshot experiment checks the relation for all three legal boundaries at each capacity and arrival setting.

## Guarantee for bounded search

For each legal candidate c in a fixed snapshot, the exact event executor predicts the completion vector E(c) of visible live requests; the objective is J(c)=sum_j E_j(c). Let L_j(c) be the per-request lower bound retaining only required compute work. In the declared single-compute-slot model, transfer, copy, synchronization, and waiting cannot make required compute finish earlier, so L_j(c)<=E_j(c). This argument does not depend on q_W: even a mistaken coverage ranking cannot make an unverified candidate win.

For feasible incumbent a and unverified, non-pruned set U, report

$$
G=J(a)-\min\left(J(a),\min_{c\in U}\sum_j L_j(c)\right),
$$

with G=0 when U is empty. Given valid floors and consistent gain/per-request-loss checks, 0<=J(a)-J*<=G. An exact optimum claim (G=0) requires that U be empty or every unverified candidate be safely excluded by its floor. With nonzero forecast-error tolerance, the implementation expands thresholds for adverse error on both baseline and challenger.

Thus the theorem guarantees feasibility and bounded regret for budgeted search, not universally correct ordering by q_W. ?Accounting for the factor? is operationally testable: coverage is an online feature capable of changing search order/scenario budget; exact evaluation and G expose residual loss when the heuristic misses. This proof does not automatically extend to inaccurate coverage estimates, concurrent compute engines, or changed event semantics. The current exact forecasts retain actual coverage behavior even in the `adaptive-no-window` control; the control ablates selection-time ordering/classification, not the physical masking mechanism in the executor.

## Evidence

### Fixed-snapshot crossover

`check_fixture.py` contains a paired two-challenger snapshot. Candidate 1 needs 10 ms compute plus 10 ms incremental transfer and has 10 ms visible coverage; candidate 2 needs 15 ms compute with no incremental transfer. Holding other inputs fixed, reducing candidate 1's coverage from 10 ms to zero reverses its priority relative to candidate 2. At the real executor snapshot with equal candidate completion times, candidate 576 is a tied completion with candidate 1152 and the longer-boundary rule chooses 1152; moving the arrival 0.75 ms earlier breaks the tie in favor of 576. At a 70 MiB budget, the actual selector records a different priority winner at the first snapshot when coverage is masked, but both variants still choose 576 after exact forecasting. This confirms the feature changes ordering at the tie boundary, but not the final action. The equal-time ordering is checked through the repository fixture and the 8-snapshot script.

The fixture's outcomes are explicit toy forecasts and establish the ranking/certificate interface, not trace performance. The actual-executor sweep keeps head completion fixed at 30 ms and varies target arrival around the 26.75 ms arrival threshold for equal completion. It covers 70/128 MiB capacity and eight states. For every candidate, exact target completion equals head completion plus the computed candidate priority in this isolated, no-queue state. Other fixture checks exercise capacity, sharing, event-clock, transfer, and lower-bound invariants.

### Frozen trace holdout and ablation

The replay uses archived public trace `qwen_traceB_blksz_16.jsonl`, windows 200-202, cohort 200, Qwen3.5-4B 4K prefill grid, SSD path, and matched 512-token checkpoints. Each I/O scale (8, 11, 12, 16, 32) has 36 configurations: 3 windows x split/RAM/SSD placement x spacing 1/1/4 x 128/384 MiB budget. That is 3,360 requests per scale and 16,800 per policy. `adaptive-no-window` masks W in priority and scenario classification while preserving candidates, floors, true execution costs, and exact event forecasts.

| I/O scale | Policy | Zero-overhead mean gain ms/config | Gain after injected decision cost ms/config | Boundary changes | Requests regressed >10ms after injection |
|---:|---|---:|---:|---:|---:|
| 8 | adaptive | 0.00 | -0.23 | 0 | 0 |
| 11 | adaptive | 39.08 | 38.66 | 20 | 9 |
| 12 | adaptive | 893.15 | 892.38 | 440 | 4 |
| 16 | adaptive | 112.49 | 111.79 | 68 | 4 |
| 32 | adaptive | 0.00 | -0.03 | 0 | 0 |

Important negative result: across the 180 paired Trace B configurations, window ablation produced identical complete boundary sequences and completion sequences (SHA-256 comparison), as well as mean gains. Only at scale 32 did replay counts differ in 12 configurations (1840 for ablation vs 1880 full, i.e. fewer for the ablation). On the separate 648-configuration archived regression (trace windows 109?111 across Coder/Trace A/Trace B), the boundary and completion sequences were also identical. Therefore, this bounded-lookahead implementation did not convert coverage-aware ordering into a changed final action in these trace samples; the fixture verifies its influence on candidate ordering near a real-executor tie, not a trace-level service improvement. The trace does not justify extra routine decision overhead based on this factor alone. The fixed crossover establishes a budget-limited mechanism; this trace shows that mechanism did not translate into a selection gain here. Preserve the negative result: the factor is retained because R03 and the crossover establish a consequential subdomain, not because universal trace benefit has been demonstrated. Scales are simulation sensitivities, not measured hardware values. Decision-cost injection measures this Python policy locally, not GPU or end-to-end service latency.

## Reproduction

```powershell
python scripts/check_fixture.py
python -m scripts.validation.check_r04_snapshots
python -m scripts.cpu.replay.study_load_routing --cohort 200 --windows 3 --trace qwen_traceB_blksz_16.jsonl --path ssd --rates 8 11 12 16 32 --max-prompt 3584 --all-checkpoints --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --zero-only --policies next4 entry adaptive adaptive-cost-blind adaptive-no-transfer adaptive-no-window --fingerprints --out outputs/r04-cpu-02-traceB-200-zero.csv
python -m scripts.cpu.replay.study_load_routing --cohort 200 --windows 3 --trace qwen_traceB_blksz_16.jsonl --path ssd --rates 8 11 12 16 32 --max-prompt 3584 --all-checkpoints --grid docs/evidence/prefill-vulkan-4k.json --request-metrics --no-snapshots --policies next4 entry mixed-reuse-estimate adaptive adaptive-no-window --fingerprints --out outputs/r04-cpu-02-traceB-200-timed.csv
python -m scripts.cpu.replay.study_load_routing --cohort 109 --windows 3 --trace qwen_coder_blksz_16.jsonl qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl --path ssd --rates 1 8 11 12 16 32 --max-prompt 3584 --request-metrics --no-snapshots --fingerprints --zero-only --policies next4 entry mixed-reuse-estimate adaptive-audit adaptive-no-window --out outputs/r04-regression-109.csv
python scripts/validation/summarize_r04_cpu02.py --zero outputs/r04-cpu-02-traceB-200-zero.csv --timed outputs/r04-cpu-02-traceB-200-timed.csv --regression outputs/r04-regression-109.csv --out docs/evidence/r04-cpu-02-summary.json
```

- [Summary](../../evidence/r04-cpu-02-summary.json)
- [Fixed-snapshot crossover and F13 queue-debt sweeps](../../evidence/r04-fixed-snapshots.json)
- [Archived cohort-109 heldout regression, sequence hashes and audit](../../evidence/r04-regression-109.csv.gz)
- [Zero-overhead per-configuration results](../../evidence/r04-cpu-02-zero-overhead.csv.gz)
- [Decision-cost-injected per-configuration results](../../evidence/r04-cpu-02-decision-overhead-injected.csv.gz)
- [SHA256 manifest](../../evidence/r04-cpu-02-manifest.json)
- [Reproducible snapshot checker](../../../scripts/validation/check_r04_snapshots.py)
- [Summary regeneration script](../../../scripts/validation/summarize_r04_cpu02.py)
- Implementation: [adaptive config, classification, priority and bounded search](../../../src/restore_edge_select/adaptive.py), [event state and exact candidate forecasts](../../../src/restore_edge_select/simulator.py), [trace replay](../../../scripts/cpu/replay/study_load_routing.py), [fixture verification](../../../scripts/check_fixture.py).

## Conclusion and limitations

R04-CPU-02 now has an executable coverage-aware online ordering strategy, online-only inputs, conservative compute floors, and exact candidate verification. A fixed crossover confirms that the factor can affect budget-limited exploration; the trace ablation is negative for outcome gains. Applicability is limited to the declared fixed-placement, single-slot CPU event model. Adoption beyond it requires paired holdout across traces and coverage-estimation error regimes, including total decision cost.

Queue-debt-only experiment: 56 actual-executor snapshots vary capacity (70/128/160/192 MiB), preceding boundary, head-compute length, and arrival. Positive debt occurs in 32 snapshots. Removing only the queue-debt term changes the candidate score in 3/56 snapshots and changes the action or bound in 0/56; the maximum oracle regret in these fixtures is 0.75 ms for both policies. This does not establish an incremental policy benefit from F13; the direct condition was exercised and its negative result is retained.
