# R04-CPU-01: F11 Transfer-Cost-Aware Online Boundary Selection

- **Status:** User line delivered; the R04 CPU scope is jointly assessed in [final acceptance](../../r04-final-acceptance.md). R04 as a whole is not claimed to cover GPU/service validation.
- **Question:** Given fixed cache placement and event semantics, how should incremental RAM/SSD transfer cost guide online recovery-boundary selection, and what is guaranteed under a finite search budget?
- **Scope:** Qwen3.5-4B CPU event simulation; one compute slot, serialized transfers, shared finite prefetch capacity, exact checkpoints, and first-entry boundary selection. Results are not real service end-to-end latency.

## Algorithm

For each legal candidate c in the current decision snapshot, aggregate candidate dependencies by object identity. Estimate remaining incremental RAM/SSD transfer, in-flight residual, queued transfer debt, visible and committed-ahead compute window, suffix continuation, and required device copy. Use a low-cost projected score to prioritize the bounded rollout budget:

$$
q(c)=C(c)+K(c)+\mathbf{1}_{M(c)>0}\begin{cases}
\max(0,T(c)+I(c)+Q(c)-W(c)), & \text{prefetch admissible},\\
T(c)+\max(0,I(c)-W(c)), & \text{not currently admissible}.
\end{cases}
$$

Here C is suffix compute, K required device-copy cost, T=T_RAM+T_SSD incremental transfer, I in-flight residual, Q queued transfer debt, W currently visible compute coverage, and M indicates missing dependency bytes. The score only orders candidates; it is not an admissible lower bound. The original event executor handles actual admission, demand loading, shared dependencies, handoff, release, and event ties during candidate forecasts.

The `adaptive-cost-blind` control removes transfer from ordering but keeps exact-cost rollouts. `adaptive-no-transfer` scales decision-side transfer information close to zero in the candidate forecast, priority, and route classification; the selected boundary is still executed with the true modeled cost. These controls separate ordering contribution from decision-time cost information.

## Bounded-regret argument

Fix the decision snapshot, legal candidates, cost table, and continuation rule. Let E_j(c) be the event executor's completion time for visible request j under candidate c, J(c)=sum_j E_j(c), and L_j(c) be the lower bound retaining only that request's required compute work. In the fixed single-compute-slot model, transfer, synchronization, copy, and waiting cannot complete required compute earlier, so L_j(c)<=E_j(c) for every legal candidate.

Let S be exactly evaluated candidates, a the feasible incumbent, and U the candidates not exactly evaluated, not safely pruned, including those left out when the budget is exhausted. Report

$$
G=J(a)-\min\left(J(a),\min_{c\in U}\sum_j L_j(c)\right),
$$

with G=0 if U is empty. Since every legal candidate has J(c)>=sum_j L_j(c), 0<=J(a)-J*<=G under the stated assumptions and a common objective/feasibility rule. Exact optimality is claimed only after every potentially winning candidate is evaluated or safely excluded. Exhausted budgets must expose the remaining floor and cannot be called exact.

This is a simulator-level guarantee for a fixed CPU executor and cost model. It does not calibrate costs or establish accuracy for a real GPU, concurrent service, or changed event semantics.

## Experiment and results

The frozen heldout replay uses archived public trace `qwen_traceB_blksz_16.jsonl`, windows 200-202, cohort 200, Qwen3.5-4B 4K prefill cost grid, SSD transfer path, and matched 512-token checkpoints. Each SSD cost multiplier (8, 11, 12, 16, 32) has 36 configurations: 3 windows x split/RAM/SSD placement x spacing 1/1/4 x 128/384 MiB prefetch budgets. That is 3,360 requests per multiplier. Maximum prompt length is 3,584 tokens. Multipliers are sensitivity inputs, not hardware measurements.

| SSD multiplier | Policy | Zero-overhead mean gain ms/config | Gain after injected decision cost ms/config | Requests regressed >10ms | Maximum request loss ms |
|---:|---|---:|---:|---:|---:|
| 8 | adaptive | 0.00 | -0.20 | 0 | 1.48 |
| 11 | adaptive | 39.08 | 38.60 | 10 | 11.90 |
| 12 | adaptive | 893.15 | 892.38 | 4 | 10.55 |
| 16 | adaptive | 112.49 | 111.81 | 4 | 11.28 |
| 16 | adaptive-no-transfer | 761.43 | 760.13 | 28 | 29.95 |
| 32 | adaptive | 0.00 | -0.04 | 0 | 0.75 |
| 32 | adaptive-no-transfer | 84.01 | 81.85 | 1,072 | 1,727.50 |

Benefits depend strongly on cost scale and configuration; the data do not support a claim that the policy beats all baselines in all conditions. At scale 16, full transfer information avoids the near-blind ablation's long-tail regressions. At scale 32, the full policy falls back to local-cost and has no simulated zero-overhead gain, while the near-blind ablation incurs severe tail losses. This indicates that transfer cost can prevent bad choices in a subset of the modeled domain, not a universal improvement.

The `adaptive-cost-blind` ordering ablation had identical recorded outcomes in all 180 paired configurations; this experiment does not show an independent benefit from the transfer term in ordering alone. The `adaptive-no-transfer` decision-information ablation differed from the full policy in 98/180 configurations. Each scale's 3,360 adaptive decisions reported zero regret gap; a separate 72-configuration audit at scales 16 and 32 checked skipped candidates and safe floor-based exclusion. Decision CPU cost was about 17.4-19.6 ms/configuration in this Python implementation. Injected cost remains a local policy-cost sensitivity estimate, not GPU or end-to-end latency.

## Reproducible artifacts

- [Experiment summary](../../evidence/r04-cpu-01-summary.json)
- [Zero-overhead request results](../../evidence/r04-cpu-01-zero-overhead.csv.gz)
- [Decision-cost-injected request results](../../evidence/r04-cpu-01-decision-overhead-injected.csv.gz)
- [Candidate-floor audit](../../evidence/r04-cpu-01-certificate-audit.csv.gz)
- [SHA256 manifest](../../evidence/r04-cpu-01-manifest.json)
- [Summary regeneration script](../../../scripts/validation/summarize_r04_cpu01.py)
- Implementation: [adaptive selector](../../../src/restore_edge_select/adaptive.py), [event executor](../../../src/restore_edge_select/simulator.py), [trace replay](../../../scripts/cpu/replay/study_load_routing.py), [fixture checks](../../../scripts/check_fixture.py).

## Limitations

Evidence comes from three windows of one public trace under a CPU event simulator. Cost multiplier sensitivity is not measured link performance; the target is the simulator's aggregate completion objective, not real TTFT/TPOT. Search and measurement overhead can erase small gains. The report's conclusions must not be extrapolated to live GPU service without cost calibration and heldout validation.
