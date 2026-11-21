# R05: Static/Dynamic Strategy Regimes and Online Portfolio Switching

- **Status:** CPU scope completed (2026-11-20); primary acceptance passed on frozen holdout. Report: [R05 CPU report](r05-strategy-portfolio-report.md). Preregistration: [R05 preregistration](r05-preregistration.md). Owner: user side / Codex execution. No collaborator split in this round.
- **Purpose:** Identify the static configurations and dynamic decision states in which the existing boundary-selection strategies are best, then use only decision-time observable information to select among them. Deliver one integrated portfolio policy and test whether it beats the strongest fixed baseline on frozen holdout data.
- **Scope:** Fixed cache placement, exact checkpoint semantics, and existing CPU event executor. This is recovery-boundary selection, not eviction, active-KV allocation, or scheduler redesign. GPU/service conclusions are explicitly out of scope unless separately measured.

## Research questions

1. For which static regimes (cost table / RAM-SSD placement / capacity and prefetch budget) and dynamic states (visible queue, remaining capacity, ready/in-flight dependencies, coverage window and queue debt) does each existing policy perform best?
2. Can an online selector identify those regimes using only features available at its decision time, without using future arrivals, completion times, trace labels, or offline oracle labels as runtime inputs?
3. Does the integrated policy improve total objective after selector, forecast, certificate, and fallback costs, while retaining bounded or explicitly reported loss when its regime prediction is wrong?

## Policy portfolio and comparison contract

Candidate policies must be frozen before the holdout comparison. At minimum include `entry`, `local-cost`, and R04 `adaptive`; include another existing policy only if its implementation and semantics are documented. `entry-replay` is an evaluation optimization, not a selectable boundary policy. The integrated policy may either choose one candidate policy or evaluate multiple proposed boundaries with the shared exact executor; it must not silently change execution semantics.

Use two complementary references:

- **Strongest fixed baseline:** On calibration data only, choose the best single existing policy under the declared net objective. Freeze that identity before holdout. On holdout, compare the portfolio against this same fixed policy and also publish each fixed policy's result.
- **Offline strategy oracle:** Per snapshot or configuration, select the best among the frozen candidate policies using fully evaluated outcomes. This is diagnostic upper-bound/reference only, never an online feature. Report oracle gap and the interval where the exact best is not certified.

The R04 `entry` and `local-cost` names must retain their implemented meanings. Do not call exhaustive per-candidate evaluation the baseline unless explicitly labeling it as the offline oracle. Report both per-decision boundary objective and the closed-loop trace outcome; do not infer one from the other.

## Work packages (all owned by user side)

| Task | Work | Dependencies | Deliverables | Acceptance | Status |
| --- | --- | --- | --- | --- | --- |
| `R05-CPU-01` | Inventory and normalize candidate policies, objectives, runtime inputs, and cost accounting; reproduce R04 results and establish paired regime labels/oracle intervals. | R03/R04 accepted artifacts | Strategy registry, reproducible paired dataset, static/dynamic regime definition, oracle and label audit | Inputs and outputs align with existing evidence; labels use only declared candidate set; any non-exact optimum is interval-valued; no future/after-the-fact data enters features. | Accepted: semantics, calibration baseline, runtime-input audit, and oracle comparison delivered |
| `R05-CPU-02` | Map strategy crossover conditions. Separate static cost/layout/budget regimes from dynamic snapshot state; test factors individually and in selected interactions, including near ties, capacity boundaries, shared/in-flight states and queue debt. | CPU-01 | Crossover tables/plots, counterexamples, interpretable conditions and uncertainty | Same-input paired evaluation; calibration/holdout split frozen by trace window/configuration (not rows from one trace); document flip rate, objective delta, tails, and decision cost. | Accepted: crossover audit and ablations delivered on calibration and frozen holdout |
| `R05-CPU-03` | Design the portfolio selector using online-available signals. Start with a transparent threshold/tree/rule list; compare static-only gating, dynamic gating, and always-run fixed policies. Add safe fallback/certification where selector confidence or state validity is inadequate. | CPU-02 | Integrated policy implementation, frozen config and feature/decision audit | Runtime feature audit proves no leakage; fallback is feasible; uncertain/unsupported conditions route to declared conservative policy; all candidates share the exact executor and objective. | Accepted: `portfolio-v4` implemented with conservative exact fallback and leakage audit |
| `R05-CPU-04` | Frozen holdout and robustness evaluation against all fixed policies and the strongest fixed baseline; include decision and certificate overhead, ablations, and explanations of wins/losses. | CPU-03 | Evidence bundle, report, reproduction/validation command, final acceptance | Primary success: positive net improvement over strongest fixed baseline on the predeclared aggregate holdout objective, with uncertainty analysis and no violation of declared per-request/tail guardrails. Also show which observable conditions caused gains and a paired counterfactual/ablation supporting the explanation. If unmet, R05 is a valid negative result and no superiority claim is made. | Accepted: frozen holdout passed; report and machine summary delivered |

## Experimental design and metrics

- Freeze trace windows, seeds, static cost configurations, candidate set, objective, guardrails, and split before fitting thresholds. Reuse R04 trace B only as a reproduction/calibration source; reserve disjoint windows/configurations for holdout. Do not treat many configurations from one trace window as independent production samples.
- Cover both static regimes (RAM/SSD/split placement, transfer/compute cost ratio, prefetch budget) and dynamic states (visible compute coverage, residual capacity, ready/in-flight bytes, shared-object identity, queued debt). Include boundary neighborhoods and selected joint states; independent single-factor results do not establish joint safety.
- Primary objective: sum of visible request response times within the fixed replay window, with explicit accounting for strategy-selection CPU time, forecasts, certificate work, and fallback. Also report zero-overhead policy quality separately.
- Report mean and P95 response, net gain versus strongest fixed policy, per-request regression count and maximum loss, boundary/policy switch rate, decision cost, certified regret gap/oracle interval, and stratified performance by static and dynamic regime. State when a claimed gain is only against `local-cost` and not against the strongest fixed baseline.
- Explanation requirement: for every material aggregate gain, show the regimes/states that triggered switching, paired cases where the selected strategy beats the alternative, and ablation removing the switching signal. Include failures and unnecessary-switch cases. Correlation or a feature importance score alone is not a causal explanation.

## Decision and acceptance rules

1. Success is not ?sometimes beats a baseline.? The portfolio must beat the strongest fixed baseline on the predeclared holdout net objective, with an uncertainty interval supporting positive improvement and within predeclared safety guardrails. Guardrail thresholds are to be set from calibration before holdout; if no defensible threshold is available, report a sensitivity curve rather than choosing one after results.
2. A holdout win without explanation, a gain only before decision overhead, or a gain only against a weaker baseline does not pass the primary criterion.
3. If the portfolio fails, retain the evidence, identify whether failure came from no stable regime, weak signals, selector overhead, switching instability, or baseline strength, and propose a narrower follow-up. Do not tune on the holdout and relabel it as validation.
4. This CPU acceptance cannot establish real GPU/service benefit. Hardware-cost calibration and end-to-end verification remain a separate later task.

## Change log

- 2026-11-20: R05 started; `R05-CPU-01` began with policy/objective inventory and paired audit of archived R04 replay outputs. Existing R04 Trace B results are reproduction/calibration evidence, not new holdout validation.
- 2026-11-20: R05 planned after R04 acceptance. Single-owner scope confirmed by user; no parallel collaborator split. Primary success criterion explicitly set against the strongest fixed baseline, including decision overhead, with interpretable attribution.
