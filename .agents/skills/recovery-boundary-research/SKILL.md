---
name: recovery-boundary-research
description: Apply the project's research method for mixed-cache recovery-boundary studies, especially when selecting, modeling, evaluating, or omitting dynamic factors under competing HBM prefetch capacity. Use for experiments, strategy design, evidence review, and reports in this repository.
metadata:
  short-description: Prioritize measurable high-impact factors and bound omission loss
---

# Recovery-boundary research

Before acting, read `docs/recovery-research-protocol.md` in the repository. It is the source for the project's scope, fixed semantics, factor-priority method, omission-loss acceptance, evidence/reporting rules, and handoff status. Then read only the task-relevant primary report (`docs/trace-oracle.md`, `docs/dynamic-gpu-lookahead-report.md`, `docs/dynamic-cpu-completion.md`, or `docs/simulator-baseline.md`).

## Non-negotiable method

- Study recovery-boundary selection under fixed cache placement and execution semantics; do not silently turn the task into FIFO/LRU eviction, active-KV allocation, approximate recovery, or scheduler redesign.
- Rank dynamic factors on two separate axes: expected decision impact and measurability/evaluation cost. Start with factors that are both high-impact and easy to quantify. A hard-to-measure factor is not low-impact by default.
- For suspected high-impact but difficult factors, run a discriminating paired experiment, proxy-observation check, or bound analysis before omitting them.
- Omit a factor only after same-input ablation and holdout evidence show the loss is within a declared metric budget over a declared scope. Check critical boundaries, ties, shared/in-flight states, and combinations; do not infer joint safety from independent small losses.
- Preserve feasibility and information constraints even when average-loss budgets allow simplification. Future arrivals, post-hoc completion times, and offline labels are not online features.
- Separate zero-overhead policy quality from total quality including measurement, decision, certificate, and fallback cost. Report average response, decision time, net gain, P95, per-request regressions, and offline-optimal intervals as applicable.
- Distinguish proof, CPU simulation, GPU measurement, and extrapolation. Do not call reduced simulation work an online speedup until certificate construction plus remaining evaluation is measured.

## Handoff

B1-B3 CPU work is accepted and should not be redone as unfinished work. Current open work is general multi-request state sufficiency, real state-error coverage, composable multi-step guarantees, and proving net benefit of simplification. `entry-replay` is an evaluation optimization, not a new boundary-selection policy. The CPU conclusion does not close A-group GPU validation or the project.