# R05-CPU-01: Candidate Policy and Objective Inventory (initial audit)

- **Status:** In progress; this is the opening audit, not task acceptance.
- **Audit date:** 2026-11-20.
- **Inputs:** R04 replay implementation and archived `outputs/r04-cpu-02-traceB-200-timed.csv` (180 paired configurations; five I/O scales, 36 configurations each).
- **Purpose:** Make policy semantics and comparison baselines explicit before assigning any policy a regime or using outcomes to design a selector.

## Policy registry (implementation-level)

| Name | Role in current replay | Decision semantics / information | Notes for R05 |
| --- | --- | --- | --- |
| `local-cost` | Primary direct baseline and per-configuration reference | `Experiment.run("local-cost")`; the replay computes this once as `fixture` and reports each policy's response-time gain against it. | Do not conflate it with `entry` or the offline oracle. |
| `entry` | Fixed-search comparator | In `study_load_routing.py`, it belongs to `EXACT_SEARCH`; uses `screen_search="entry"` with the common four-request queue horizon. | Not an exhaustive offline optimum. |
| `adaptive` | R04 bounded candidate search | Uses `AdaptiveConfig`; orders candidates with online transfer/coverage inputs, performs bounded forecasts with the shared executor, and reports a regret certificate. | A strategy candidate, not a universal winner. |
| `mixed-reuse-estimate` | Lower-cost routed comparator | Reuses the snapshot pressure view and the `estimate` search mode, with ready-reuse enabled. | Existing policy comparator; include in R05 portfolio only after documenting its exact route and supported states. |
| `next4` | R04 orchestration/reference path | Uses the unsettled score with queue horizon four. | Preserve as a diagnostic fixed policy if needed; not interchangeable with `entry`. |

These labels describe the current R04 runner. Before R05 builds a policy router, verify each policy's exact output and runtime feature availability in both the runner and simulator; names alone are not semantic specifications.

## Baseline and objective audit

R04 timed CSV `timed_gain_ms` is computed as the injected-cost `local-cost` response mean minus the corresponding policy response mean. Positive values therefore mean an improvement over `local-cost`, after local decision-time injection. This is not the same as a pairwise win over `entry`. The closed-loop objective is visible-request response-time aggregation in the fixed replay window. Per-request losses and tail metrics are separate guardrails.

R05 therefore keeps three distinct references: (1) every fixed policy, including `local-cost`; (2) the strongest fixed policy selected on calibration only and frozen before holdout; and (3) a per-snapshot offline policy oracle, diagnostic only. The existing R04 Trace B output is reused as reproduction/calibration evidence and cannot be presented as R05's independent holdout.

## R04 paired reproduction summary (decision-cost-injected gain vs `local-cost`)

| I/O scale | Policy | Mean gain (ms/config) | Configs with positive gain / 36 |
| ---: | --- | ---: | ---: |
| 8 | `adaptive` | -0.229 | 12 |
| 8 | `entry` | -0.088 | 7 |
| 8 | `mixed-reuse-estimate` | -0.039 | 9 |
| 11 | `adaptive` | 38.657 | 18 |
| 11 | `entry` | 38.928 | 13 |
| 11 | `mixed-reuse-estimate` | 24.514 | 10 |
| 12 | `adaptive` | 892.381 | 30 |
| 12 | `entry` | 892.543 | 28 |
| 12 | `mixed-reuse-estimate` | 737.446 | 21 |
| 16 | `adaptive` | 111.790 | 26 |
| 16 | `entry` | 766.965 | 28 |
| 16 | `mixed-reuse-estimate` | 640.133 | 18 |
| 32 | `adaptive` | -0.027 | 18 |
| 32 | `entry` | 539.274 | 27 |
| 32 | `mixed-reuse-estimate` | 527.557 | 26 |

Machine-readable recomputation: [R04 paired summary](evidence/r05-cpu-01-r04-reproduction.json), produced by `python scripts/validation/summarize_r05_cpu01.py`. These are per-policy aggregate gains against `local-cost`; positive-gain counts are not policy-vs-policy win counts. Across all 36 paired configurations per scale, `adaptive` has higher injected-cost gain than `entry` in 15/36, 13/36, 17/36, 8/36, and 7/36 configurations at scales 8, 11, 12, 16, and 32, respectively. Thus the high mean gains at scales 11-16 mostly reflect that several policies improve on `local-cost`; they do not establish that `adaptive` beats `entry`. The strong `entry` aggregates at scales 16 and 32 make a static-policy portfolio a plausible hypothesis, but do not yet prove a stable online regime rule.

## Open items before CPU-01 acceptance

- Re-run the summarized aggregates from a committed, bounded script and compare to the archived R04 summary/manifests.
- Register exact candidate semantics, online features and per-decision overhead fields; identify unsupported states and conservative fallback.
- Build an oracle label table from same-input candidate outcomes, preserving ties/intervals and distinguishing per-decision from closed-loop objectives.
- Freeze an R05 calibration/holdout split that does not reuse R04's Trace B 200-202 as independent validation.
