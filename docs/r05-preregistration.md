# R05 preregistration: strategy portfolio holdout

- Frozen: 2026-11-20, before any run on Trace B windows 300-311.
- Owner: user side / Codex only; no collaborator split.
- Scope: CPU replay, fixed placements and exact checkpoint/executor semantics only.

## Split and configuration

Calibration is archived R04 Trace B cohort 200-202 and the existing R05-CPU-01 summary; these have already been inspected and will not be counted as holdout. Holdout is Trace B windows 350-361 (cohort 350, 12 consecutive 352-event windows), disjoint by trace window. If the trace has fewer than 312 complete windows, stop and amend this preregistration before inspecting or substituting any outcomes. Static grid: placements split/RAM/SSD, spacing 1 and 1/4, budgets 128 and 384, SSD transfer scales 1, 8, 11, 12, 16, 32, max prompt 3584. Use the archived R04 cost grid, I/O profile, overlap profile, and model/fixture unchanged. One trace means inference is limited to these windows and this simulator; windows/configurations are not interchangeable with independent production workloads.

## Candidates and selector

Fixed candidates: local-cost, entry, mixed-reuse-estimate, adaptive, adaptive-screened, adaptive-mean. Preserve their existing implementation semantics. The online portfolio may use only cost/layout/budget configuration and decision-time snapshot features (queue pressure/debt, visible coverage, ready/in-flight state and shared identity) already available to the runner. It may not use future arrivals, realized completion/response, trace/window/cohort identifiers, policy outcome labels, or the offline oracle. Candidate outputs are compared under the same exact executor. Any policy search/certificate overhead is included using measured decision CPU/injected cost; zero-overhead replay is secondary.

The strongest fixed baseline is selected using calibration timed net response objective only, from the fixed candidates above, then frozen and written to the run metadata before holdout execution. The portfolio rule must be specified from calibration alone. Per-decision best candidate is an offline diagnostic oracle, never a runtime signal.

## Outcomes and decision rule

Primary outcome: paired reduction in the sum/mean of request response times over all holdout configurations, including policy decision overhead, versus the frozen strongest fixed baseline. Aggregate by window clusters for uncertainty; do not bootstrap individual requests/configurations as independent. Report per-window and per-static-regime results, zero-overhead result, decision overhead, P95 response, requests regressed by >10 ms, maximum request regression, switches/fallbacks, and offline-oracle gap.

Pass only if the portfolio's net improvement is positive and the one-sided 95% cluster-bootstrap lower confidence bound is above zero, with no >10 ms worsening in aggregate P95 versus baseline and no increase in the count of requests regressed by >10 ms. Also show ablation removing each switching signal and paired cases demonstrating the claimed reason for switching. If bootstrap is unstable with 12 clusters, report exact per-window signs and interval limitations; do not substitute configuration-level pseudoreplication. Any failed criterion is a negative result, not permission to tune on holdout.

## Deviations

No deviations permitted after holdout outcomes are inspected. If implementation or data prevents the registered run, document the failure first and register a dated amendment before replacement runs.

## Amendment 2026-11-20 (before any R05 holdout run)

The existing calibration audit shows `next4` is a documented executable fixed
policy. Its timed aggregate is 447.395 ms/config gain versus `local-cost`,
compared with 447.524 ms/config for `entry`; higher gain is better, so `entry`
currently leads by 0.130 ms/config among these two policies. `next4` is added to
the fixed candidate set for calibration-only baseline selection. The strongest
baseline will be the policy with maximum mean `timed_gain_ms`, with ties resolved
lexicographically, and the measured identity will be frozen before holdout.
This corrects an omitted eligible candidate; it is not a response to holdout
outcomes.

## Amendment 2026-11-20: exclude pilot window from holdout

A one-window implementation pilot was inadvertently executed and inspected on Trace B window 300 before the registered full holdout. Window 300 and the entire neighboring cohort 300-311 are therefore excluded from confirmatory evidence and may not be used for tuning. The untouched confirmatory holdout is moved to windows 350-361. No outcomes from windows 301-349 or 350 onward were inspected when this amendment was recorded. All other registered configurations, candidates, rules, metrics, and acceptance thresholds remain unchanged. The pilot file was temporary implementation/debug evidence only and was removed during repository cleanup; it is not part of the confirmatory result.
