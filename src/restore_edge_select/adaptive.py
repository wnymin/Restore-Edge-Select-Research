"""Budgeted boundary search with a certificate for the visible FIFO window."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class AdaptiveConfig:
    later: int = 4
    light_replays: int = 2
    coupled_replays: int = 3
    loss_ms: Fraction | None = Fraction(10)
    gain_per_request_ms: Fraction = Fraction(1)
    audit: bool = False
    screen: bool = False
    transfer_aware_order: bool = True
    coverage_aware_order: bool = True
    decision_transfer_scale: Fraction = Fraction(1)
    forecast_error_ms: Fraction = Fraction(0)
    route: Callable | None = None
    compact: bool = False

    def __post_init__(self):
        if (type(self.later) is not int or self.later < 0
                or type(self.light_replays) is not int or self.light_replays < 1
                or type(self.coupled_replays) is not int or self.coupled_replays < self.light_replays
                or (self.loss_ms is not None and self.loss_ms < 0)
                or type(self.transfer_aware_order) is not bool
                or type(self.coverage_aware_order) is not bool
                or self.decision_transfer_scale <= 0
                or min(self.gain_per_request_ms, self.forecast_error_ms) < 0
                or (self.route is not None and (not callable(self.route) or self.audit or self.screen))
                or (self.compact and (self.route is None or self.loss_ms is not None))):
            raise ValueError("Invalid adaptive window, replay budget, or tolerance")


def projected_priority(row, *, include_transfer=True, transfer_scale=Fraction(1),
                       include_coverage=True, include_queue_debt=True):
    """Cheap search-order score; never a feasibility or pruning certificate."""
    io = ((row["incremental_ram_ms"] + row["incremental_ssd_ms"]) * transfer_scale
          if include_transfer else 0)
    slack = (row["head_slack_ms"] + row["committed_ahead_compute_ms"]
             if include_coverage else 0)
    debt = row["inflight_ms"] + (row["queue_debt_ms"] if include_queue_debt else 0)
    if row["prefetch_fits"]:
        wait = max(0, io + debt - slack)
    else:
        wait = io + max(0, row["inflight_ms"] - slack)
    return row["serial_compute_ms"] + row["serial_copy_ms"] + (wait if row["missing_bytes"] else 0)


def classify(rows, base, budget, *, transfer_scale=Fraction(1), include_coverage=True):
    """Classify on the current snapshot; never use an I/O stress multiplier."""
    candidates = [r for c, r in rows.items() if c]
    if any(r['head_distance'] and not r['prefetch_fits']
           and 0 < r['extra_reserve_bytes'] <= budget for r in candidates):
        return 'capacity'
    if any(r['prefetch_fits'] and (r['ready_bytes'] or r['shared_reservation_bytes']
                                   or r['active_new_kv_bytes']) for r in candidates):
        return 'shared'
    if len({r['checkpoint_tier'] for r in candidates}) > 1:
        return 'mixed-tier'
    for r in candidates:
        io = (r['incremental_ram_ms'] + r['incremental_ssd_ms']) * transfer_scale
        slack = (r['head_slack_ms'] + r['committed_ahead_compute_ms']
                 if include_coverage else 0)
        if (r['prefetch_fits'] and io > 0 and slack > 0
                and io + r['queue_debt_ms'] + r['inflight_ms'] >= max(
                    0, rows[0]['serial_compute_ms'] - r['serial_compute_ms'])):
            return 'overlap'
    if rows[base]['queue_debt_ms'] or rows[base]['inflight_ms']:
        return 'backlog'
    return 'light'


def select(rows, base, budget, config, lower, forecast, rank, *, lower_sums=None):
    """Return a feasible boundary and an upper bound on its nominal regret.

    Floors and forecasts are alive-request completion vectors, in FIFO order.
    Mean-only compact routing can supply the sum of each floor instead.
    The gain threshold is against the same fixed continuation as the baseline.
    """
    scenario = classify(rows, base, budget, transfer_scale=config.decision_transfer_scale,
                         include_coverage=config.coverage_aware_order)
    if config.screen and not (scenario in {'capacity', 'shared', 'overlap', 'backlog'}
                              or any(rank[c] < rank[base] for c in rows)):
        missed = None
        if config.audit:
            _, _, gap, gain, _ = select(rows, base, budget, AdaptiveConfig(
                config.later, len(rows), len(rows), config.loss_ms,
                config.gain_per_request_ms, audit=True,
                transfer_aware_order=config.transfer_aware_order,
                coverage_aware_order=config.coverage_aware_order,
                decision_transfer_scale=config.decision_transfer_scale,
                forecast_error_ms=config.forecast_error_ms),
                lower, forecast, rank)
            assert gap == 0
            missed = gain
        return base, 'screened-' + scenario, None, 0, missed
    limit = config.light_replays if scenario == 'light' else config.coupled_replays
    baseline = tuple(forecast(base))
    base_sum = sum(baseline)
    # Both the challenger and baseline may err in the adverse direction.
    error = 2 * config.forecast_error_ms
    gain_floor = len(baseline) * (config.gain_per_request_ms + error)
    ceilings = None if config.loss_ms is None else tuple(t + config.loss_ms - error for t in baseline)
    best, best_sum = base, base_sum
    unknown = []
    seen = {base: baseline}

    def exceeds_loss(ends):
        return ceilings is not None and any(x > y for x, y in zip(ends, ceilings, strict=True))

    def feasible(ends):
        return sum(ends) < base_sum - gain_floor and not exceeds_loss(ends)

    def total_floor(c):
        return lower_sums[c] if lower_sums is not None else sum(lower[c])

    order = sorted((c for c in rows if c != base), key=lambda c: (rank[c], total_floor(c), -c))
    for c in order:
        optimistic = total_floor(c)
        if (optimistic >= base_sum - gain_floor or (optimistic, -c) >= (best_sum, -best)
                or (ceilings is not None and exceeds_loss(lower[c]))):
            continue
        if len(seen) >= limit:
            unknown.append(optimistic)
            continue
        ends = seen[c] = tuple(forecast(c))
        if feasible(ends) and (sum(ends), -c) < (best_sum, -best):
            best, best_sum = c, sum(ends)
    gap = best_sum - min(best_sum, *unknown) if unknown else 0
    if config.audit:
        for c in rows:
            if c not in seen:
                seen[c] = tuple(forecast(c))
        assert all(all(lo <= t for lo, t in zip(lower[c], ends, strict=True))
                   for c, ends in seen.items()), (lower, seen)
        optimum = min([(base_sum, -base), *((sum(v), -c) for c, v in seen.items() if feasible(v))])
        assert 0 <= best_sum - optimum[0] <= gap, (scenario, best_sum, optimum, gap)
        if limit >= len(rows):
            assert (best_sum, -best) == optimum
    return best, scenario, gap, base_sum - best_sum, None
