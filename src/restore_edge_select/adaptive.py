"""Budgeted boundary search with a certificate for the visible FIFO window."""

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
    forecast_error_ms: Fraction = Fraction(0)

    def __post_init__(self):
        if (type(self.later) is not int or self.later < 0
                or type(self.light_replays) is not int or self.light_replays < 1
                or type(self.coupled_replays) is not int or self.coupled_replays < self.light_replays
                or (self.loss_ms is not None and self.loss_ms < 0)
                or min(self.gain_per_request_ms, self.forecast_error_ms) < 0):
            raise ValueError("Invalid adaptive window, replay budget, or tolerance")


def classify(rows, base, budget):
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
        io = r['incremental_ram_ms'] + r['incremental_ssd_ms']
        slack = r['head_slack_ms'] + r['committed_ahead_compute_ms']
        if (r['prefetch_fits'] and io > 0 and slack > 0
                and io + r['queue_debt_ms'] + r['inflight_ms'] >= max(
                    0, rows[0]['serial_compute_ms'] - r['serial_compute_ms'])):
            return 'overlap'
    if rows[base]['queue_debt_ms'] or rows[base]['inflight_ms']:
        return 'backlog'
    return 'light'


def select(rows, base, budget, config, lower, forecast, rank):
    """Return a feasible boundary and an upper bound on its nominal regret.

    Floors and forecasts are alive-request completion vectors, in FIFO order.
    The gain threshold is against the same fixed continuation as the baseline.
    """
    scenario = classify(rows, base, budget)
    signal = (scenario in {'capacity', 'shared', 'overlap', 'backlog'}
              or any(rank[c] < rank[base] for c in rows))
    if config.screen and not signal:
        missed = None
        if config.audit:
            _, _, gap, gain, _ = select(rows, base, budget, AdaptiveConfig(
                config.later, len(rows), len(rows), config.loss_ms,
                config.gain_per_request_ms, audit=True, forecast_error_ms=config.forecast_error_ms),
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

    order = sorted((c for c in rows if c != base), key=lambda c: (rank[c], sum(lower[c]), -c))
    for c in order:
        floor = lower[c]
        optimistic = sum(floor)
        if (optimistic >= base_sum - gain_floor or optimistic >= best_sum
                or exceeds_loss(floor)):
            continue
        if len(seen) >= limit:
            unknown.append(optimistic)
            continue
        ends = seen[c] = tuple(forecast(c))
        if feasible(ends) and sum(ends) < best_sum:
            best, best_sum = c, sum(ends)
    gap = best_sum - min(best_sum, *unknown) if unknown else 0
    if config.audit:
        for c in rows:
            if c not in seen:
                seen[c] = tuple(forecast(c))
        assert all(all(lo <= t for lo, t in zip(lower[c], ends, strict=True))
                   for c, ends in seen.items()), (lower, seen)
        optimum = min([base_sum, *(sum(v) for v in seen.values() if feasible(v))])
        assert 0 <= best_sum - optimum <= gap, (scenario, best_sum, optimum, gap)
    return best, scenario, gap, base_sum - best_sum, None
