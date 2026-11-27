"""R10 bridge from a visible executor snapshot to the frozen stationary chooser."""

from dataclasses import dataclass
from fractions import Fraction

from .simulator import IO_PATHS, number


@dataclass(frozen=True)
class ObservedDecision:
    now_ms: Fraction
    head: int
    arrived: int
    request: int
    committed: tuple[int, ...]
    admitted: tuple[tuple[int, tuple[str, ...]], ...]
    ready: frozenset[str]
    active_deps: tuple[str, ...] = ()
    compute_start_ms: Fraction | None = None
    suffix_start_ms: Fraction | None = None  # only set once suffix has started
    transfer_object: str | None = None
    transfer_start_ms: Fraction | None = None


def _score(row):
    # Frozen R07 fifo_snapshot scoring rule, specialized to this one mode.
    compute = row["serial_compute_ms"] + row["serial_copy_ms"]
    if not row["missing_bytes"]:
        return compute
    io = row["incremental_ram_ms"] + row["incremental_ssd_ms"]
    slack = row["head_slack_ms"] + row["committed_ahead_compute_ms"]
    inflight = row["inflight_ms"]
    if row["prefetch_fits"]:
        return compute + max(0, io + inflight + row["queue_debt_ms"] - slack)
    return compute + io + max(0, inflight - slack)


def _options(policy):
    if policy == "local-cost":
        return policy, {}
    if policy != "entry-replay":
        raise ValueError("R10 supports only the two frozen main arms")
    return "snapshot-screened", dict(score_candidate=_score, screen_mode="unsettled", queue_horizon=4,
                                     screen_search="entry", cached_costs=True, committed_choices=True, scalar_replay=True)


def _check_scope(exp):
    if exp.dynamic or exp.per_tier or exp.host_cache_bytes or exp.release_at != "start" or not exp._stationary_costs:
        raise ValueError("Live snapshot requires stationary costs, shared serial transfers, and release at service start")


def choose_observed(exp, policy: str, s: ObservedDecision) -> int:
    """Evaluate the original chooser, with no future arrivals or completion times supplied."""
    _check_scope(exp)
    mode, opts = _options(policy)
    now = number(s.now_ms)
    if not 0 <= s.head <= s.request < s.arrived <= len(exp.requests) or exp.arrivals[s.arrived - 1] > now:
        raise ValueError("Invalid visible request range")
    if len(s.committed) != s.request or any(c not in exp.eligible[i] for i, c in enumerate(s.committed)):
        raise ValueError("Missing or ineligible committed action")
    reservations = dict(s.admitted)
    if len(reservations) != len(s.admitted) or any(
            not s.head <= i < s.request or len(set(deps)) != len(deps) or
            set(deps) != set(exp.candidates[i][s.committed[i]]) - exp.initial
            for i, deps in reservations.items()):
        raise ValueError("Reservations must be exact sets of pre-existing committed dependencies")
    admitted = {i: set(objects) for i, objects in reservations.items()}
    if exp.size(set().union(*admitted.values())) > exp.budget or not exp.initial <= s.ready <= exp.objects.keys():
        raise ValueError("Over-budget reservation or invalid ready objects")

    def past(t):
        if t is None:
            return None
        v = number(t)
        if not 0 <= v <= now:
            raise ValueError("Phase start must be observed at or before the decision")
        return v

    run_start, suffix_start, transfer_start = map(past, (s.compute_start_ms, s.suffix_start_ms, s.transfer_start_ms))
    running = run_start is not None
    if running:
        if s.head == s.request or s.active_deps != exp.candidates[s.head][s.committed[s.head]] or (
                suffix_start is not None and suffix_start < run_start):
            raise ValueError("Invalid active head or active dependency identity")
        if suffix_start is None:
            if not s.active_deps:
                raise ValueError("A request without dependencies cannot be copying")
            # Indicates that the suffix has not been submitted. The chooser
            # computes its own residual copy estimate from observed start.
            suffix_start = now + Fraction(1, 1_000_000)
    elif suffix_start is not None:
        raise ValueError("No suffix should be active without an active head")
    if (s.transfer_object is None) != (transfer_start is None) or (
            s.transfer_object is not None and (s.transfer_object not in exp.objects or s.transfer_object in s.ready)):
        raise ValueError("Invalid serial transfer observation")

    # These two now-valued markers encode activity only. The original chooser's
    # stationary branch derives residual work using starts and calibrated costs.
    live_transfer = (now, s.transfer_object) if transfer_start is not None else None
    active = s.active_deps if running else ()
    snapshot = (now, s.head, s.arrived, admitted, set(s.ready), now if running else None, live_transfer, active, set(), None, {})
    return exp._run(mode, 0, None, 0, s.arrived, snapshot=snapshot, forecast_speed=dict.fromkeys(IO_PATHS, 1),
                    decision_cursor=(s.request, s.committed, run_start, suffix_start, transfer_start), **opts)


def audit_simulated_decisions(exp, policy, inspect_observation):
    """Capture normal execution snapshots; callers can compare independent live replay."""
    _check_scope(exp)
    mode, opts = _options(policy)

    def inspect(index, choice, now, head, arrived, selected, admitted, ready, running, deps, compute, suffix, transfer, transfer_start):
        row = ObservedDecision(now, head, arrived, index, selected,
                               tuple((i, tuple(sorted(obj))) for i, obj in sorted(admitted.items())), ready,
                               deps if running else (), compute if running else None,
                               suffix if running and suffix <= now else None, transfer, transfer_start if transfer else None)
        inspect_observation(row, choice)

    return exp._run(mode, 0, None, 0, len(exp.requests), decision_observer=inspect, **opts)
