"""A pre-decision gate for the single-slot FIFO simulator."""

from dataclasses import dataclass
from fractions import Fraction


def static_cost(row):
    return row["serial_compute_ms"] + row["serial_copy_ms"] + row["serial_io_ms"]


def projected_cost(row):
    """Estimate the target's work plus transfer not hidden by preceding compute."""
    io = row["incremental_ram_ms"] + row["incremental_ssd_ms"]
    slack = row["head_slack_ms"] + row["committed_ahead_compute_ms"]
    debt = row["inflight_ms"] + row["queue_debt_ms"]
    wait = max(0, io + debt - slack) if row["prefetch_fits"] else io + max(0, debt - slack)
    return (row["serial_compute_ms"] + row["serial_copy_ms"]
            + (wait if row["missing_bytes"] else 0))


@dataclass(frozen=True)
class StateGate:
    """Route to local cost, a fully ready alternative, or bounded window search."""

    margin_ms: Fraction
    use_state: bool = True
    check_blocked_baseline: bool = False
    require_io_margin: bool = False

    def __call__(self, rows, base, budget):
        original = rows[base]
        alternatives = {c: row for c, row in rows.items() if c != base}
        if not alternatives:
            return "gate-local", base
        # A static rival qualifies if its transfer-free cost can beat the
        # baseline and the two serial costs are within one transfer span.
        if not self.use_state:
            worthwhile = any(
                row["serial_compute_ms"] + row["serial_copy_ms"] + self.margin_ms < static_cost(original)
                and abs(static_cost(row) - static_cost(original))
                <= max(row["serial_io_ms"], original["serial_io_ms"])
                for row in alternatives.values()
            )
            return ("gate-search", None) if worthwhile else ("gate-local", base)

        scores = {c: projected_cost(row) for c, row in rows.items()}
        rival = min(alternatives, key=lambda c: (scores[c], -c))
        challenger = rows[rival]
        gain = scores[base] - scores[rival]
        if (gain > self.margin_ms and rival > base and challenger["prefetch_fits"]
                and not challenger["missing_bytes"] and not challenger["extra_reserve_bytes"]):
            return "gate-direct", rival

        # A cheaper target can still hurt peers. Search when the snapshot
        # predicts a win or pressure makes the direct estimate unreliable.
        pressure = any(
            row["head_distance"] and (
                0 < row["extra_reserve_bytes"] <= budget and not row["prefetch_fits"]
                or row["shared_reservation_bytes"]
                or (row["inflight_ms"] or row["queue_debt_ms"]) and row["missing_bytes"]
            )
            and static_cost(original) + self.margin_ms
            > row["serial_compute_ms"] + row["serial_copy_ms"]
            for row in alternatives.values()
        )
        if gain > self.margin_ms or pressure:
            return "gate-search", None
        # A target-only estimate can favor recovery even when its on-demand
        # I/O consumes the prefetch opportunity of already-arrived followers.
        if (self.check_blocked_baseline and base and original["visible_after"]
                and original["head_distance"] and not original["prefetch_fits"]
                and original["missing_bytes"] and 0 < original["extra_reserve_bytes"] <= budget):
            if (not self.require_io_margin or original["incremental_ram_ms"] + original["incremental_ssd_ms"]
                    > max(self.margin_ms, -gain)):
                return "gate-blocked-baseline", None
        return "gate-local", base
