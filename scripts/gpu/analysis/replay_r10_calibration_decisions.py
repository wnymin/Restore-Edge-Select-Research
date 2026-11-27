"""Replay two frozen choices at each actually observed calibration snapshot."""

import argparse
from dataclasses import fields
from fractions import Fraction
import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from restore_edge_select.live_snapshot import ObservedDecision, choose_observed
from restore_edge_select.simulator import Experiment
from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints
from scripts.validation.check_r10_native_catalog import digest


def reconstruct(raw):
    times = {f.name for f in fields(ObservedDecision) if f.name.endswith('_ms')}
    o = {key: Fraction(value) if key in times and value is not None else value for key, value in raw.items()}
    o['committed'] = tuple(o['committed'])
    o['active_deps'] = tuple(o['active_deps'])
    o['admitted'] = tuple((request, tuple(deps)) for request, deps in o['admitted'])
    o['ready'] = frozenset(o['ready'])
    return ObservedDecision(**o)


def latency(rows):
    def sample(data):
        data = sorted(data)
        return dict(median_ns=median(data), p95_ns=data[ceil(.95 * len(data)) - 1])

    if not rows:
        return dict(snapshots=0)
    a, b, d, baseline = zip(*rows, strict=True)
    return dict(snapshots=len(rows), local_cost=sample(a), entry_replay=sample(b),
                entry_minus_local_paired=sample(d), live_recorded_local_cost=sample(baseline))


def arm(prefix, schedule, profile, timing_repeats=0):
    p, native, _ = endpoints(prefix)
    recorded = json.loads(p.read_bytes())
    assert recorded['policy'] == 'local-cost' and 'calibration diagnostic' in recorded['scope'] and recorded.get('warmup_grid')
    assert recorded['cost_profile_sha256'] == digest(profile)
    audited = check(p, native, schedule)
    s = json.loads(schedule.read_bytes())
    s['trace']['cost'] = json.loads(profile.read_bytes())['cost']
    exp = Experiment(s['trace'], s['model'], Fraction(recorded['capacity_bytes'], 2**20))
    actual = list(map(Fraction, recorded['observed_arrivals_ms']))
    alternatives, multiple, changes = 0, 0, []
    measurements = []
    for row in recorded['decisions']:
        snapshot = reconstruct(row['snapshot'])
        # Do not expose arrival times or completions of unseen requests.
        n = snapshot.arrived
        exp.arrivals = actual[:n] + [snapshot.now_ms + 1] * (len(actual) - n)
        local = choose_observed(exp, 'local-cost', snapshot)
        candidate = choose_observed(exp, 'entry-replay', snapshot)
        assert local == row['action']
        noptions = len(exp.eligible[snapshot.request])
        alternatives += noptions > 1
        multiple += noptions > 2
        if candidate != local:
            changes.append(dict(request=snapshot.request, local=local, entry=candidate, eligible=list(exp.eligible[snapshot.request]),
                                visible_queue=snapshot.arrived - snapshot.head, active=snapshot.compute_start_ms is not None,
                                in_flight=snapshot.transfer_object is not None))
        if timing_repeats:
            strategies = ('local-cost', 'entry-replay')
            a, b, d = [], [], []
            for rep in range(timing_repeats):
                x = {}
                # Counterbalance call order at each identical, already warmed snapshot.
                for policy in strategies if (snapshot.request + rep) % 2 == 0 else reversed(strategies):
                    start = perf_counter_ns()
                    action = choose_observed(exp, policy, snapshot)
                    x[policy] = perf_counter_ns() - start
                    assert action == (local if policy == 'local-cost' else candidate)
                a.append(x['local-cost'])
                b.append(x['entry-replay'])
                d.append(b[-1] - a[-1])
            measurements.append((noptions, (median(a), median(b), median(d), row['choose_ns'])))
    result = dict(block=recorded['block'], run=prefix.name, audited_requests=audited['requests'], with_alternative=alternatives,
                  with_two_nonzero_alternatives=multiple, changes=changes, receipt_sha256=digest(p))
    if timing_repeats:
        result['host_chooser_benchmark'] = dict(repeats=timing_repeats, all=latency([t for _, t in measurements]),
                                               one_option=latency([t for n, t in measurements if n == 1]),
                                               alternatives=latency([t for n, t in measurements if n > 1]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', action='append', type=Path, nargs=2, required=True, metavar=('PREFIX', 'SCHEDULE'))
    ap.add_argument('--cost-profile', type=Path, required=True)
    ap.add_argument('--chooser-repeats', type=int, default=0, help='even paired repeats of the warmed CPU chooser per real snapshot')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.chooser_repeats not in range(0, 101, 2):
        ap.error('chooser repeats must be even and in [0, 100]')
    rows = [arm(*r, a.cost_profile, a.chooser_repeats) for r in a.arm]
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='counterfactual chooser only, on recorded local-cost states; no runtime effect', arms=rows),
                         sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
