"""Report the before/after machine snapshots of closed primary pairs, without effect metrics."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median

from scripts.gpu.analysis.inspect_r10_primary_progress import inspect, read


GUID = re.compile(r'\b[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}\b', re.I)
GPU_COLS = ('index', 'gpu_id', 'name', 'temp_c', 'graphics_mhz', 'memory_mhz', 'watts', 'pstate', 'memory_mib', 'util_pct')
METRICS = ('temp_c', 'graphics_mhz', 'memory_mhz', 'watts', 'memory_mib', 'util_pct')


def snapshot(raw):
    gpu_info, plan_info = (raw.get(k) or {} for k in ('gpu', 'power_plan'))
    guids = GUID.findall(' '.join(plan_info.get('output', []))) if plan_info.get('code') == 0 else []
    gpu = None
    if gpu_info.get('code') == 0 and len(gpu_info.get('output', [])) == 1:
        cells = next(csv.reader(gpu_info['output'], skipinitialspace=True))
        if len(cells) == len(GPU_COLS):
            gpu = dict(zip(GPU_COLS, cells, strict=True))
            for k in METRICS:
                try:
                    value = float(gpu[k])
                    gpu[k] = value if isfinite(value) else None
                except ValueError:
                    gpu[k] = None
    return dict(ac_online=raw.get('ac_online'), power_plan=guids[0].lower() if len(guids) == 1 else None, gpu=gpu)


def range_of(values):
    return dict(observations=len(values), min=min(values), median=median(values), max=max(values)) if values else dict(observations=0)


def report(campaign, progress):
    pairs, counts = [], Counter()
    for spec in progress['completed_prefix']:
        root = campaign / f"r{spec['round']:02d}-{spec['block']}"
        ledger = read(root / 'pair.json')
        armed = ledger['arms']
        obs = {a['policy']: {k: snapshot(a.get(k) or {}) for k in ('setup', 'after')} for a in armed}
        pre = {k: r['setup'] for k, r in obs.items()}
        gap = (datetime.fromisoformat(armed[1]['start_utc']) - datetime.fromisoformat(armed[0]['end_utc'])).total_seconds()
        if gap < 0:
            raise ValueError(f'overlapping paired arms: {root.name}')
        temps = {p: (s['gpu'] or {}).get('temp_c') for p, s in pre.items()}
        delta = temps['entry-replay'] - temps['local-cost'] if all(v is not None for v in temps.values()) else None
        for k, rule in (('ac_online', lambda r: r['ac_online']), ('power_plan', lambda r: r['power_plan']),
                        ('gpu_identity', lambda r: (r['gpu'] or {}).get('gpu_id'))):
            observations = [rule(r) for phases in obs.values() for r in phases.values()]
            counts[k + '_pairs_missing_observation'] += any(x is None for x in observations)
            counts[k + '_pairs_with_observed_change'] += len({x for x in observations if x is not None}) > 1
        counts['pairs_with_battery_observation'] += any(r['ac_online'] is False for phases in obs.values() for r in phases.values())
        pairs.append(dict(block=spec['block'], round=spec['round'], family=spec['family'], first_policy=armed[0]['policy'],
                          between_arm_timestamps_s=gap, entry_minus_local_start_temp_c=delta, arms=obs))
    by_phase = {}
    for phase in ('setup', 'after'):
        samples = [arm[phase]['gpu'] for row in pairs for arm in row['arms'].values() if arm[phase]['gpu']]
        by_phase[phase] = {key: range_of([g[key] for g in samples if g[key] is not None]) for key in METRICS}
        by_phase[phase]['pstate_counts'] = dict(Counter(g['pstate'] for g in samples))
    return dict(scope='endpoints only; inter-arm timestamps have whole-second resolution and include controller overhead; no continuous GPU trace, latency or exclusion decision',
                lock_sha256=progress['lock_sha256'], completed_pairs=len(pairs), counts=dict(counts),
                entry_minus_local_start_temp_c={p: range_of([x['entry_minus_local_start_temp_c'] for x in pairs
                    if x['first_policy'] == p and x['entry_minus_local_start_temp_c'] is not None]) for p in ('local-cost', 'entry-replay')},
                between_arm_timestamps_s=range_of([x['between_arm_timestamps_s'] for x in pairs]), by_phase=by_phase, pairs=pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('lock', 'campaign', 'catalogues', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    result = report(a.campaign, inspect(a.lock, a.campaign, a.catalogues))
    with a.out.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(result, indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
