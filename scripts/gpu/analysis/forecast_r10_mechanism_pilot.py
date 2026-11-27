"""Preregister mechanism predictions without using any forced-arm timings."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from restore_edge_select.simulator import Experiment
from scripts.validation.check_r10_native_catalog import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('prefix', 'profile', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    paths = {k: a.prefix.with_name(a.prefix.name + '-' + k + '.json') for k in ('shallow', 'deep')}
    path = a.prefix.with_suffix('.schedule.json')
    h = digest(path)
    control = {k: json.loads(v.read_bytes()) for k, v in paths.items()}
    assert all(s['schedule_sha256'] == h for s in control.values())
    rows = []
    # 350 ms is a simulated intermediate point only; measure 0 and 700 ms.
    for ms in (0, 350, 700):
        s = json.loads(path.read_bytes())
        s['trace']['cost'] = json.loads(a.profile.read_bytes())['cost']
        src = s['trace']['cost']['ssd_to_device']
        src['startup_ms'] = str(Fraction(src['startup_ms']) + ms)
        exp = Experiment(s['trace'], s['model'], Fraction(329, 4))
        results = {k: exp.run('fixed', boundaries=v['boundaries'])['completion_ms'] for k, v in control.items()}
        gains = [float(x - y) for x, y in zip(results['shallow'], results['deep'], strict=True)]
        rows.append(dict(soft_hold_ms=ms, shallow_minus_deep_completion_ms=gains,
                         shallow_minus_deep_total_ms=sum(gains), hardware_arms_planned=ms in (0, 700)))
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='CPU stationary forecast, no hardware outcome used; diagnostic intervention only',
                              capacity_mib=82.25, schedule_sha256=h, cost_sha256=digest(a.profile), results=rows),
                         indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
