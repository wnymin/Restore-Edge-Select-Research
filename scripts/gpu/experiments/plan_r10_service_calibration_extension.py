"""Choose another service calibration clock using only audited baseline queue windows."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts.gpu.analysis.summarize_r10_service_calibration import anchors
from scripts.gpu.experiments.run_r10_catalog_prepare import write_new
from scripts.gpu.experiments.run_r10_service_calibration import EXTENSION_SCOPE, SCOPE
from scripts.validation.check_r10_native_catalog import digest


def propose(design, prior):
    points = []
    e = design['extension_rule']
    for block, info in design['blocks'].items():
        family = info['family']
        for profile in design['output_profiles']:
            rows = [r for r in prior['arms'] if (r['block'], r['profile'], r['family']) == (block, profile, family)]
            used = {r['rate']: Fraction(r['arrival_scale']) for r in rows}
            if len(used) != len(rows) or not all(used.get(k) == Fraction(v) for k, v in info['arrival_scale'].items()):
                raise ValueError('baseline ladder does not match the original rate registration')
            selected = anchors(rows, design['anchor_rules'])
            if prior['chosen'][f'{family}/{profile}'] != selected:
                raise ValueError('saved rate anchors differ from the recorded queue windows')
            n = len(used) - len(info['arrival_scale'])
            proposals = []
            if selected['low'] is None:
                proposals.append(('low', max(used.values()) * Fraction(e['clock_multiplier_if_low_missing'])))
            if selected['overload'] is None:
                proposals.append(('overload', min(used.values()) * Fraction(e['clock_multiplier_if_overload_missing'])))
            if selected['shoulder'] is None and not proposals:
                fast, slow = used[selected['overload']], used[selected['low']]
                if not fast < slow:
                    raise ValueError('observed anchor clocks are not ordered; examine non-monotone calibration')
                bracket = sorted({v for v in used.values() if fast <= v <= slow})
                _, high, low = max((b-a, b, a) for a, b in zip(bracket, bracket[1:]))
                proposals.append(('shoulder', (high+low)/2))
            for target, clock in proposals[:max(0, e['max_added_rates_per_stratum']-n)]:
                if not clock > 0 or clock in used.values():
                    raise ValueError('extension failed to find an untested positive clock')
                n += 1
                rate = f'extra{n:02d}'
                if rate in used:
                    raise ValueError(f'expected a fresh added rate label: {rate}')
                points.append(dict(block=block, profile=profile, rate=rate, family=family,
                                   target=target, arrival_scale=str(clock)))
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('design', 'calibration-summary', 'calibration-archive', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    design, prior = (json.loads(x.read_bytes()) for x in (a.design, a.calibration_summary))
    design_hash = digest(a.design)
    archive_hash = digest(a.calibration_archive)
    if design['scope'] != SCOPE or prior['design_sha256'] != design_hash or prior['archive_sha256'] != archive_hash:
        raise ValueError('rate extension needs the verified baseline calibration evidence')
    points = propose(design, prior)
    if not points:
        print('No additional registered rates are needed or available; inspect missing anchors before freezing.')
        return
    write_new(a.out, dict(scope=EXTENSION_SCOPE, design_sha256=design_hash,
                          baseline_summary_sha256=digest(a.calibration_summary), baseline_archive_sha256=archive_hash, points=points))


if __name__ == '__main__':
    main()
