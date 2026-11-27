"""Check decode room and remaining prompt suffix for the registered service inputs, without running inference."""

import argparse
import json
from pathlib import Path

from scripts.gpu.experiments.run_r10_service_calibration import generation_shape
from scripts.validation.check_r10_native_catalog import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('design', 'membership', 'plans', 'extra-plans', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    design = json.loads(a.design.read_bytes())
    heldout = json.loads(a.membership.read_bytes())
    entries = [(k, v, a.plans, 'calibration') for k, v in design['blocks'].items()]
    entries += [(r['id'], r, a.extra_plans if r['family'] == 'oasst' else a.plans, 'service membership') for r in heldout['blocks']]
    rows = []
    for ident, rec, root, origin in entries:
        path = root / (ident + '.schedule.json')
        if digest(path) != rec['schedule_sha256']:
            raise ValueError(f'{ident}: schedule hash does not match its preregistered input')
        sch = json.loads(path.read_bytes())
        for profile, limits in design['output_profiles'].items():
            row = generation_shape(sch, limits)
            if rec.get('requests', row['requests']) != row['requests']:
                raise ValueError('request coverage differs from the registration')
            rows.append(dict(block=ident, family=rec['family'], origin=origin, profile=profile,
                             schedule_sha256=rec['schedule_sha256'], **row))
    with a.out.open('x', encoding='utf8') as f:
        print(json.dumps(dict(scope='static context and suffix audit only; no holdout runtime or output quality evidence',
                              membership_sha256=digest(a.membership), design_sha256=digest(a.design), rows=rows),
                         sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
