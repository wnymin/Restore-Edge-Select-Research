"""Audit all registered native and injected-wait mechanism replications."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.audit_r10_mechanism_pair import pair
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints
from scripts.gpu.experiments.run_r10_catalog_prepare import paths
from scripts.validation.check_r10_native_catalog import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ('design', 'root', 'runs', 'archive', 'out'):
        ap.add_argument('--' + option, required=True, type=Path)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('mechanism evidence paths must be new')
    d = json.loads(a.design.read_bytes())
    start = a.runs / 'design.json'
    stamp = json.loads(start.read_bytes())
    assert stamp['design_sha256'] == digest(a.design)
    files = {p.name: p for p in (a.design, start)}
    inputs, rows = {}, {}
    for spec in d['mechanism_pairs']:
        tag, key = spec['tag'], spec['input']
        if key not in inputs:
            data = d['mechanism_inputs'][key]
            folder, block = a.root / data['directory'], data['block']
            schedule = folder / f'{block}.schedule.json'
            assert digest(schedule) == data['schedule_sha256']
            prep = paths(folder / f'{block}-native')
            seal = json.loads(prep['seal'].read_bytes())
            assert seal['schedule_sha256'] == digest(schedule) and seal['split'] == 'calibration'
            assert seal['run_receipt_sha256'] == digest(prep['run'])
            for p in prep.values():
                files[f'input/{key}/{p.name}'] = p
            inputs[key] = folder, block, schedule
        folder, block, schedule = inputs[key]
        shallow, deep = (a.runs / f'{tag}-{arm}' for arm in ('shallow', 'deep'))
        schedules = tuple(folder / f'{block}-{arm}.json' for arm in ('shallow', 'deep'))
        r = pair(shallow, deep, schedule, *schedules)
        assert r['capacity_bytes'] == Fraction(spec['capacity']) * 2**20 and r['soft_ssd_hold_ms'] == spec['hold_ms']
        assert r['publication_anchor'] == spec['anchor']
        assert all(r[arm]['audit']['runtime_worker_sha256'] == stamp['sha256']['mechanism_worker'] for arm in ('shallow', 'deep'))
        receipts = {arm: json.loads(endpoints(a.runs / f'{tag}-{arm}')[0].read_bytes()) for arm in ('shallow', 'deep')}
        x, y = [receipts[arm] for arm in spec['order']]
        assert x['drain_observed_utc'] < y['arrival_epoch_utc']
        for p in [*endpoints(shallow), *endpoints(deep), *schedules]:
            files[f'pair/{tag}/{p.name}'] = p
        rows[tag] = dict(r, registered_order=spec['order'], input=key, seed=d['mechanism_inputs'][key]['seed'])
    plain, delayed = (rows[k] for k in ('repeat-shared-tight', 'repeat-shared-soft700'))
    assert plain['schedule_sha256'] == delayed['schedule_sha256'] and plain['capacity_bytes'] == delayed['capacity_bytes']
    raw_gain, delayed_gain = [x['shallow_minus_deep_total_response_ms'] for x in (plain, delayed)]
    with ZipFile(a.archive, 'x') as z:
        for name, p in files.items():
            z.writestr(ZipInfo(name), p.read_bytes(), compress_type=ZIP_DEFLATED)
    summary = dict(scope='forced-action native GPU replications; no production incidence or strategy A/B estimate',
                   design_sha256=digest(a.design), archive_sha256=digest(a.archive), pairs=rows,
                   new_seed_handoff=dict(raw_gain_ms=raw_gain, delayed_gain_ms=delayed_gain,
                                         earlier_direction_reproduced=raw_gain > 0 > delayed_gain))
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
