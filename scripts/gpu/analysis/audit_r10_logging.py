"""Audit basic event emission on/off with counterbalanced calibration pairs."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints, p95
from scripts.validation.check_r10_native_catalog import digest


def run(prefix, schedule, enabled):
    receipt, raw, _ = endpoints(prefix)
    row = json.loads(receipt.read_bytes())
    fixed_keys = ('block', 'policy', 'capacity_bytes', 'worker_sha256', 'observer_sha256', 'seal_sha256',
                  'cost_profile_sha256', 'planned_arrival_offsets_ms', 'publish_rest_on')
    assert row['warmup'] and row['warmup_grid'] and row['policy'] == 'local-cost'
    assert row['publish_rest_on'] == 'open-loop' and row['native_events'] == enabled
    assert not any(row.get(k) for k in ('resources', 'chunk_events', 'pipeline_events', 'qualification', 'ssd_hold_ms', 'holdout_lock_sha256'))
    if enabled:
        check(receipt, raw, schedule)
    else:
        header = 'event,native_us,request,identity,bytes,charged_bytes,ticket,interval_us'
        assert raw.read_text(encoding='ascii').splitlines() == [header]
    a, o, z = (list(map(Fraction, row[k])) for k in ('client_arrivals_ms', 'observed_arrivals_ms', 'completions_ms'))
    decisions = sorted(row['decisions'], key=lambda x: x['snapshot']['request'])
    choices = [d['action'] for d in decisions]
    assert len(choices) == len(a) == len(o) == len(z)
    assert [d['snapshot']['request'] for d in decisions] == [d['compute']['request'] for d in row['starts']] == list(range(len(a)))
    assert choices == [d['compute']['boundary'] for d in row['starts']]
    assert all(i <= j <= k for i, j, k in zip(a, o, z, strict=True))
    t = [float(k - i) for i, k in zip(a, z, strict=True)]
    return {k: row[k] for k in fixed_keys}, choices, dict(mean_ms=mean(t), p95_ms=p95(t), receipt_sha256=digest(receipt),
                                                         start_utc=row['arrival_epoch_utc'], drain_utc=row['drain_observed_utc'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', required=True, action='append', type=Path, nargs=2, metavar=('LOGGED', 'QUIET'))
    for key in ('schedule', 'aa', 'archive', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    if len(a.pair) != 2 or len({x for group in a.pair for x in group}) != 4:
        p.error('supply two independent runs in each mode, with reversed execution order')
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('use new verification and archive paths')
    baseline = json.loads(a.aa.read_bytes())
    pairs = []
    for on, off in a.pair:
        fixed, choices_on, mode_on = run(on, a.schedule, True)
        quiet_fixed, choices_off, mode_off = run(off, a.schedule, False)
        assert fixed == quiet_fixed and fixed['block'] == baseline['block'] and fixed['cost_profile_sha256'] == baseline['cost_profile_sha256']
        assert mode_on['drain_utc'] < mode_off['start_utc'] or mode_off['drain_utc'] < mode_on['start_utc']
        pairs.append(dict(on=mode_on, off=mode_off, on_only_ms=mode_on['mean_ms'] - mode_off['mean_ms'],
                          p95_change_ms=mode_on['p95_ms'] - mode_off['p95_ms'],
                          different_choices=sum(x != y for x, y in zip(choices_on, choices_off, strict=True)),
                          first='on' if mode_on['start_utc'] < mode_off['start_utc'] else 'off'))
    assert {r['first'] for r in pairs} == {'on', 'off'}
    margin = max(5., mean(p['off']['mean_ms'] for p in pairs) * .03, baseline['abs_mean_difference_ms'])
    effect = mean(p['on_only_ms'] for p in pairs)
    out = dict(scope='two counterbalanced controls on one calibration block; no equivalence inference',
               aa_summary_sha256=digest(a.aa), schedule_sha256=digest(a.schedule), records=pairs,
               observed_logging_mean_change_ms=effect, max_pair_abs_mean_change_ms=max(abs(p['on_only_ms']) for p in pairs),
               reference_effect_ms=margin, mean_change_reaches_reference=abs(effect) >= margin)
    with ZipFile(a.archive, 'x') as z:
        for group in a.pair:
            for prefix in group:
                for path in endpoints(prefix):
                    z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(out, archive_sha256=digest(a.archive)), indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
