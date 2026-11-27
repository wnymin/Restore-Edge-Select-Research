"""Audit the service calibration ladder and identify observed rate anchors."""

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.audit_r10_service_pipe import audit, audit_failed_sse
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints
from scripts.gpu.analysis.summarize_r10_pilot_load import measured_window
from scripts.gpu.experiments.run_r10_local_method_checks import plan
from scripts.gpu.experiments.run_r10_service_calibration import SCOPE, extensions, matrix, service_deadline
from scripts.validation.check_r10_native_catalog import digest


def anchors(rows, rules):
    low, shoulder, overload = [], [], []
    for row in rows:
        q = row.get('queue_window')
        if not q or row.get('failed'):
            continue
        d = q['drain_after_last_arrival_ms'] / q['submission_span_ms']
        b = q['fraction_time_any_request_outstanding']
        if b <= rules['low']['max_busy_fraction'] and d <= rules['low']['max_drain_fraction_of_arrival_span']:
            low.append(row)
        if (b >= rules['shoulder']['min_busy_fraction'] and q['max_outstanding'] >= rules['shoulder']['min_peak_outstanding'] and
                d <= rules['shoulder']['max_drain_fraction_of_arrival_span']):
            shoulder.append((abs(b - .8), d, row['rate']))
        if b >= rules['overload']['min_busy_fraction'] and d > rules['overload']['min_drain_fraction_of_arrival_span_exclusive']:
            overload.append((d, row['rate']))
    return dict(low=low[0]['rate'] if low else None, shoulder=min(shoulder)[-1] if shoulder else None,
                overload=min(overload)[-1] if overload else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('design', 'plans', 'runs', 'archive', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--extension', action='append', type=Path, default=[])
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('use new evidence paths')
    design = json.loads(a.design.read_bytes())
    added = extensions(a.extension, digest(a.design))
    stamp_file = a.runs / 'design.json'
    stamp = json.loads(stamp_file.read_bytes())
    assert design['scope'] == SCOPE and digest(a.design) == stamp['design_sha256']
    slo = design['experimental_slo']
    target = slo['visible_ttft_ms'], slo['client_request_tpot_ms']
    rows, raw, joined = [], {a.design.name: a.design, 'run-design.json': stamp_file}, defaultdict(list)
    for i, extension in enumerate(a.extension, 1):
        raw[f'extension/{i:02d}.json'] = extension
    for block, profile, rate, details, setting in matrix(design, added):
        _, schedule = plan(a.plans, block)
        assert digest(schedule) == details['schedule_sha256']
        source = json.loads(schedule.read_bytes())['trace']['requests']
        prefix = a.runs / f'{block}-{profile}-{rate}'
        p, n, err = endpoints(prefix)
        failed = prefix.with_suffix('.failure.json')
        record = dict(block=block, family=details['family'], profile=profile, rate=rate,
                      arrival_scale=details['arrival_scale'][rate], schedule_sha256=digest(schedule),
                      registered_timeout_s=service_deadline(design, source, details['arrival_scale'][rate]))
        if failed.exists() and not p.exists():
            note = json.loads(failed.read_bytes())
            assert (note['block'], note['policy']) == (block, 'local-cost')
            used = failed, n, err
            record.update(failed=True, error=note['error_type'], client=audit_failed_sse(note, len(source), *target))
        else:
            r = json.loads(p.read_bytes())
            assert (r['block'], r['policy'], r['publish_rest_on'], r['worker_sha256'], r['cost_profile_sha256']) == (
                block, 'local-cost', 'open-loop', stamp['sha256']['worker'], stamp['sha256']['cost_profile'])
            assert r['observer_sha256'] == stamp['sha256']['observer'] and r['warmup_grid'] and not r['qualification']
            assert r['capacity_bytes'] == Fraction(design['capacity_mib']) * 2**20
            assert (r['service']['short_limit'], r['service']['long_limit']) == (setting['short_tokens'], setting['long_tokens'])
            assert 'http_sse_requests' in r['service']
            arrivals = [Fraction(str(s['arrival_ms'])) for s in source]
            assert list(map(Fraction, r['planned_arrival_offsets_ms'])) == [
                (x - arrivals[0]) * Fraction(record['arrival_scale']) for x in arrivals]
            queue = measured_window(*(list(map(Fraction, r[k])) for k in ('client_arrivals_ms', 'completions_ms')))
            record.update(failed=False, client=audit(prefix, schedule, *target)['http_sse'], queue_window=queue)
            used = p, n, err
        for path in (*used, prefix.with_suffix('.driver.log')):
            raw[f'raw/{path.name}'] = path
        rows.append(record)
        joined[details['family'], profile].append(record)
    chosen = {f'{f}/{p}': anchors(rs, design['anchor_rules']) for (f, p), rs in joined.items()}
    with ZipFile(a.archive, 'x') as z:
        for label, path in raw.items():
            z.writestr(ZipInfo(label), path.read_bytes(), compress_type=ZIP_DEFLATED)
    result = dict(scope='baseline-only single-slot HTTP/SSE service calibration, no retained service performance estimate',
                  design_sha256=digest(a.design), archive_sha256=digest(a.archive), chosen=chosen, arms=rows,
                  extensions_sha256=[digest(x) for x in a.extension],
                  all_anchors_found=all(x is not None for group in chosen.values() for x in group.values()))
    with a.out.open('x', encoding='utf8') as f:
        print(json.dumps(result, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
