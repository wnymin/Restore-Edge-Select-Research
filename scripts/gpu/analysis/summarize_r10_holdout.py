"""Verify and summarize the complete 60-pair local primary holdout campaign."""

import argparse
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import json
from math import ceil
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.experiments.r10_holdout_lock import analysis_hashes, arrival_scale, load
from scripts.validation.check_r10_native_catalog import digest


def percentile(xs, fraction):
    return sorted(xs)[ceil(len(xs) * fraction) - 1]


def arm(ledger, prefix, schedule, lock_hash, round_, block, lock):
    if 'driver_sha256' not in ledger:
        raise ValueError(f'missing driver provenance for {prefix}')
    for field, suffix in [('receipt_sha256', '.run.json'), ('native_sha256', '.native.csv'), ('failure_sha256', '.failure.json'),
                          ('stderr_sha256', '.stderr'), ('driver_sha256', '.driver.log')]:
        if field in ledger and digest(prefix.with_suffix(suffix)) != ledger[field]:
            raise ValueError(f'altered {field} for {prefix}')
    if ledger['return_code']:
        if 'receipt_sha256' in ledger:
            raise ValueError('failed arm has a successful receipt')
        failure = prefix.with_suffix('.failure.json')
        note = json.loads(failure.read_bytes()) if failure.exists() else None
        return dict(failed=True, failure=note, requests=block['requests'], return_code=ledger['return_code'])
    p, n = prefix.with_suffix('.run.json'), prefix.with_suffix('.native.csv')
    receipt = json.loads(p.read_bytes())
    audit = check(p, n, schedule)
    assert receipt['scope'] == 'R10 primary local holdout arm'
    assert (receipt['policy'], receipt['round'], receipt['block'], receipt['holdout_lock_sha256']) == (
        ledger['policy'], round_, block['id'], lock_hash)
    assert receipt['schedule_sha256'] == digest(schedule) == block['schedule_sha256']
    assert receipt['worker_sha256'] == lock['environment']['worker_sha256'] and receipt['observer_sha256'] == lock['environment']['observer_sha256']
    assert receipt['cost_profile_sha256'] == lock['cost_profile_sha256']
    assert audit['requests'] == block['requests']
    assert receipt['warmup_grid'] and receipt['native_events'] and not any(receipt[k] for k in ('qualification', 'chunk_events', 'pipeline_events', 'resources'))
    arrival, observed, completion = (list(map(Fraction, receipt[k])) for k in ('client_arrivals_ms', 'observed_arrivals_ms', 'completions_ms'))
    assert len(arrival) == len(observed) == len(completion) == audit['requests']
    assert all(a <= o <= c for a, o, c in zip(arrival, observed, completion, strict=True))
    plan = list(map(Fraction, receipt['planned_arrival_offsets_ms']))
    trace = json.loads(schedule.read_bytes())['trace']['requests']
    t = sorted(Fraction(str(r['arrival_ms'])) for r in trace)
    assert plan == [(x - t[0]) * Fraction(arrival_scale(lock, block)) for x in t]
    responses = [float(c - a) for a, c in zip(arrival, completion, strict=True)]
    by_request = sorted(receipt['decisions'], key=lambda r: r['snapshot']['request'])
    bytes_by_object = {str(r['id']): r['bytes'] for r in json.loads(schedule.read_bytes())['trace']['objects']}
    return dict(failed=False, requests=audit['requests'], mean_ms=mean(responses), p50_ms=percentile(responses, .5),
                p95_ms=percentile(responses, .95), choices=[r['action'] for r in by_request], responses_ms=responses,
                arrival_drift_ms=max(abs(float(a - b)) for a, b in zip(arrival, plan, strict=True)),
                observed_lag_ms=percentile([float(o - a) for a, o in zip(arrival, observed, strict=True)], .95),
                peak_stage_bytes=audit['peak_prefetch_bytes'], chooser_p95_ms=percentile([r['choose_ns'] / 1e6 for r in by_request], .95),
                total_uploaded_bytes=sum(bytes_by_object[r['read']['object_id']] for r in receipt['transfers']),
                event_counts=audit['device_event_counts'])


def pairs(out, plans, lock, lock_hash, limit=None):
    blocks = {r['id']: r for r in lock['blocks']}
    for p in lock['run_pairs'][:limit]:
        row = blocks[p['block']]
        root = out / f"r{p['round']:02d}-{row['id']}"
        rec = json.loads((root / 'pair.json').read_bytes())
        assert rec['lock_sha256'] == lock_hash and all(rec[k] == v for k, v in p.items())
        values = {}
        for i, (policy, entry) in enumerate(zip(p['arm_order'], rec['arms'], strict=True), 1):
            assert (entry['order'], entry['policy'], entry['prefix']) == (i, policy, f'{i}-{policy}')
            prefix = root / entry['prefix']
            assert json.loads(prefix.with_suffix('.arm.json').read_bytes()) == entry
            values[policy] = arm(entry, prefix, plans / (row['id'] + '.schedule.json'), lock_hash, p['round'], row, lock)
        yield dict(p, family=row['family'], source=row['source'], arms=values)


def budgets(campaign, catalogues, rows, total_pairs):
    """Inspect the first two preregistered pairs without making a strategy claim."""
    seen, arms, raw, catalogue_bytes = set(), [], 0, 0
    for row in rows:
        root = campaign / f"r{row['round']:02d}-{row['block']}"
        raw += sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        receipt = json.loads((root / 'pair.json').read_bytes())
        arms.extend((datetime.fromisoformat(p['end_utc']) - datetime.fromisoformat(p['start_utc'])).total_seconds()
                    for p in receipt['arms'])
        block = row['block']
        if block not in seen:
            seen.add(block)
            catalogue_bytes += sum(p.stat().st_size for p in (catalogues / block).iterdir() if p.is_file())
    return dict(scope='timing/storage budget from the first two ordered pairs; no strategy effect calculated',
                observed_blocks=len(seen), observed_families=sorted({r['family'] for r in rows}),
                observed_arms=len(arms), observed_run_log_bytes=raw,
                observed_catalogue_bytes=catalogue_bytes, arm_wall_seconds=arms,
                projection_if_observed_arm_times_generalized_excluding_prep_seconds=[2 * total_pairs * min(arms), 2 * total_pairs * max(arms)],
                raw_run_log_projection_bytes=round(raw * total_pairs / len(rows)))


def archive_campaign(out, lock_file, archive, lock):
    with ZipFile(archive, 'x') as z:
        z.writestr(ZipInfo('execution-lock.json'), lock_file.read_bytes(), compress_type=ZIP_DEFLATED)
        paths = [out / 'campaign.json', *sorted(out.glob('*.prepare.log'))]
        for r in lock['run_pairs']:
            folder = out / f"r{r['round']:02d}-{r['block']}"
            paths.extend(sorted(p for p in folder.iterdir() if p.is_file()))
        for p in paths:
            z.writestr(ZipInfo(p.relative_to(out).as_posix()), p.read_bytes(), compress_type=ZIP_DEFLATED)


def metrics(rows, lock):
    def average_key(items, key):
        return mean(r[key] for r in items)

    metrics = lock['acceptance']
    by_block = defaultdict(list)
    time_cap = lock['timing']['timeout_s'] * 1000
    for p in rows:
        a, b = (p['arms'][k] for k in ('local-cost', 'entry-replay'))
        mean_a, mean_b = (r['mean_ms'] if not r['failed'] else time_cap for r in (a, b))
        p['failure_penalized_gain_ms'] = mean_a - mean_b
        if not (a['failed'] or b['failed']):
            reg = [y - x for x, y in zip(a['responses_ms'], b['responses_ms'], strict=True)]
            p.update(gain_ms=mean_a - mean_b, tail_change_ms=b['p95_ms'] - a['p95_ms'],
                     baseline_mean_ms=mean_a, baseline_p95_ms=a['p95_ms'], max_request_regression_ms=max(reg),
                     fraction_request_regressions_over_budget=sum(x > metrics['request_wait_budget_ms'] for x in reg) / len(reg),
                     actions_differ=sum(x != y for x, y in zip(a['choices'], b['choices'], strict=True)))
        by_block[p['block']].append(p)
    blocks = []
    for ident, batch in by_block.items():
        ready = [x for x in batch if 'gain_ms' in x]
        row = dict(block=ident, family=batch[0]['family'], source=batch[0]['source'], ok_pairs=len(ready),
                   failure_penalized_gain_ms=average_key(batch, 'failure_penalized_gain_ms'))
        if ready:
            row.update({key: average_key(ready, key) for key in ('gain_ms', 'tail_change_ms', 'baseline_mean_ms',
                       'baseline_p95_ms', 'fraction_request_regressions_over_budget')})
            row.update(gain_pair_range_ms=[min(p['gain_ms'] for p in ready), max(p['gain_ms'] for p in ready)],
                       max_request_regression_ms=max(p['max_request_regression_ms'] for p in ready),
                       actions_differ=sum(p['actions_differ'] for p in ready))
        blocks.append(row)
    available = [r for r in blocks if 'gain_ms' in r]
    counts = {}
    for policy in ('local-cost', 'entry-replay'):
        arms = [r['arms'][policy] for r in rows]
        counts[policy] = sum(r['requests'] for r in arms if r['failed']) / sum(r['requests'] for r in arms)
    result = dict(blocks=blocks, successful_pairs=sum(r['ok_pairs'] for r in blocks), failed_request_fraction=counts,
                  penalized_gain_ms=average_key(blocks, 'failure_penalized_gain_ms'))
    if not available:
        return dict(result, decision='insufficient timing; no complete paired arms')
    gain, tail, baseline, baseline95, regressions = [average_key(available, key) for key in (
        'gain_ms', 'tail_change_ms', 'baseline_mean_ms', 'baseline_p95_ms', 'fraction_request_regressions_over_budget')]
    strata = {f: [r for r in available if r['family'] == f] for f in ('oasst', 'bailian')}
    sources = {s: [r for r in available if r['source'] == s] for s in sorted({r['source'] for r in available})}
    gain_by_family = {f: average_key(v, 'gain_ms') if v else None for f, v in strata.items()}
    tail_margin = max(metrics['p95_absolute_noninferiority_ms'], metrics['p95_relative_noninferiority'] * baseline95)
    gain_margin = max(metrics['min_absolute_gain_ms'], metrics['min_relative_gain'] * baseline, metrics['aa_p90_ms'])
    gates = dict(point_gain=gain > gain_margin, tail=tail < tail_margin,
                 both_families_positive=all(v is not None and v > 0 for v in gain_by_family.values()),
                 request_regression_rate=regressions <= metrics['max_fraction_over_wait_budget'],
                 availability=counts['entry-replay'] - counts['local-cost'] <= metrics['max_failed_request_fraction_increase'])
    result.update(mean_gain_ms=gain, mean_gain_margin_ms=gain_margin, mean_tail_change_ms=tail, tail_margin_ms=tail_margin,
                  observed_block_range_ms=[min(r['gain_ms'] for r in available), max(r['gain_ms'] for r in available)],
                  gain_by_family_ms=gain_by_family, gain_by_source_ms={s: average_key(v, 'gain_ms') for s, v in sources.items()},
                  gain_without_source_ms={s: average_key([r for r in available if r['source'] != s], 'gain_ms')
                                          for s in sources if len(sources[s]) < len(available)},
                  over_wait_budget_fraction=regressions, gates=gates,
                  decision=('all descriptive point gates pass' if all(gates.values()) else 'one or more descriptive point gates fail'),
                  inference='ranges and strata are descriptive; sources and blocks within one trace family need not be independent')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'plans', 'campaign', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--budget-first-two', action='store_true', help='audit the first two ordered pairs only; report runtime and storage')
    p.add_argument('--catalogues', type=Path, help='catalogue root for the budget inspection')
    p.add_argument('--archive', type=Path, help='archive all full-campaign raw receipts and logs')
    a = p.parse_args()
    if a.budget_first_two != bool(a.catalogues) or (a.budget_first_two and a.archive):
        p.error('budget inspection requires catalogues and cannot archive the full campaign')
    if a.out.exists() or (a.archive and a.archive.exists()):
        raise FileExistsError('select new primary evidence paths')
    lock, lock_hash = load(a.lock), digest(a.lock)
    if analysis_hashes() != lock['environment']['analysis_sha256']:
        raise ValueError('primary analysis code differs from the execution freeze')
    if json.loads((a.campaign / 'campaign.json').read_bytes())['lock_sha256'] != lock_hash:
        raise ValueError('campaign and execution lock differ')
    rows = list(pairs(a.campaign, a.plans, lock, lock_hash, 2 if a.budget_first_two else None))
    if a.budget_first_two:
        evidence = dict(lock_sha256=lock_hash, budget=budgets(a.campaign, a.catalogues, rows, len(lock['run_pairs'])))
    else:
        evidence = dict(lock_sha256=lock_hash, result=metrics(rows, lock), pair_rows=rows)
    if a.archive:
        archive_campaign(a.campaign, a.lock, a.archive, lock)
        evidence['archive_sha256'] = digest(a.archive)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(evidence, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
