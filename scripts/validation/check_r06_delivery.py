"""Recompute the R06 acceptance statistics from paired run records."""

import argparse
from collections import defaultdict
import gzip
import json
from random import Random
from statistics import mean, median

from scripts.cpu.replay.study_r06_transfer import CONFIG, POLICIES, ROOT, accepted, domains, read, store, digest
from scripts.validation.study_r06_snapshots import payload


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def matrix(cfg, stage):
    split = 'holdout' if stage == 'holdout' else 'calibration'
    return {(w, label, m, layout, b) for w in cfg[f'{split}_windows']
            for label, m, _ in domains(cfg, stage) for layout in cfg['layouts'] for b in cfg['budgets']}


def key(row):
    k = row['key']
    return tuple(k[field] for field in ('window', 'domain', 'model', 'placement', 'budget'))


def check_calibration(cfg, calibration, frozen):
    assert frozen['config_sha256'] == digest(CONFIG)
    for stage in ('single', 'calibration'):
        rows = [r for r in calibration if r['stage'] == stage]
        assert len(rows) == len(matrix(cfg, stage)) and {key(r) for r in rows} == matrix(cfg, stage)
    for margin in cfg['unified_margins']:
        values = [(r['metrics']['margin-' + margin], r['metrics']['entry-replay']) for r in calibration]
        assert frozen['stable'][margin] == all(accepted(a, b, cfg['budget']) for a, b in values)
        assert frozen['skips'][margin] == sum(a['routes'].get('fast', 0) for a, _ in values)
    from fractions import Fraction
    assert frozen['margin'] == max((m for m in frozen['stable'] if frozen['stable'][m]),
                                  key=lambda m: (frozen['skips'][m], Fraction(m)))
    for model, layouts in frozen['thresholds'].items():
        for layout, actual in layouts.items():
            in_domain = [r for r in calibration if r['stage'] == 'calibration'
                         and r['key']['model'] == model and r['key']['placement'] == layout]
            assert in_domain and actual == max(t for t in cfg['calibration_thresholds'] if all(
                accepted(r['metrics']['always-early'], r['metrics']['entry-replay'], cfg['budget'])
                for r in in_domain if int(r['key']['ssd']) <= t))


def summarize(cfg, holdout):
    expected = matrix(cfg, 'holdout')
    assert len(holdout) == len(expected) and {key(r) for r in holdout} == expected
    groups = defaultdict(list)
    for r in holdout:
        k, zero, timed = r['key'], r['zero'], r['timed']
        assert zero['entry']['sequence'] == zero['entry-replay']['sequence']
        assert len(timed) == cfg['repeats'] and all(set(t) == set(POLICIES) for t in timed)
        groups[k['model'], k['domain']].append(r)
    generator = Random(cfg['seed'])
    n = len(cfg['holdout_windows'])
    draws = [generator.choices(range(n), k=n) for _ in range(cfg['bootstrap_draws'])]
    output = []
    for (model, cost), rows in sorted(groups.items()):
        for policy in POLICIES:
            if policy == 'entry-replay':
                continue
            zeroes, ref = [[r['zero'][p] for r in rows] for p in (policy, 'entry-replay')]
            differences = defaultdict(list)
            maximum_request = p95_losses = regressions = 0
            timed_response, timed_tail, decision = [], [], []
            for r in rows:
                paired = [(t[policy], t['entry-replay']) for t in r['timed']]
                differences[r['key']['window']].append(median(b['mean'] - a['mean'] for a, b in paired))
                p95_losses = max(p95_losses, median(a['p95'] - b['p95'] for a, b in paired))
                maximum_request = max(maximum_request, *(a['max_loss'] for a, _ in paired))
                regressions += sum(a['regressions'] for a, _ in paired)
                timed_response.append(median(a['mean'] for a, _ in paired))
                timed_tail.append(median(a['p95'] for a, _ in paired))
                decision.append(median(a['decision_ms'] for a, _ in paired))
            by_window = [mean(differences[w]) for w in cfg['holdout_windows']]
            samples = sorted(mean(by_window[i] for i in indices) for indices in draws)
            interval = [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]
            budgets = cfg['budget']
            passed = (all(accepted(a, b, budgets) for a, b in zip(zeroes, ref, strict=True))
                      and p95_losses <= budgets['timed_config_p95_ms']
                      and maximum_request <= budgets['timed_max_request_ms'])
            output.append(dict(model=model, cost=cost, policy=policy, cases=len(rows),
                               requests=sum(r['requests'] for r in rows),
                               zero_boundary_changes=sum(a['boundary_changes'] for a in zeroes),
                               zero_mean_loss_ms=mean(a['mean'] - b['mean'] for a, b in zip(zeroes, ref)),
                               zero_max_config_loss_ms=max(a['mean'] - b['mean'] for a, b in zip(zeroes, ref)),
                               zero_max_request_loss_ms=max(a['max_loss'] for a in zeroes),
                               zero_replays=sum(a['replays'] for a in zeroes),
                               timed_mean_ms=mean(timed_response), timed_p95_ms=mean(timed_tail),
                               reference_timed_mean_ms=mean(median(t['entry-replay']['mean'] for t in r['timed']) for r in rows),
                               reference_decision_ms_per_window=mean(median(t['entry-replay']['decision_ms'] for t in r['timed']) for r in rows),
                               decision_ms_per_window=mean(decision),
                               net_ms=mean(by_window), cluster_95_ms=interval,
                               window_net_ms=dict(zip(cfg['holdout_windows'], by_window, strict=True)),
                               timed_worst_paired_p95_ms=p95_losses, timed_max_request_loss_ms=maximum_request,
                               timed_regressions_gt10ms=regressions,
                               routes={name: sum(r['routes'].get(name, 0) for r in rows) for name in ('fast', 'fallback')}
                               if policy == 'unified' else None,
                               budget_passed=passed, performance_passed=passed and interval[0] > 0))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', default=ROOT / 'docs/evidence', type=type(ROOT))
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    path = args.evidence
    cfg = read(CONFIG)
    for name, expected in read(path / 'r06-manifest.json')['files'].items():
        source = ROOT / name
        if source.parent == ROOT / 'docs/evidence':
            if args.write and source.name == 'r06-summary.json':
                continue
            source = path / source.name
        assert digest(source) == expected, source
    for name, expected in cfg['inputs'].items():
        assert digest(ROOT / name) == expected, name
    calibration = path / 'r06-calibration.jsonl.gz'
    frozen = read(path / 'r06-frozen.json')
    assert digest(calibration) == frozen['calibration_sha256']
    check_calibration(cfg, records(calibration), frozen)
    result = summarize(cfg, records(path / 'r06-holdout.jsonl.gz'))
    summary = path / 'r06-summary.json'
    if args.write:
        store(summary, result)
    else:
        assert json.loads(json.dumps(result)) == read(summary)
    assert payload() == read(path / 'r06-snapshots.json')
    print(f'R06 checked: 216 calibration and 432 holdout configurations, 38 snapshot witnesses, {len(result)} paired strata.')


if __name__ == '__main__':
    main()
