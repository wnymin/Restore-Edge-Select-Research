"""Execute the frozen R06 transfer matrix and archive paired observations."""

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import gzip
from hashlib import file_digest, sha256
from itertools import islice, product
import json
from pathlib import Path
from random import Random
from statistics import mean

from scripts.cpu.replay.study_bailian import Experiment, ROOT, build, p95, read
from scripts.cpu.replay.study_boundary_factors import predict
from scripts.cpu.replay.study_load_routing import run_case
from restore_edge_select.cost_transfer import allow_serial

CONFIG = ROOT / 'configs/r06-matrix.json'
ANCHOR = '4b'
ALT = 'synthetic'
MODELS = {ANCHOR: 'configs/qwen3.5-4b.json', ALT: 'configs/r06-synthetic-gqa-gdn.json'}
POLICIES = ('local-cost', 'entry', 'entry-replay', 'portfolio-v4', 'recalibrated', 'unified')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def store(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def windows(cfg, split):
    manifest = read(ROOT / 'data/bailian-manifest.json')
    path = ROOT / 'data/raw/bailian' / cfg['trace']
    assert digest(path) == next(r['sha256'] for r in manifest['files'] if r['name'] == path.name)
    width = cfg['window_rows']
    with path.open(encoding='utf-8') as stream:
        previous = 0
        for window in cfg[f'{split}_windows']:
            skip = (window - previous) * width
            rows = [json.loads(line, parse_float=str) for line in islice(stream, skip, skip + width)]
            if len(rows) != width:
                raise ValueError(f'Incomplete window {window}')
            yield window, rows
            previous = window + 1


def domains(cfg, stage):
    if stage == 'single':
        yield 'anchor', ANCHOR, (1, 1, 11)
        yield 'kv-only', 'kv-only', (1, 1, 11)
        yield 'state-only', 'state-only', (1, 1, 11)
        yield 'compute-only', ANCHOR, ('3/4', 1, 11)
        yield 'ram-only', ANCHOR, (1, 2, 11)
        yield 'ssd-only', ANCHOR, (1, 1, 32)
    elif stage == 'calibration':
        for model, ssd in product(MODELS, cfg['calibration_ssd']):
            yield str(ssd), model, (1, 1, ssd)
    else:
        for model, (level, factors) in product(MODELS, cfg['holdout_profiles'].items()):
            yield level, model, factors


def cases(cfg, stage):
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    models = {key: read(ROOT / path)['models'][0] for key, path in MODELS.items()}
    for name, fields in (
        ('kv-only', ('attention_bytes_per_token',)),
        ('state-only', ('recurrent_bytes_per_checkpoint', 'conv_bytes_per_checkpoint', 'checkpoint_bytes')),
    ):
        model = deepcopy(models[ANCHOR])
        for field in fields:
            model['payload'][field] = models[ALT]['payload'][field]
        models[name] = model
    grid = read(ROOT / 'docs/evidence/prefill-vulkan-4k.json')
    io = read(ROOT / 'docs/evidence/load-staged-ssd-profile.json')['cost']
    overlap = read(ROOT / 'configs/overlap-observed-upper.json')
    split = 'holdout' if stage == 'holdout' else 'calibration'
    for window, rows in windows(cfg, split):
        for (domain, model_key, factors), layout in product(domains(cfg, stage), cfg['layouts']):
            model = models[model_key]
            compute, ram, ssd = map(F, factors)
            continuation = dict(grid, samples=[dict(row, ms=[str(F(str(x)) * compute) for x in row['ms']])
                                               for row in grid['samples']])
            cost = dict(fixture['cost'], recompute=continuation)
            for path, factor in zip(('ram_to_device', 'ssd_to_device', 'device_copy'), (ram, ssd, 1)):
                cost[path] = dict(startup_ms=str(F(io[path]['startup_ms']) * factor),
                                  bytes_per_ms=str(F(io[path]['bytes_per_ms']) / factor))
            data = build(rows, cfg['history'], fixture, model, F(cfg['spacing']), layout, cfg['max_prompt'])
            for budget in cfg['budgets']:
                key = dict(window=window, domain=domain, model=model_key, placement=layout,
                           compute=str(compute), ram=str(ram), ssd=str(ssd), budget=budget)
                yield key, Experiment(dict(data, cost=cost, overlap_slowdown=overlap), model, budget)


def score(row):
    return predict('fifo_snapshot', row)


def run(exp, key, policy, frozen=None, *, timed=False, audit=None, overrides=None):
    if policy == 'local-cost':
        return exp.run('local-cost', inject_measured=timed, decision_overrides=overrides)
    mode = 'entry' if policy == 'entry' else 'entry-replay'
    options = {}
    if policy in ('portfolio-v4', 'recalibrated', 'always-early'):
        rate = F(key['ssd'])
        layout = key['placement']
        if policy == 'portfolio-v4':
            # Unchanged R05 predicate for its path="ssd" experiment. R06 has an independent RAM factor.
            enabled = layout == 'ram' or rate <= 11
        elif policy == 'recalibrated':
            threshold = frozen['thresholds'][key['model']][layout]
            enabled = threshold > 0 and rate <= threshold
        else:
            enabled = True
        if enabled:
            mode = 'entry-portfolio-early'
    if policy == 'unified':
        margin = F(frozen['margin'])

        def select(rows, base, _):
            skip = allow_serial(rows, base, exp.budget, margin)
            if audit is not None:
                audit['fast' if skip else 'fallback'] += 1
            return None if skip else 4

        mode = 'entry-portfolio'
        options['screen_selector'] = select
    return exp.run('snapshot-screened', score_candidate=score, screen_mode='unsettled', queue_horizon=4,
                   screen_search=mode, inject_measured=timed, decision_overrides=overrides, **options)


def responses(exp, result):
    return [end - arrival for end, arrival in zip(result['completion_ms'], exp.arrivals, strict=True)]


def pack(exp, result, reference):
    response = responses(exp, result)
    losses = [float(a - b) for a, b in zip(result['completion_ms'], reference['completion_ms'], strict=True)]
    boundaries = result['boundaries_tokens']
    return dict(mean=float(result['response_mean_ms']), p95=float(p95(response)),
                decision_ms=sum(ns for _, ns in result.get('decision_profile_ns', [])) / 1e6,
                max_loss=max(0, *losses), regressions=sum(x > 10 for x in losses),
                boundary_changes=sum(a != b for a, b in zip(boundaries, reference['boundaries_tokens'], strict=True)),
                replays=result.get('decision_replays', 0),
                sequence=sha256(json.dumps([boundaries, list(map(str, result['completion_ms']))]).encode()).hexdigest())


def archive_check(exp, key, original):
    class Discard:
        def writerow(self, row):
            pass

    captured = {}
    run_case(exp, dict(key, path='ssd', io_scale=F(key['ssd'])), Discard(), None, True,
             ('portfolio-v4',), capture=captured)
    assert original == captured['portfolio-v4'], key


def accepted(actual, ref, budget):
    return (actual['mean'] - ref['mean'] <= budget['zero_config_mean_ms']
            and actual['max_loss'] <= budget['zero_max_request_ms'])


def calibrate(cfg, out):
    budget = cfg['budget']
    margins = cfg['unified_margins']
    stable = {m: True for m in margins}
    skips = Counter()
    viable = {}
    target = out / 'calibration.jsonl.gz'
    with gzip.open(target, 'wt', encoding='utf-8') as stream:
        for stage in ('single', 'calibration'):
            count = 0
            for key, exp in cases(cfg, stage):
                exact = run(exp, key, 'entry-replay')
                ref = pack(exp, exact, exact)
                old = run(exp, key, 'portfolio-v4')
                if key['window'] == cfg['calibration_windows'][0]:
                    archive_check(exp, key, old)
                output = {'entry-replay': ref, 'local-cost': pack(exp, run(exp, key, 'local-cost'), exact),
                          'portfolio-v4': pack(exp, old, exact),
                          'always-early': pack(exp, run(exp, key, 'always-early'), exact)}
                for margin in margins:
                    audit = Counter()
                    variant = run(exp, key, 'unified', {'margin': margin}, audit=audit)
                    output['margin-' + margin] = actual = pack(exp, variant, exact) | dict(routes=dict(audit))
                    stable[margin] &= accepted(actual, ref, budget)
                    skips[margin] += audit['fast']
                if stage == 'calibration':
                    group = (key['model'], key['placement'], int(key['ssd']))
                    viable[group] = viable.get(group, True) and accepted(output['always-early'], ref, budget)
                stream.write(json.dumps(dict(stage=stage, key=key, metrics=output)) + '\n')
                count += 1
            print(f'{stage}: {count} cases', flush=True)
    thresholds = {}
    for model in MODELS:
        thresholds[model] = {layout: max(t for t in cfg['calibration_thresholds'] if all(
            viable[model, layout, ssd] for ssd in cfg['calibration_ssd'] if ssd <= t)) for layout in cfg['layouts']}
    margin = max((m for m in margins if stable[m]), key=lambda m: (skips[m], F(m)))
    store(out / 'frozen.json', dict(config_sha256=digest(CONFIG), calibration_sha256=digest(target),
                                   thresholds=thresholds, margin=margin, skips=skips, stable=stable))


def holdout(cfg, out):
    frozen = read(out / 'frozen.json')
    assert frozen['config_sha256'] == digest(CONFIG)
    assert frozen['calibration_sha256'] == digest(out / 'calibration.jsonl.gz')
    rng = Random(cfg['seed'])
    with gzip.open(out / 'holdout.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for count, (key, exp) in enumerate(cases(cfg, 'holdout'), 1):
            audit = Counter()
            zero = {p: run(exp, key, p, frozen, audit=audit if p == 'unified' else None) for p in POLICIES}
            ref = zero['entry-replay']
            assert all(zero['entry'][x] == ref[x] for x in ('boundaries_tokens', 'completion_ms')), key
            order = list(POLICIES)
            rng.shuffle(order)
            for _ in range(cfg['timing_warmups']):
                for policy in order:
                    run(exp, key, policy, frozen, timed=True)
            rounds = []
            for repeat in range(cfg['repeats']):
                sequence = order[repeat:] + order[:repeat]
                timed = {p: run(exp, key, p, frozen, timed=True) for p in sequence}
                rounds.append({p: pack(exp, r, timed['entry-replay']) for p, r in timed.items()})
            stream.write(json.dumps(dict(key=key, requests=len(exp.requests), routes=dict(audit),
                                         zero={p: pack(exp, r, ref) for p, r in zero.items()}, timed=rounds)) + '\n')
            stream.flush()
            if count % 36 == 0:
                print(f'holdout: {count} cases, through window {key["window"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calibrate', 'holdout'))
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/r06')
    args = parser.parse_args()
    cfg = read(CONFIG)
    for path, expected in cfg['inputs'].items():
        assert digest(ROOT / path) == expected, path
    args.out.mkdir(parents=True, exist_ok=True)
    (calibrate if args.stage == 'calibrate' else holdout)(cfg, args.out)


if __name__ == '__main__':
    main()
