"""Freeze and evaluate preselection GPU signals on paired shallow/deep trials."""

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from scripts.gpu.analysis.summarize_dynamic_gpu_delay import enrich, values


CUTS = {'pending': (None,), 'age': (150, 450, 750, 1050),
        'age_work': (0.10, 0.25, 0.50, 0.75, 1.0)}
OBJECTIVES = {'target': 0, 'sum': 2, 'finish': 3}


def read_pairs(path, log=None, out=None):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No trial rows')
    if log:
        enrich(rows, log, include_peer_position=True)
    blocks = defaultdict(dict)
    rows = [r for r in rows if int(r['rep']) >= 0]
    for row in rows:
        block = blocks[tuple(int(row[k]) for k in ('seed', 'rep', 'peer_total_tokens', 'lag_ms'))]
        arm = row['storage']
        if arm in block or (arm, int(row['boundary'])) not in {('resident', 1024), ('fresh', 1152)}:
            raise ValueError('Unexpected or duplicate trial arm')
        if (int(row['inflight']) != int(int(row['peer_total_tokens']) > 0) or
                int(row['peer_pending_at_selection']) not in (0, 1) or
                (not int(row['inflight']) and int(row['peer_pending_at_selection']))):
            raise ValueError('Unexpected peer signal')
        block[arm] = row
    pairs = []
    for key, block in sorted(blocks.items()):
        if set(block) != {'resident', 'fresh'}:
            raise ValueError(f'Incomplete paired block {key}')
        a, b = block['resident'], block['fresh']
        read = int(b['restored_bytes'])
        if int(a['restored_bytes']) or not 0 < read == int(b['warm_file_bytes']) <= int(b['volume_read_bytes']):
            raise ValueError(f'Unverified cold SSD read {key}')
        # Evaluate against the resident run's preselection snapshot for both alternatives.
        va = values({**a, 'actual_peer_lead_ms': a['actual_peer_lead_ms'] or '0'})
        vb = values({**b, 'actual_peer_lead_ms': b['actual_peer_lead_ms'] or '0'})
        same = a['greedy_token'] == b['greedy_token']
        for row in (a, b):
            row['target_token_match'] = int(same)
        stage = tuple(int(r['peer_prefix_at_target_batch']) >= key[2] - 4 for r in (a, b)) if key[2] else None
        pairs.append(dict(key=key, workload=key[2], pending=int(a['peer_pending_at_selection']), stage=stage,
                          age=float(a['actual_peer_lead_ms'] or 0), same=same,
                          pending_agree=a['peer_pending_at_selection'] == b['peer_pending_at_selection'],
                          gain=tuple(x-y for x, y in zip(va, vb, strict=True)), baseline=va))
    if out:
        with out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    return pairs


def group(pair, signal, cut):
    if signal == 'solo' or not pair['pending']:
        return 0
    if signal == 'pending':
        return 1
    age = pair['age']
    if signal == 'age_work':
        age /= pair['workload'] - 512
    return 1 if age <= cut else 2


def fit(pairs, signal, objective):
    index = OBJECTIVES[objective]
    solo = mean(p['gain'][index] for p in pairs if not p['workload'])
    if signal == 'solo':
        return dict(signal=signal, cut=None, means={'0': solo})
    candidates = []
    for cut in CUTS[signal]:
        groups = defaultdict(list)
        for pair in pairs:
            groups[group(pair, signal, cut)].append(pair['gain'][index])
        if signal != 'pending' and any(len(groups[k]) < 4 for k in (1, 2)):
            continue
        means = {str(k): mean(v) for k, v in groups.items()}
        means['0'] = solo  # No pending peer: use the independent-run cost comparison.
        model = dict(signal=signal, cut=cut, means=means)
        choices = [p['gain'][index] for p in pairs if predict(p, model) > 0]
        candidates.append((sum(choices), -len(choices), model))
    return max(candidates, key=lambda x: x[:2])[2] if candidates else fit(pairs, 'pending', objective)


def predict(pair, model):
    key = str(group(pair, model['signal'], model['cut']))
    return model['means'].get(key, model['means']['0'])


def fit_stage(pairs, arm):
    peers = [p for p in pairs if p['stage'] is not None]
    rates = {'0': mean(p['stage'][arm] for p in peers)}
    model = dict(signal='solo', cut=None, means=rates)
    candidates = []
    for signal, cuts in CUTS.items():
        for cut in cuts:
            groups = defaultdict(list)
            for pair in peers:
                groups[group(pair, signal, cut)].append(pair['stage'][arm])
            if any(len(groups[k]) < 4 for k in ((1,) if signal == 'pending' else (1, 2))):
                continue
            fitted = dict(signal=signal, cut=cut, means=rates | {str(k): mean(v) for k, v in groups.items()})
            brier = mean((predict(p, fitted) - p['stage'][arm]) ** 2 for p in peers)
            candidates.append((brier, fitted))
    return {'constant': model, 'dynamic': min(candidates, key=lambda x: x[0])[1] if candidates else model}


def evaluate_stage(pairs, rules):
    peers = [p for p in pairs if p['stage'] is not None]
    for index, arm in enumerate(('resident', 'fresh')):
        for name, rule in rules[arm].items():
            errors = [(predict(p, rule) - p['stage'][index]) ** 2 for p in peers]
            print(f'near-end stage {arm} {name}: signal={rule["signal"]} cut={rule["cut"]} '
                  f'Brier={mean(errors):.4f} on {len(peers)} trials')


def evaluate(pairs, models):
    print(f'Pairs {len(pairs)}, same target token {sum(p["same"] for p in pairs)}, '
          f'pending agrees across arms {sum(p["pending_agree"] for p in pairs)}')
    always_deep = tuple(round(mean(p['gain'][i] for p in pairs), 2) for i in range(4))
    print(f'Fixed deep vs resident: target/peer/sum/finish gains={always_deep} ms')
    for objective, rules in models.items():
        index = OBJECTIVES[objective]
        oracle = mean(max(p['gain'][index], 0) for p in pairs)
        for name, model in rules.items():
            predictions = [predict(p, model) for p in pairs]
            changed = [p for p, pred in zip(pairs, predictions, strict=True) if pred > 0]
            gains = [sum(p['gain'][k] for p in changed) / len(pairs) for k in range(4)]
            loss = max((max(0, -p['gain'][0]) for p in changed), default=0)
            mae = mean(abs(pred-p['gain'][index]) for p, pred in zip(pairs, predictions, strict=True))
            correct = sum((pred > 0) == (p['gain'][index] > 0)
                          for p, pred in zip(pairs, predictions, strict=True))
            by_prompt = defaultdict(list)
            for pair, pred in zip(pairs, predictions, strict=True):
                by_prompt[pair['key'][:2]].append(pair['gain'][index] if pred > 0 else 0)
            block_gains = [mean(v) for v in by_prompt.values()]
            print(f'{objective} {name} cut={model["cut"]}: switch={len(changed)}, '
                  f'target/peer/sum/finish gains={tuple(round(v, 2) for v in gains)}, '
                  f'largest target loss={loss:.2f}, sign correct={correct}/{len(pairs)}, '
                  f'MAE={mae:.2f}, paired-oracle regret={oracle-gains[index]:.2f} ms, '
                  f'prompt-block gains {min(block_gains):+.2f}..{max(block_gains):+.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('calibrate', 'evaluate', 'archive'))
    parser.add_argument('csv', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--matching-tokens-only', action='store_true')
    args = parser.parse_args()
    pairs = read_pairs(args.csv, args.log, args.out)
    if args.mode == 'archive':
        return
    if not args.rules:
        parser.error('The model file is required')
    if args.mode == 'calibrate':
        models = {obj: {signal: fit(pairs, signal, obj) for signal in ('solo', *CUTS)} for obj in OBJECTIVES}
        payload = dict(seeds=sorted({p['key'][0] for p in pairs}),
                       peer_workloads=sorted({p['workload'] for p in pairs if p['workload']}),
                       input_sha256=sha256(args.csv.read_bytes()).hexdigest(), models=models,
                       stage_models={arm: fit_stage(pairs, i) for i, arm in enumerate(('resident', 'fresh'))})
        with args.rules.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
    else:
        with args.rules.open(encoding='utf-8') as stream:
            payload = json.load(stream)
        if set(payload['seeds']) & {p['key'][0] for p in pairs} or set(payload['peer_workloads']) & {
                p['workload'] for p in pairs if p['workload']}:
            raise ValueError('Holdout overlaps with calibration seeds or nonzero peer workloads')
        models = payload['models']
    if args.matching_tokens_only:
        if args.mode != 'evaluate':
            parser.error('Token filtering is a holdout sensitivity only')
        pairs = [pair for pair in pairs if pair['same']]
    evaluate(pairs, models)
    evaluate_stage(pairs, payload['stage_models'])


if __name__ == '__main__':
    main()
