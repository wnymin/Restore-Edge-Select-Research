"""Check whether the existing forced mechanism states expose different policy decisions."""

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

from restore_edge_select.live_snapshot import choose_observed
from restore_edge_select.simulator import Experiment
from scripts.gpu.analysis.replay_r10_calibration_decisions import reconstruct
from scripts.validation.check_r10_native_catalog import digest


SSD_MULTIPLIERS = ('9/10', '1', '13/10', '2', '4', '8', '16')
PREFILL_MULTIPLIERS = ('1/4', '1/2', '3/4', '1', '5/4', '2')


def replay(receipt, plan, cost, request=None):
    exp = Experiment(dict(plan['trace'], cost=cost), plan['model'], Fraction(receipt['capacity_bytes'], 2**20))
    observed = list(map(Fraction, receipt['observed_arrivals_ms']))
    result = []
    for row in sorted(receipt['decisions'], key=lambda x: x['snapshot']['request']):
        if request is not None and row['snapshot']['request'] != request:
            continue
        snap = reconstruct(row['snapshot'])
        exp.arrivals = observed[:snap.arrived] + [snap.now_ms + 1] * (len(observed) - snap.arrived)
        choices = {p: choose_observed(exp, p, snap) for p in ('local-cost', 'entry-replay')}
        result.append(dict(request=snap.request, forced_boundary=row['action'], eligible=list(exp.eligible[snap.request]),
                           choices=choices, differs=len(set(choices.values())) == 2, visible=snap.arrived - snap.head,
                           active_compute=snap.compute_start_ms is not None, active_transfer=snap.transfer_object is not None))
    if [x['request'] for x in result] != ([0, 1, 2] if request is None else [request]):
        raise ValueError('expected the three registered mechanism request states')
    return result


def scaled(cost, ssd, prefill):
    candidate = deepcopy(cost)
    path = candidate['ssd_to_device']
    path['startup_ms'] = str(Fraction(path['startup_ms']) * ssd)
    path['bytes_per_ms'] = str(Fraction(path['bytes_per_ms']) / ssd)
    recompute = candidate['recompute']
    if recompute['kind'] != 'chunk-sampled-prefill':
        raise ValueError('sensitivity grid requires the frozen chunk-sampled prefill profile')
    for sample in recompute['samples']:
        sample['ms'] = [str(Fraction(ms) * prefill) for ms in sample['ms']]
    return candidate


def sensitivity(sources, base_cost):
    cells = []
    for s, c in product(SSD_MULTIPLIERS, PREFILL_MULTIPLIERS):
        cost = scaled(base_cost, Fraction(s), Fraction(c))
        records = [(tag, arm, replay(receipt, plan, cost, request=1)[0]) for tag, arm, receipt, plan in sources]
        counts = Counter('/'.join(str(row['choices'][p]) for p in ('local-cost', 'entry-replay')) for _, _, row in records)
        cells.append(dict(ssd_cost_multiplier=s, prefill_cost_multiplier=c, snapshots=len(records),
                          choice_pair_counts=dict(counts), disagreements=[dict(pair=t, arm=a, **r['choices']) for t, a, r in records if r['differs']]))
    return dict(scope='fixed recorded target snapshots with hypothetical estimator costs; does not reproduce altered execution or platform performance',
                choice_pair_order=['local-cost', 'entry-replay'], cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'root', 'archive', 'mechanism-summary', 'cost-profile', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cost-sensitivity', action='store_true', help='run the separate fixed-snapshot estimator sensitivity grid')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    d, summary, profile = (json.loads(p.read_bytes()) for p in (a.design, a.mechanism_summary, a.cost_profile))
    design_hash, archive_hash, cost_hash = map(digest, (a.design, a.archive, a.cost_profile))
    if (design_hash, archive_hash) != (summary['design_sha256'], summary['archive_sha256']) or \
            d['scope'] != 'R10 calibration only; no primary holdout input or timing':
        raise ValueError('requires the registered, audited calibration mechanism archive')
    rows, sources = [], []
    with ZipFile(a.archive) as z:
        for s in d['mechanism_pairs']:
            tag, spec = s['tag'], d['mechanism_inputs'][s['input']]
            path = a.root / spec['directory'] / (spec['block'] + '.schedule.json')
            static = path.read_bytes()
            if sha256(static).hexdigest() != spec['schedule_sha256']:
                raise ValueError(f'{tag}: mechanism schedule changed')
            plan = json.loads(static)
            seal_raw = z.read(f"input/{s['input']}/{spec['block']}-native.seal.json")
            seal_hash, seal = sha256(seal_raw).hexdigest(), json.loads(seal_raw)
            if (seal['schedule_sha256'], seal['split']) != (spec['schedule_sha256'], 'calibration'):
                raise ValueError(f'{tag}: mechanism seal does not match calibration schedule')
            for arm in ('shallow', 'deep'):
                raw = z.read(f'pair/{tag}/{tag}-{arm}.run.json')
                receipt = json.loads(raw)
                if (receipt['policy'], receipt['block'], receipt['seal_sha256']) != ('forced', spec['block'], seal_hash):
                    raise ValueError(f'{tag}/{arm}: archived run does not match the registered mechanism')
                sources.append((tag, arm, receipt, plan))
                rows.append(dict(pair=tag, recorded_arm=arm, receipt_sha256=sha256(raw).hexdigest(),
                                 states=replay(receipt, plan, profile['cost'])))
    targets = [r['states'][1] for r in rows]
    all_states = [x for r in rows for x in r['states']]
    optional = {'cost_sensitivity': sensitivity(sources, profile['cost'])} if a.cost_sensitivity else {}
    with a.out.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(dict(scope='same-snapshot policy replay on synthetic forced-action GPU calibration states; no online policy effect',
                              design_sha256=design_hash, archive_sha256=archive_hash, cost_profile_sha256=cost_hash,
                              all_snapshots=len(all_states), all_disagreements=sum(x['differs'] for x in all_states),
                              target_snapshots=len(targets), target_disagreements=sum(x['differs'] for x in targets), arms=rows, **optional),
                         indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
