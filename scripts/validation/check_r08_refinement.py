"""Validate and condense R08 candidate self-test evidence (no blind data)."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
from itertools import permutations
import json
from pathlib import Path
from statistics import median
from subprocess import run
import sys

from scripts.cpu.replay.study_r08_refinement import (
    ABLATIONS, BASE, CANDIDATE, CFG, LOCAL, POLICIES, ROOT, capacity, replacement, records, run_policy, tasks,
)
from restore_edge_select.entry_refinement import EntryRefiner, Refinement
from scripts.validation.archive_source import source_matches

EXPECTED = {'micro':72, 'design':144, 'regression':25, 'historical':1068}


def functional_checks():
    model, scale = capacity.models(capacity.read(capacity.R07_CONFIG))['4b']
    exp = capacity.open_exp(capacity.micro_fixture('shared-kv'),model,16,256*capacity.MIB,scale)
    baseline = capacity.run(exp,BASE)
    decisions = []

    def no_op(i, head, arrived, action, coarse, running, charged, ready, serial, replay):
        assert i < arrived and head <= i and action in exp.eligible[i]
        assert isinstance(ready, frozenset)
        decisions.append(i)
        replay((action,))
        for plan in ((-1,), (*[0]*(arrived-i),0)):
            try:
                replay(plan)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted an illegal or not-yet-arrived replay')
        return action

    untouched = capacity.run(exp,BASE,extra={'entry_refiner':no_op})
    assert decisions and untouched == baseline

    cfg = Refinement(replay_limit=1)
    result, info = run_policy(exp,CANDIDATE,cfg)
    assert capacity.sequence(result) == capacity.sequence(baseline)
    assert max([0, *map(int,info['additional_replays'])]) <= 1
    assert info['routes']['capped-stage'] > 0
    try:
        capacity.run(exp,BASE,horizon=0,extra={'entry_refiner':EntryRefiner(exp,Refinement())})
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported execution context accepted')


def case_metrics(row):
    zero,timed,key = (row[k] for k in ('zero','timed','key'))
    assert len(timed) == 6 and {tuple(t['order']) for t in timed} == set(permutations(POLICIES))
    cap = capacity.read(CFG)['policy']['replay_limit']
    n = len(row['request_ids'])
    repeated = {p:Counter() for p in POLICIES}
    differences = defaultdict(list)
    for trial in timed:
        policies = trial['policies']
        assert set(policies) == set(POLICIES)
        a,b,l = (policies[p] for p in (CANDIDATE,BASE,LOCAL))
        for p,z in policies.items():
            assert len(z['response_ms']) == n == len(z['plan'])
            expected = {i for i,v in enumerate(z['extra_vs_local_ms']) if v>10}
            assert expected == set(z['late'])
            repeated[p].update(z['late'])
            d = z['decision']
            assert 0 <= d['p50_ms'] <= d['p95_ms'] <= d['max_ms'] <= d['total_ms']
            differences[p+'-p50'].append(d['p50_ms'])
            differences[p+'-p95'].append(d['p95_ms'])
            differences[p+'-max'].append(d['max_ms'])
        for name,value in dict(mean=a['mean']-b['mean'], p95=a['p95']-b['p95'],
                               net_local=l['mean']-a['mean'], prior_local=l['mean']-b['mean'],
                               worst_local=a['max_local_extra'],
                               decision=a['decision']['total_ms']-b['decision']['total_ms']).items():
            differences[name].append(value)
    assert row['repeat_tail'] == {p:[i for i,n in sorted(x.items()) if n>=4] for p,x in repeated.items()}
    for z in [v for v in zero.values() if 'additional_replays' in v] + [t['policies'][CANDIDATE] for t in timed]:
        assert max([0,*map(int,z['additional_replays'])]) <= cap
        assert sum(z['additional_replays'].values()) == z['routes'].get('triggered',0)
    a,b,l = (zero[p] for p in (CANDIDATE,BASE,LOCAL))
    ma,mb,ml = (F(x['mean_exact']) for x in (a,b,l))
    m = {k:median(v) for k,v in differences.items()}
    for p in POLICIES:
        m[p+'-peak-across-rounds'] = max(differences[p+'-max'])
    signed = [(x,y) for x,y in zip(differences['net_local'],differences['prior_local'],strict=True) if y>0]
    net = median(x/y for x,y in signed) if signed else None
    new_tail = sorted(set(row['repeat_tail'][CANDIDATE])-set(row['repeat_tail'][BASE]))
    ref = row['oracle']
    lower = F(str(ref.get('lower_exact',ref['lower'])))
    upper = F(str(ref.get('upper_exact',ref.get('upper_bound',ref['upper']))))
    gap_audit = None
    # Refuse to treat a rounded historical point as an exact certificate.
    tolerance = F(0) if row['archive_exact'] else F(1,10**9)
    if not row['archive_exact']:
        lower,upper = lower-tolerance,upper+tolerance
    upper = min(upper,ma,mb,ml)
    room_min = ml-upper
    old_max = mb-lower
    old_min = max(0,mb-upper)
    new_max = ma-lower
    new_min = max(0,ma-upper)
    if room_min > 0 and old_min > 0:
        # With the same unknown optimum O, ratios (m-O)/(l-O) are
        # monotone in O, so evaluate both endpoints instead of mixing them.
        gap_fraction = sorted((new_min/room_min, new_max/(ml-lower)))
        capture = sorted(((ml-ma)/(ml-lower),(ml-ma)/room_min))
        recovered = sorted(((mb-ma)/old_min,(mb-ma)/old_max))
        passes = recovered[0] >= F(1,2) and (new_max <= 1 or gap_fraction[1] <= F(1,20))
        fails = recovered[1] < F(1,2) or (new_min > 1 and gap_fraction[0] > F(1,20))
        gap_audit = dict(gap_ms=[float(new_min),float(new_max)],
                         old_gap_ms=[float(old_min),float(old_max)],
                         gap_fraction=list(map(float,gap_fraction)),
                         recovered_fraction=list(map(float,recovered)),
                         savings_capture_fraction=list(map(float,capture)),
                         bound_kind='exact-rational archive' if row['archive_exact'] else 'rounded archive',
                         outcome='failure-certified' if fails else 'pass-certified' if passes else 'uncertain')
    no_room = row['archive_exact'] and ml == lower
    certified_near = (row['archive_exact'] and ml-upper > 0 and (mb-lower)*20 <= ml-lower)
    known_good = row['archive_exact'] and (mb == lower or certified_near)
    # An exact incumbent remains feasible. Candidate vs old interval is not a
    # proof of optimality when lower < upper.
    if row['archive_exact']:
        assert ma >= lower
    flags = dict(optimal_frozen=row['archive_exact'] and mb == lower, no_room=no_room, certified_near_95=certified_near,
                 good=known_good, mean_fail=m['mean']>1, p95_fail=m['p95']>1,
                 strict_mean_fail=max(differences['mean'])>1, strict_p95_fail=max(differences['p95'])>1,
                 retention_fail=net is not None and net<0.99,
                 strict_retention_fail=bool(signed) and min(x/y for x,y in signed)<0.99,
                 new_repeated_tail=bool(new_tail),
                 zero_opt_fail=row['archive_exact'] and mb == lower and ma>mb)
    extra = {str(i):[t['policies'][CANDIDATE]['extra_vs_local_ms'][i] for t in timed] for i in new_tail}
    return dict(key=key, flags=flags, timed_median=m, net_retention=net,
                zero_gain=float(mb-ma), zero_single_peak=a['max_local_extra'],
                offline=dict(lower=ref.get('lower_exact',ref['lower']),
                             upper=ref.get('upper_exact',ref.get('upper_bound',ref['upper'])), exact=ref['exact']),
                gap_gate=gap_audit,
                frozen_repeat_tail=row['repeat_tail'][BASE],
                candidate_repeat_tail=row['repeat_tail'][CANDIDATE],
                excess_repeat_indices=new_tail, repeat_loss_ms=extra,
                routes=a['routes'], additional_replays=a['additional_replays'],
                ablations={p:float(mb-F(z['mean_exact'])) for p,z in zero.items() if p in ABLATIONS})


def aggregate(rows):
    agg = Counter()
    timing = defaultdict(list)
    routes = Counter()
    replay_hist = Counter()
    for row in rows:
        f = row['flags']
        agg['rows'] += 1
        agg['zero_gain'] += row['zero_gain'] > 1e-9
        agg['zero_regression'] += row['zero_gain'] < -1e-9
        agg['net_gain'] += row['timed_median']['mean'] < -1
        for k,v in f.items():
            agg[k] += v
            if f['good'] and k.endswith('_fail'):
                agg['good_'+k] += v
        replay_hist.update(row['additional_replays'])
        routes.update(row['routes'])
        for k,v in row['timed_median'].items():
            timing[k].append(v)
    triggered = routes['triggered']
    route_rates = dict(trigger=triggered/routes['eligible'] if routes['eligible'] else None,
                      no_change_or_rejected=(triggered-routes['changed'])/triggered if triggered else None)
    return dict(counts=dict(agg), route_rates=route_rates, timing={k:dict(median=median(v),min=min(v),max=max(v)) for k,v in timing.items()},
                routes=dict(routes), additional_replays=dict(replay_hist))


def membership(suite):
    if suite == 'regression':
        return {json.dumps(row['key'],sort_keys=True) for row,_ in tasks(suite)}
    if suite == 'historical':
        patch = capacity.read(replacement.AMENDMENT)
        historical = (r['key'] for r in records('r08-holdout').values()
                      if (r['key']['trace'],r['key']['window']) != (patch['trace'],patch['exclude_window']))
        return {json.dumps(k,sort_keys=True) for k in historical} | {
            json.dumps(r['key'],sort_keys=True)
            for part in ('r08-replacement','r08-native') for r in records(part).values()}
    return {json.dumps(r['key'],sort_keys=True) for r in records('r08-'+suite).values()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--reproduce',action='store_true',
                   help='produce --results from the full archived suite before validating; requires new paths')
    args = p.parse_args()
    if args.reproduce and (args.results is None or args.out is None):
        p.error('--reproduce requires --results and --out')
    functional_checks()
    print('R08 refiner functional checks passed')
    if not args.results:
        return
    if args.reproduce:
        run([sys.executable,'-m','scripts.cpu.replay.study_r08_refinement','all','--out',str(args.results)],
            cwd=ROOT,check=True)
    meta = capacity.read(args.results.with_name(args.results.name+'.manifest.json'))
    assert meta['output'] == capacity.file_hash(args.results)
    pins = dict(config=CFG, entry_refinement=ROOT/'src/restore_edge_select/entry_refinement.py',
                simulator=ROOT/'src/restore_edge_select/simulator.py',
                evaluator=ROOT/'scripts/cpu/replay/study_r08_refinement.py')
    assert all(source_matches(v,meta[k],historical=not args.reproduce) for k,v in pins.items())
    cohorts = defaultdict(list)
    by_model = defaultdict(list)
    by_source = defaultdict(list)
    keys = set()
    opener = gzip.open if args.results.suffix=='.gz' else open
    with opener(args.results,'rt',encoding='utf-8') as stream:
        for raw in stream:
            row = json.loads(raw)
            k = (row['suite'], json.dumps(row['key'],sort_keys=True))
            assert k not in keys
            keys.add(k)
            item = case_metrics(row)
            cohorts[row['suite']].append(item)
            by_model[row['suite'],row['key'].get('model','r07-4b'),row['key'].get('cost','archived')].append(item)
            if 'window' in row['key']:
                tier = 'measured-matrix' if 'cost' in row['key'] else 'archived'
                trace = row['key'].get('trace','qwen_traceB_blksz_16.jsonl')
                by_source[row['suite'],tier,trace,str(row['key']['window'])].append(item)
    assert {k:len(v) for k,v in cohorts.items()} == meta['rows']
    assert all(len(v) == EXPECTED[k] for k,v in cohorts.items())
    for suite in cohorts:
        expected = membership(suite)
        assert expected == {k for s,k in keys if s == suite}, suite
    report = dict(provenance=meta, suites={k:aggregate(v) for k,v in cohorts.items()},
                  cohorts={':'.join(k):aggregate(v) for k,v in by_model.items()},
                  regression=cohorts.get('regression', []),
                  source_windows={':'.join(k):dict(points=len(v),
                      net_gain=sum(r['timed_median']['mean'] < -1 for r in v),
                      worst_mean=max(r['timed_median']['mean'] for r in v),
                      worst_p95=max(r['timed_median']['p95'] for r in v),
                      new_tail=sum(r['flags']['new_repeated_tail'] for r in v)) for k,v in by_source.items()})
    if 'historical' in cohorts:
        hist = cohorts['historical']
        report['historical_worst'] = dict(mean=max(hist,key=lambda r:r['timed_median']['mean']),
                                         p95=max(hist,key=lambda r:r['timed_median']['p95']),
                                         repeated_tail=max(hist,key=lambda r:len(r['excess_repeat_indices'])))
    if args.out is not None:
        with args.out.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(report,stream,indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({k:v['counts'] for k,v in report['suites'].items()},indent=2))


if __name__ == '__main__':
    main()
