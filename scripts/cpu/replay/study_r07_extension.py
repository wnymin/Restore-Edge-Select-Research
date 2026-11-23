"""R07 extension: paired models, workload and sharing; unchanged simulator and policy."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

from scripts.cpu.dynamic.dynamic_cpu_inputs import (
    FIXTURE, GRID, IO, MANIFEST, MODEL, OVERLAP, ROOT, make_experiment,
    sha256 as file_hash, window_data,
)
from scripts.cpu.replay.study_r07_arrivals import (
    Experiment, POLICIES, pack, paired, read, response, run, score, sequence, transform,
)
from scripts.cpu.replay.study_r07_diagnostics import diagnose
from restore_edge_select.offline_bounds import search

CONFIG = ROOT / 'configs/r07-extension-matrix.json'
SYNTH = ROOT / 'configs/r06-synthetic-gqa-gdn.json'
DESIGN_MODELS = ('4b', 'kv-only', 'state-only', 'compute-only', 'r06-synth')
HELDOUT_MODELS = ('layer-mix', '27b')
TRACE_DIR = ROOT / 'data/raw/bailian'


def structured(spec):
    layers, interval = spec['layers'], spec['full_every']
    assert layers % interval == 0
    full = layers // interval
    linear = layers - full
    gqa, gdn = spec['gqa'], spec['gdn']
    kv = full * 2 * gqa['kv_heads'] * gqa['head_dim'] * 2
    rec = linear * gdn['v_heads'] * gdn['key_dim'] * gdn['value_dim'] * 4
    channels = 2 * gdn['qk_heads'] * gdn['key_dim'] + gdn['v_heads'] * gdn['value_dim']
    conv = linear * channels * (gdn['kernel'] - 1) * 2
    return dict(model_id=spec['model_id'], context_tokens=spec['context_tokens'],
                payload=dict(attention_bytes_per_token=kv, recurrent_bytes_per_checkpoint=rec,
                             conv_bytes_per_checkpoint=conv, checkpoint_bytes=rec+conv))


def models(cfg):
    anchor, synth = read(MODEL)['models'][0], read(SYNTH)['models'][0]
    result = {name:(deepcopy(anchor), F(1)) for name in DESIGN_MODELS}
    result['r06-synth'] = (synth, F(1))
    for name, fields in [('kv-only', ['attention_bytes_per_token']),
                         ('state-only', ['recurrent_bytes_per_checkpoint', 'conv_bytes_per_checkpoint', 'checkpoint_bytes'])]:
        result[name][0]['payload'].update({k:synth['payload'][k] for k in fields})
    result['compute-only'] = (anchor, F(3,4))
    for name, key in [('layer-mix','synthetic_layer_mix'), ('27b','model_27b')]:
        result[name] = (structured(cfg[key]), F(cfg[key]['compute_proxy']))
    return result


def isolate_objects(trace):
    assert not trace['scheduling']['initial_device_object_ids']
    objects = {o['id']:o for o in trace['objects']}
    new, reqs = {}, []
    for r in trace['requests']:
        choices = []
        for c in r['candidates']:
            deps = []
            for oid in c['load_order']:
                aid = f"{r['id']}:{oid}"
                deps.append(aid)
                new.setdefault(aid, dict(objects[oid], id=aid))
            choices.append(dict(c, load_order=deps))
        reqs.append(dict(r, candidates=choices))
    return dict(trace, requests=reqs, objects=list(new.values()))


def change_payload(data, model):
    pay = model['payload']
    reqs = data['requests']
    kv, state = pay['attention_bytes_per_token'], pay['checkpoint_bytes']
    longest = max(sum(t['count'] for t in r['input_token_runs']) for r in reqs)
    m = data['memory']
    mib = data['units']['bytes_per_mib']
    block = m['kv_block_tokens']
    need = longest*kv + state + m['workspace_mib']*mib + ((longest+block-1)//block+1)*m['extra_alignment_bytes']
    execution = (need+mib-1)//mib
    return dict(data, memory=dict(m, execution_mib=execution, device_mib=m['fixed_mib']+execution+384),
                objects=[dict(o, bytes=(o['end_token_inclusive']-o['start_token_exclusive'])*kv
                                   if o['kind']=='attention_kv' else state) for o in data['objects']])


def experiment(data, key, model, scale):
    trace = transform(change_payload(data, model), key['arrival'])
    if key['sharing'] == 'private':
        trace = isolate_objects(trace)
    exp = make_experiment(read(FIXTURE), model, trace, 'ssd', key['rate'], key['budget'])
    if scale != 1:
        grid = exp.cost['recompute']
        changed = dict(grid, samples=[dict(row, ms=[str(F(str(x))*scale) for x in row['ms']])
                                      for row in grid['samples']])
        exp = Experiment(dict(exp.trace, cost=dict(exp.cost, recompute=changed)), model, key['budget'])
    return exp


def specs(cfg, split):
    rates, main, second = cfg['rates'], cfg['main_budget_mib'], cfg['secondary_budget_mib']
    if split == 'design':
        yield from ((m,r,'original','natural',main) for m,r in product(DESIGN_MODELS,rates))
        yield '4b', 16, 'original', 'private', main
        return
    for m,r in product((*DESIGN_MODELS,*HELDOUT_MODELS),rates):
        yield m,r,'original','natural',main
    for m,r in product(('4b',*HELDOUT_MODELS),rates[1:]):
        for a in ('knee-even','knee-burst'):
            yield m,r,a,'natural',main
        yield m,r,'original','private',main
    for r in rates[1:]:
        yield '27b',r,'original','natural',second


def cases(cfg, split):
    manifest = {e['name']:e for e in read(MANIFEST)['files']}
    anchor = read(MODEL)['models'][0]
    fixture = read(FIXTURE)
    model_map = models(cfg)
    for trace, windows in cfg[split].items():
        path = TRACE_DIR / trace
        if path.stat().st_size != manifest[trace]['bytes'] or file_hash(path) != manifest[trace]['sha256']:
            raise ValueError(f'Unpinned input: {trace}')
        for window, _, _, data in window_data(TRACE_DIR,trace,windows,fixture,anchor,[('split',F(1))]):
            for m,r,a,s,b in specs(cfg,split):
                key = dict(trace=trace,window=window,model=m,rate=r,arrival=a,sharing=s,budget=b)
                yield key,data,*model_map[m],cfg['node_budget']


def route_audit(exp, expected):
    """Instrument the unchanged screen only in the zero-overhead evaluation."""
    routes = Counter()
    max_followers = 0

    def observe(row):
        nonlocal max_followers
        v = sys._getframe(1).f_locals
        routes[v['i']] += 1
        max_followers = max(max_followers,v['arrived']-v['i']-1)
        return score(row)

    actual = run(exp,'entry-replay',extra={'score_candidate':observe})
    assert sequence(actual) == sequence(expected)
    # Screen entries and evaluated candidate rollouts are different quantities.
    return dict(single=sum(len(c)==1 for c in exp.eligible), screened=len(routes), max_followers=max_followers,
                bypassed_multi=sum(len(c)>1 for c in exp.eligible)-len(routes),
                rollouts=actual['decision_replays'], pruned=actual['pruned_rollouts'])


def evaluate(task):
    key, data, model, scale, nodes = task
    exp = experiment(data,key,model,scale)
    z = {p:run(exp,p) for p in POLICIES}
    base, main = z['local-cost'],z['entry-replay']
    assert sequence(main)==sequence(z['entry']), key
    assert exp.run('fixed', boundaries=main['boundaries_tokens'])['completion_ms']==main['completion_ms'], key
    lo,hi,plan,used = search(exp,list(z.values()),nodes,resource_floor=True,dominance=True)
    n=len(exp.requests)
    assert lo <= hi <= min(base['response_sum_ms'],main['response_sum_ms'])
    assert exp.run('fixed',boundaries=plan)['response_sum_ms']==hi
    baseline = response(exp,base)
    freq = Counter(o for cs in exp.candidates for o in {i for deps in cs.values() for i in deps})
    r = dict(key=key,n=n, oracle=dict(lower=float(lo/n),upper=float(hi/n),exact=lo==hi,nodes=used,
                                      lower_exact=str(lo/n),upper_exact=str(hi/n),plan=plan),
             zero={p:pack(exp,v,baseline) for p,v in z.items()}, plan=main['boundaries_tokens'],
             routes=route_audit(exp,main), baseline_exact=str(base['response_mean_ms']),
             main_exact=str(main['response_mean_ms']),
             input=dict(request_ids=[x['id'] for x in exp.requests], lengths=exp.lengths,
                        arrivals_hash=sha256('|'.join(map(str,exp.arrivals)).encode()).hexdigest(),
                        arrival_span=float(exp.arrivals[-1]-exp.arrivals[0]),
                        eligible_hash=sha256(json.dumps(exp.eligible).encode()).hexdigest(),
                        payload=model['payload'],compute_factor=str(scale),execution_bytes=exp.execution,
                        shared_objects=sum(v>1 for v in freq.values()), referenced_bytes=exp.size(freq),
                        stageable_nonzero=sum(bool(deps) and exp.size(deps)<=exp.budget
                                              for cs in exp.candidates for deps in cs.values()),
                        peak_bytes=main['peak_prefetch_bytes'],
                        transferred_bytes=main['transferred_bytes_by_path']))
    if hi < main['response_sum_ms']:
        witness=diagnose(exp,r)
        r['diagnostic'] = witness[0] if isinstance(witness,tuple) else witness
    return r


def preflight(cfg):
    """Positive acceptance/negative dependency tests, not a sliding-window oracle."""
    base=read(FIXTURE)
    anchor=read(MODEL)['models'][0]
    positive={}
    for name,(model,_) in models(cfg).items():
        spec=change_payload(base,model)
        exp=Experiment(spec,model,128)
        exact=exp.offline()
        z=[run(exp,p) for p in POLICIES]
        lo,hi,plan,_=search(exp,z,cfg['node_budget'],resource_floor=True,dominance=True)
        assert lo==hi==exact['response_sum_ms']
        assert sequence(z[1])==sequence(z[2])
        positive[name]=dict(oracle_exact=str(lo), plan=plan, eligible=exp.eligible, payload=model['payload'])
    negatives={}
    for case in ('rolling-kv-only','state-missing','checkpoint-misaligned'):
        spec=deepcopy(base)
        r,c=next((r,c) for r in spec['requests'] for c in r['candidates'] if len(c['load_order'])>=3)
        if case=='rolling-kv-only': c['load_order'].pop(0)
        elif case=='state-missing': c['load_order'].pop()
        else: c['boundary_tokens'] -= 1
        try:
            Experiment(spec,anchor,128)
        except ValueError as e:
            negatives[case]=str(e)
        else:
            raise AssertionError(case)
    return dict(compatible=positive, unsupported=negatives)


def write_rows(rows, target):
    target.parent.mkdir(exist_ok=True,parents=True)
    with target.open('x', encoding='utf-8',newline='\n') as f:
        for n,r in enumerate(rows,1):
            f.write(json.dumps(r,separators=(',',':'))+'\n'); f.flush()
            if n%25==0 or n==1: print(n,r['key'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('split',choices=('design','holdout','timed','preflight'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--untimed',type=Path)
    a=p.parse_args()
    cfg=read(CONFIG)
    if a.split=='preflight':
        with a.out.open('x',encoding='utf-8') as f: json.dump(preflight(cfg),f,indent=2)
    elif a.split=='timed':
        if a.jobs != 1 or a.untimed is None: p.error('timed requires --untimed and sequential execution')
        untimed=[json.loads(line) for line in a.untimed.read_text(encoding='utf-8').splitlines()]
        def timing():
            for idx,(task,r) in enumerate(zip(cases(cfg,'holdout'),untimed,strict=True)):
                key,data,model,scale,_=task
                assert key==r['key']
                yield dict(key=key,timed=paired(experiment(data,key,model,scale),cfg['timing_repeats'],idx))
        write_rows(timing(),a.out)
    elif a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as pool:
            write_rows(pool.map(evaluate,cases(cfg,a.split),chunksize=2),a.out)
    else:
        write_rows(map(evaluate,cases(cfg,a.split)),a.out)
    dependencies=(Path(__file__).resolve(),CONFIG,SYNTH,MODEL,MANIFEST,FIXTURE,GRID,IO,OVERLAP,
                  *(ROOT/'src/restore_edge_select').glob('*.py'),
                  *(ROOT/f'scripts/cpu/replay/{n}.py' for n in ('study_r07_arrivals','study_r07_diagnostics',
                                                              'study_bailian','study_boundary_factors')),
                  ROOT/'scripts/cpu/dynamic/dynamic_cpu_inputs.py')
    meta=dict(split=a.split,platform=platform.platform(),python=platform.python_version(),jobs=a.jobs,
              input_sha256=file_hash(a.untimed) if a.untimed else None,output_sha256=file_hash(a.out),
              files={f.relative_to(ROOT).as_posix():file_hash(f) for f in dependencies})
    a.out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
