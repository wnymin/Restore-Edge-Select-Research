"""R08 remaining-line archive audit: reaggregate raw R07 service and CPU evidence."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, mean

from scripts.cpu.replay.study_r08_counterexamples import B, arrivals, capacity, preflight

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence'
MS = 10**6


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, name):
        raw = (E / name).read_bytes()
        self.hashes[name] = sha256(raw).hexdigest()
        if name.endswith('.gz'):
            return list(map(json.loads,gzip.decompress(raw).splitlines()))
        return json.loads(raw)


def interval(peer, start=None, end=None):
    times = peer['tokens_ns']
    gaps = [b-a for a,b in zip(times,times[1:]) if start is None or (a<end and b>start)]
    return max(gaps) / MS


def elapsed(a,b):
    return (b-a)/MS


def service(inputs):
    arms = {}
    for arm in ('4b-cb512','4b-static512'):
        _,*rows = inputs.read(f'r07-live-{arm}.jsonl.gz')
        pairs = {(r['block'],r['mode'],r['depth']):r for r in rows if r['block']>=0}
        for mode in ('active','queue'):
            shifts = []
            for b in range(8):
                def measures(depth):
                    r = pairs[b,mode,depth]
                    peer = r['peer']
                    assert r['target']['timings']['cache_n']==depth
                    return dict(target=elapsed(r['restore_start'],r['target']['tokens_ns'][0]),
                                handler=r['restore']['timings']['restore_ms'],
                                peer_stream=elapsed(peer['tokens_ns'][0],peer['tokens_ns'][-1]),
                                max_all=interval(peer),max_overlap=interval(peer,r['restore_start'],r['restore_end']))
                shallow,deep = measures(512),measures(1024)
                shifts.append({k:deep[k]-shallow[k] for k in deep})
            arms[f'{arm}/{mode}'] = dict(medians={k:median(r[k] for r in shifts) for k in shifts[0]},
                                         blocks=shifts, target_win_peer_max_over_10=sum(
                                            r['target']<0 and r['max_all']>10 for r in shifts))
    assert all(arms[f'4b-static512/{m}']['medians']['target']<0<arms[f'4b-static512/{m}']['medians']['max_all']
               and arms[f'4b-cb512/{m}']['medians']['max_all']<0<arms[f'4b-cb512/{m}']['medians']['max_overlap']
               for m in ('active','queue'))
    _,*obs = inputs.read('r07-live-observer.jsonl.gz')
    obs = [r for r in obs if r['block']>=0]
    by_arm = {(r['block'],r['mode'],r['depth']):r for r in obs}
    blocks = sorted({r['block'] for r in obs})
    missing = sorted({r['block'] for r in obs if len(r['peer']['tokens_ns']) != r['peer']['length']})
    sensitivity = {}
    for mode in ('restore','clean'):
        delta = {b:interval(by_arm[b,mode,1024]['peer'])-interval(by_arm[b,mode,512]['peer']) for b in blocks}
        sensitivity[mode] = dict(all=median(delta.values()), full_chunks=median(v for b,v in delta.items() if b not in missing), block_delta=delta)
    probes = [r for r in obs if r['mode']=='probe']
    poll = [sum(elapsed(o['start'],o['end']) for o in r['observe']) for r in probes]
    slot_progress = []
    for r in probes:
        slot_progress.append([next(s for s in o['slots'] if s['id']==1)['n_prompt_tokens'] for o in r['observe']])
    assert all(tokens==[0,r['depth']] for tokens,r in zip(slot_progress,probes,strict=True))
    def target(b,m,d):
        r=by_arm[b,m,d]
        return elapsed(r['trigger'], r['target']['tokens_ns'][0])
    probe_delta = {d:median(target(b,'probe',d)-target(b,'clean',d) for b in blocks) for d in (512,1024)}
    queue = inputs.read('r07-live-queue-interface.json')['probes']
    wait = [x['metrics']['llamacpp:requests_deferred'] for x in queue]
    active = [[s['id_task'] for s in q['snapshot']['slots']] for q in queue]
    assert wait==[0,1,2] and len({tuple(ids) for ids in active})==1
    _,*service16 = inputs.read('r07-live-16l-cb256-holdout.jsonl.gz')
    active16 = [r for r in service16 if r['block']>=0 and r['mode'] in ('active','queue')]
    same_inputs = {r['block']:r['prompt_hashes'][1] for r in active16}
    solo = {}
    for arm in ('4b','16l'):
        _,*attempts = inputs.read(f'r07-live-solo-{arm}.jsonl.gz')
        attempts = [r for r in attempts if r['block']>=0]
        assert {r['block']:r['prompt_hash'] for r in attempts}==same_inputs
        solo[arm] = dict(n=len(attempts), failed=[r['block'] for r in attempts if 'error' in r],
                         observed_errors=sorted({r['error'] for r in attempts if 'error' in r}))
    assert sum('failure' in r for r in active16)==21 and len(solo['16l']['failed'])==5 and not solo['4b']['failed']
    parser = inputs.read('r08-parse-control.json')
    env = inputs.read('r07-live-environment.json')
    assert parser['binary_sha256']==dict(env['binary_sha256'],model=env['model_sha256']['r08-16-layer-calibration.gguf'])
    formal=[p for p in parser['pairs'] if p['block']>=0]
    assert len(formal)==8 and {p['block']:p['prompt_hash'] for p in formal}==same_inputs
    for p in formal:
        a,b=(p[k]['token_ids'] for k in ('default','skip'))
        common=min(len(a),len(b))
        assert common>0 and p['common_tokens']==common and p['common_prefix_equal'] and a[:common]==b[:common]
    control={arm:dict(failed=[p['block'] for p in formal if p[arm]['predicted_n']!=64],
                     error_codes=sorted({e['code'] for p in formal for e in p[arm]['errors']}))
             for arm in ('default','skip')}
    assert all(v['failed']==solo['16l']['failed'] and v['error_codes']==[500] for v in control.values())
    return dict(paired=arms, probes=dict(n=len(probes),visible_depth=len(slot_progress), median_poll_ms=median(poll),
        target_extra_ms=probe_delta,pending=wait,activity_task_ids=active),
        isolated=dict(peer_attempts=len(active16),peer_failures=sum('failure' in r for r in active16), solo=solo),
        sse=dict(incomplete_blocks=missing, full_blocks=len(blocks)-len(missing), deltas=sensitivity),
        parser_control=dict(n=len(formal), matching_observed_prefixes=len(formal), arms=control))


def cpu(inputs):
    rows = inputs.read('r07-holdout.jsonl.gz')
    sets = defaultdict(dict)
    for r in rows:
        k=r['key']; sets[k['window'],k['rate'],k['budget']][k['variant']]=r
    contrasts = {}
    for even,burst in (('even','burst'),('knee-even','knee-burst')):
        for scale in (1,12,16):
            ds=[]
            for (w,s,_), grouping in sorted(sets.items()):
                if s!=scale: continue
                a,b=(grouping[k] for k in (even,burst))
                assert a['n']==b['n'] and a['arrival_span']==b['arrival_span']
                gap=lambda r:r['zero']['entry-replay']['mean']-r['oracle']['upper']
                vis=lambda r: r['visibility']
                ds.append(dict(window=w, mean_gap_diff=gap(b)-gap(a), visibility_even=vis(a), visibility_burst=vis(b)))
            contrasts[f'{even}/{burst}/{scale}']=dict(n=len(ds), mean_diff=mean(r['mean_gap_diff'] for r in ds), windows=ds)
    assert contrasts['knee-even/knee-burst/12']['mean_diff']<0<contrasts['knee-even/knee-burst/16']['mean_diff']
    rows = inputs.read('r08-holdout.jsonl.gz')
    keyed = {(r['key']['window'],r['key']['model'],r['key']['rate'],r['key']['level']):r for r in rows
             if r['key']['trace']=='qwen_traceB_blksz_16.jsonl' and r['key']['section']=='fixed'}
    caps = {}
    for cap in ('176','180','384'):
        r=keyed[481,'27b',16,cap]
        m,b=F(r['main_exact']),F(r['base_exact']); low=F(r['oracle']['lower']); high=F(r['oracle']['upper_bound'])
        caps[cap] = dict(fitting=r['feasible_by_size']['cold'],reserved=r['activity']['entry-replay']['reserved'],
                         savings_ms=float(b-m),frozen_gap_ms=[float(m-high),float(m-low)],baseline_gap_ms=[float(b-high),float(b-low)])
    cfg = capacity.read(capacity.CONFIG)
    cfg['holdout'] = {B:[481]}
    for key,data,model,scale in capacity.cases(cfg,'holdout'):
        label=key['level']
        if key['model']!='27b' or key['rate']!=16 or key['section']!='fixed' or label not in caps:
            continue
        exp=capacity.open_exp(data,model,key['rate'],key['capacity_bytes'],scale)
        z=preflight(exp,keyed[481,'27b',16,label])
        own=defaultdict(list)
        copy={t['request']:t for t in z['transfers'] if t['mode']=='handoff'}
        for t in z['transfers']:
            if t['mode']=='prefetch': own[t['request']].append(t)
        phases=[]
        for i,rid in enumerate(z['request_ids']):
            c=z['boundaries_tokens'][i]; ts=own[rid]; reserve=z['prefetch_reserved_at_ms'][i]
            if reserve is None and not ts: continue
            compute=exp.arrivals[i]+z['queue_wait_ms'][i]+z['head_to_compute_ms'][i]
            dispatch=copy[rid]['start_ms'] if rid in copy else compute
            dep=set(exp.candidates[i][c]); issued=sorted(t['object'] for t in ts)
            ready=sorted(t['object'] for t in ts if t['end_ms']<=dispatch and t['object'] in dep)
            assert reserve is not None and len(ready)==len(issued)
            phases.append(dict(request=rid,boundary=c,reserved_ms=str(reserve),dispatch_ms=str(dispatch),
                own_issued=issued, own_ready_and_required=ready, has_all_deps_from_own_prefetch=set(ready)==dep-exp.initial))
        assert len(phases)==caps[label]['reserved']
        caps[label].update(prefetch_issue=sum(bool(r['own_issued']) for r in phases),
            own_ready_and_consumable=sum(bool(r['own_ready_and_required']) for r in phases),phases=phases)
    assert [caps[x]['reserved'] for x in ('176','180')]==[0,13] and caps['384']['savings_ms']>680
    checks = []
    for key in ((481,'4b',12,'176'),(482,'27b',16,'384')):
        r = keyed[key]; old=F(r['main_exact']); opt=F(r['oracle']['upper_bound']); d=r['diagnostic']
        target=d['target']; best=d['all_seen_choice'] if key[0]==481 else d['joint_choice']
        forced=F(str(d['closed_loop_forced_mean_ms'][str(best)]))
        checks.append(dict(key=r['key'],target=target,frozen=d['current'],four=d['window4_choice'],all_seen=d['all_seen_choice'],
            joint=d['joint_choice'],oracle_exact=r['oracle']['exact'],gap_ms=float(old-opt),force_gain_ms=float(old-forced),
            residual_ms=max(0,float(forced-opt))))
    timed = inputs.read('r08-timed.jsonl.gz')
    row = next(r for r in timed if r['key']==keyed[481,'27b',16,'512']['key'])
    loops = [t['policies']['entry-replay'] for t in row['timed']]
    assert len(loops)==6 and all(t['late_indices']==[15,16] for t in loops)
    return dict(arrival=contrasts,capacity=caps, mean_witnesses=checks,
                tail_witness=dict(late_indices=[t['late_indices'] for t in loops], worst_ms=max(t['max_loss'] for t in loops)))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True,type=Path); a=p.parse_args()
    inputs=Inputs(); output=dict(cpu=cpu(inputs),service=service(inputs), input_sha256=inputs.hashes)
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(output,f,sort_keys=True,indent=2); f.write('\n')


if __name__=='__main__':
    main()
