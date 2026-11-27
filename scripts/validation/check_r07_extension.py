"""Summarize, verify and archive the R07 extension; timing is never recomputed by default."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
from hashlib import sha256
import json
import inspect
import sys
from pathlib import Path
from statistics import mean, median

from scripts.cpu.replay.study_r07_arrivals import POLICIES, response, run, sequence
from scripts.cpu.replay.study_r07_extension import (
    CONFIG, FIXTURE, ROOT, Experiment, cases, change_payload, evaluate, experiment, file_hash, models, read, specs,
)
from scripts.validation.archive_source import source_matches

NAMES = ('preflight','design','holdout','timed')
MAIN = 'entry-replay'
TRACE = lambda key: key['trace'].split('_')[1]
EVIDENCE = ROOT/'docs/evidence'


def rows(data):
    return [json.loads(s) for s in data.splitlines()]


def identity(key):
    return tuple(sorted(key.items()))


def opt(row):
    o = row['oracle']
    return tuple(map(F,(row['main_exact'],row['baseline_exact'],o['lower_exact'],o['upper_exact'])))


def persistent(timed,cfg):
    hit=Counter(i for t in timed for i in t['policies'][MAIN]['late_indices'])
    return sorted(i for i,n in hit.items() if n>=cfg['thresholds']['persistent_repeats'])


def manual_contract(cfg):
    fixture=read(FIXTURE)
    r=next(r for r in fixture['requests'] if r['id']=='B')
    spec=dict(fixture,requests=[dict(r,arrival_ms=0)])
    obj={o['id']:o for o in fixture['objects']}
    coeff=fixture['cost']['recompute']
    a=F(*coeff['a_ms_per_token_rational']); b=F(*coeff['b_ms_per_pair_rational'])
    n=sum(t['count'] for t in r['input_token_runs'])
    out={}
    for key,(model,_) in models(cfg).items():
        trace=change_payload(spec,model)
        exp=Experiment(trace,model,128)
        checks={}
        for cand in r['candidates']:
            c=cand['boundary_tokens']
            # One request, cold cache, serial I/O, then a single compute slot;
            # no concurrent request, checkpoint copy or overlap is possible.
            transfer=F(0)
            for oid in cand['load_order']:
                o=obj[oid]
                tier={'RAM':'ram_to_device','SSD':'ssd_to_device'}[o['tier']]
                path=fixture['cost'][tier]
                payload=model['payload']
                size=(payload['checkpoint_bytes'] if o['kind']!='attention_kv' else
                      payload['attention_bytes_per_token']*(o['end_token_inclusive']-o['start_token_exclusive']))
                transfer+=F(path['startup_ms'])+F(size,path['bytes_per_ms'])
            expected=transfer+a*(n-c)+b*(n*(n+1)-c*(c+1))/2
            assert exp.run('fixed',boundaries=(c,))['completion_ms'][0]==expected,(key,c)
            checks[str(c)]=str(expected)
        assert exp.offline()['response_sum_ms']==min(map(F,checks.values()))
        out[key]=checks
    return out


def validate(design, holdout, timing, preflight, cfg):
    assert set(preflight['compatible'])=={'4b','kv-only','state-only','compute-only','r06-synth','layer-mix','27b'}
    assert set(preflight['unsupported'])=={'rolling-kv-only','state-missing','checkpoint-misaligned'}
    assert preflight['compatible']['27b']['payload']['checkpoint_bytes']==153944064
    for split, data in [('design',design),('holdout',holdout)]:
        expected = [identity(dict(trace=t, window=w, model=m, rate=r, arrival=a, sharing=s, budget=b))
                    for t,ws in cfg[split].items() for w in ws for m,r,a,s,b in specs(cfg,split)]
        assert len(set(expected))==len(expected)
        assert list(map(lambda x:identity(x['key']),data))==expected
        lookup={identity(d['key']):d for d in data}
        for d in data:
            eye,k = d['input'], d['key']
            a,b,l,u = opt(d)
            assert l<=u<=min(a,b) and d['oracle']['exact']==(l==u)
            assert d['oracle']['nodes']<=cfg['node_budget']
            assert len(d['plan'])==len(d['oracle']['plan'])==d['n']==len(eye['request_ids'])==len(eye['lengths'])
            assert d['zero']['entry']['sequence']==d['zero'][MAIN]['sequence']
            assert sum(d['routes'][f] for f in ('single','screened','bypassed_multi'))==d['n']
            assert eye['peak_bytes']<=k['budget']*2**20
            st=dict(k,model='4b',arrival='original',sharing='natural',budget=cfg['main_budget_mib'])
            base=lookup[identity(st)]['input']
            for name in ('request_ids','lengths','eligible_hash'):
                assert eye[name]==base[name]
            if k['arrival']=='original': assert eye['arrivals_hash']==base['arrivals_hash']
            if k['sharing']=='private':
                natural=lookup[identity(dict(k,sharing='natural'))]['input']
                assert eye['shared_objects']==0 and eye['referenced_bytes']>=natural['referenced_bytes']
            else: assert eye['shared_objects']==base['shared_objects']
            if k['arrival']=='knee-burst':
                other=lookup[identity(dict(k,arrival='knee-even'))]['input']
                assert eye['arrival_span']==other['arrival_span']
        base_mod=preflight['compatible']['4b']['payload']
        for name, fields in [('kv-only',{'attention_bytes_per_token'}),
                             ('state-only',set(base_mod)-{'attention_bytes_per_token'}),('compute-only',set())]:
            assert {f for f,v in preflight['compatible'][name]['payload'].items() if v!=base_mod[f]}==fields
    for r,t in zip(holdout,timing,strict=True):
        assert r['key']==t['key']
        assert len({tuple(k['order']) for k in t['timed']})==len(t['timed'])==cfg['timing_repeats']==6
        for trial in t['timed']:
            assert set(trial['order'])==set(trial['policies'])==set(POLICIES)
            for p,v in trial['policies'].items():
                assert len(v['late_indices'])==v['losses_gt10'] and v['decision']>0
                assert abs(v['mean_loss']-(v['mean']-trial['policies']['local-cost']['mean']))<1e-7


def check_summary(data, cfg):
    groups=defaultdict(list)
    w=cfg['thresholds']
    tkey = lambda x: (x['trace'],x['window'])
    classifications=defaultdict(list)
    for row in data:
        k=row['key']
        for axis in ('all-traces', TRACE(k)):
            groups[k['model'],k['rate'],k['arrival'],k['sharing'],k['budget'],axis].append(row)
        if not (d:=row.get('diagnostic')): continue
        if 'classification' in d:
            labels=[d['classification']]
        else:
            labels=[]
            cur=d['current']; ref=d['closed_loop_forced_mean_ms'][str(cur)]
            for kind,choice in [('screen',d['window4_choice']),('visibility',d['all_seen_choice']),('joint',d['joint_choice'])]:
                delta=ref-d['closed_loop_forced_mean_ms'][str(choice)]
                if delta>1e-7 and (kind=='screen' or (kind=='visibility' and d['window4_choice']==cur)
                        or (kind=='joint' and d['all_seen_choice']==cur and d['window4_choice']==cur)):
                    labels.append(kind)
            if not labels: labels=['unresolved']
        for label in labels: classifications[label].append(k)
    table=[]
    for key,group in groups.items():
        m,r,a,s,budget,axis=key
        vals = list(map(opt,group))
        v,b,l,u=map(mean,zip(*vals))
        g=(v-u,v-l); h=(b-u,b-l)
        cap=[]
        for end in h:
            cap.append(100*float((b-v)/end) if end>0 else None)
        caps=sorted(c for c in cap if c is not None) if None not in cap else cap
        material=[row['key'] for row,(a0,b0,l0,u0) in zip(group,vals,strict=True)
                  if a0-u0>w['mean_gap_ms'] and b0-l0>0 and (a0-u0)/(b0-l0)>w['gap_fraction']]
        tmean=[]; tp95=[]; decision=[]; delta_decision=[]; tails=[]
        for row in group:
            trials=row['timed']
            def metric(name,comp='local-cost'):
                return median(t['policies'][MAIN][name]-t['policies'][comp][name] for t in trials)
            tmean.append(metric('mean')); tp95.append(metric('p95'))
            decision.append(median(t['policies'][MAIN]['decision'] for t in trials))
            delta_decision.append(metric('decision','entry'))
            if hit:=persistent(trials,cfg): tails.append(dict(key=row['key'],requests=hit))
        quality=('no-headroom' if h[1]==0 else 'material-gap' if len(material)>=w['independent_windows']
                 else 'robust-mean' if h[0]>0 and (g[1]<=w['mean_gap_ms'] or g[1]/h[0]<=w['gap_fraction'])
                 else 'unresolved')
        table.append(dict(model=m,rate=r,arrival=a,sharing=s,budget=budget,axis=axis,n=len(group),
                          exact=sum(x['oracle']['exact'] for x in group),optimum=sum(a==l for a,b,l,u in vals),
                          no_headroom=sum(b==l for a,b,l,u in vals),headroom=[float(i) for i in h],
                          gap=[float(i) for i in g],capture=caps,material=material,quality=quality,
                          timed_mean_ms=mean(tmean),timed_p95_ms=mean(tp95),
                          timed_mean_over=sum(x>w['timed_mean_loss_ms'] for x in tmean),
                          timed_p95_over=sum(x>w['timed_p95_loss_ms'] for x in tp95),
                          timed_max_request_ms=max(t['policies'][MAIN]['max_loss'] for row in group for t in row['timed']),
                          decision_ms=mean(decision),decision_vs_entry_ms=mean(delta_decision),tails=tails,
                          zero_tail=sum(row['zero'][MAIN]['losses_gt10']>0 for row in group),
                          stageable=mean(row['input']['stageable_nonzero'] for row in group),
                          screened=sum(row['routes']['screened'] for row in group),
                          bypassed_multi=sum(row['routes']['bypassed_multi'] for row in group),
                          rollouts=sum(row['routes']['rollouts'] for row in group)))
    # Interventions are paired per source, then aggregated; do not average percentages.
    lookup={identity(r['key']):r for r in data}
    contrasts=defaultdict(list)
    interactions=defaultdict(list)
    for row in data:
        k=row['key']; other=None; name=None
        if k['arrival']=='knee-burst':
            name='burst-minus-even'; other=dict(k,arrival='knee-even')
        elif k['sharing']=='private':
            name='private-minus-natural'; other=dict(k,sharing='natural')
        elif k['budget']==cfg['secondary_budget_mib']:
            name='capacity-256-minus-128'; other=dict(k,budget=cfg['main_budget_mib'])
        elif k['arrival']=='original' and k['model']!='4b':
            name='model-minus-4b'; other=dict(k,model='4b')
        if other is not None:
            a,_,l,u=opt(row); x,_,p,q=opt(lookup[identity(other)])
            pair=(a-u-(x-p),a-l-(x-q))
            contrasts[k['model'],k['rate'],name].append(pair)
            if k['model']!='4b' and name in ('burst-minus-even','private-minus-natural'):
                v,_,j,h=opt(lookup[identity(dict(k,model='4b'))])
                y,_,f,g=opt(lookup[identity(dict(other,model='4b'))])
                change_4b=(v-h-(y-f),v-j-(y-g))
                interactions[k['model'],k['rate'],name].append((pair[0]-change_4b[1],pair[1]-change_4b[0]))
    def aggregate(items):
        return [dict(model=m,rate=r,kind=n,n=len(pairs),difference=list(map(float,map(mean,zip(*pairs)))))
                for (m,r,n),pairs in items.items()]
    return dict(table=table, contrasts=aggregate(contrasts), interactions=aggregate(interactions),
                diagnostics=dict(classifications))


def tail_witnesses(data, cfg):
    hits=[]
    seen=set()
    tests=0
    for task,row in zip(cases(cfg,'holdout'),data,strict=True):
        k,spec,model,factor,_=task
        assert k==row['key']
        eligible=set(row['zero'][MAIN]['late_indices'])&set(persistent(row['timed'],cfg))
        group=(k['trace'],k['window'],k['model'])
        if not eligible or group in seen: continue
        exp=experiment(spec,k,model,factor)
        a,b=(run(exp,p) for p in (MAIN,'local-cost'))
        ar,br=response(exp,a),response(exp,b)
        ap,bp=a['boundaries_tokens'],b['boundaries_tokens']
        j=next((j for j,(x,y) in enumerate(zip(ap,bp,strict=True)) if x!=y),len(ap))
        possible=[i for i in sorted(eligible) if i<j<len(ap)]
        if not possible: continue
        tests+=1
        z=run(exp,MAIN,overrides={j:bp[j]})
        zr=response(exp,z)
        for i in possible:
            if zr[i]<=br[i]:
                hits.append(dict(key=k,request=i,request_id=exp.requests[i]['id'],decision=j,
                                 boundary=[ap[j],bp[j]],original_loss_ms=float(ar[i]-br[i]),
                                 forced_loss_ms=float(zr[i]-br[i]),
                                 mean_cost_ms=float(z['response_mean_ms']-a['response_mean_ms'])))
                seen.add(group)
                break
    return dict(interventions_tested=tests,witnesses=hits)


def audit_routes(data, cfg):
    full=[]
    filename=inspect.getfile(Experiment)
    for task,row in zip(cases(cfg,'holdout'),data,strict=True):
        key,trace,model,scale,_=task
        assert key==row['key']
        exp=experiment(trace,key,model,scale)
        routes=Counter()

        def trace_call(frame,event,_):
            if (event!='call' or frame.f_code.co_name!='choose' or frame.f_code.co_filename != filename
                    or frame.f_locals.get('policy')!='snapshot-screened'):
                return None
            frame.f_trace_lines=False
            before=frame.f_locals['decision_replays']

            def trace_return(frame,event,_):
                if event!='return': return trace_return
                v=frame.f_locals
                hits=v['decision_replays']-before
                if len(v['choices'])==1: label='single-choice'
                elif 'fallback' not in v: label='isolated-multi'
                elif v['approximate']==v['fallback'] and not v['unsettled']: label='coarse-fallback'
                elif hits: label='replay'
                else: label='entered-search-no-rollout'
                routes[label]+=1
                routes['rollouts']+=hits
                return None
            return trace_return

        original=sys.gettrace()
        sys.settrace(trace_call)
        try: z=run(exp,MAIN)
        finally: sys.settrace(original)
        expected=row['routes']
        assert sequence(z)==row['zero'][MAIN]['sequence'],key
        assert routes['rollouts']==expected['rollouts'] and routes['single-choice']==expected['single']
        assert routes['isolated-multi']==expected['bypassed_multi']
        assert sum(routes[c] for c in ('coarse-fallback','replay','entered-search-no-rollout'))==expected['screened']
        full.append(dict(key=key,routes=dict(routes)))
    total=Counter()
    for r in full: total.update(r['routes'])
    return dict(table=full,totals=dict(total))


def blobs(prefix):
    out={}
    for n in NAMES:
        ext='json' if n=='preflight' else 'jsonl'
        path=prefix.with_name(prefix.name+'-'+n+'.'+ext)
        out[n]=(path.read_bytes(),read(path.with_suffix('.manifest.json')))
    return out


def archive(src):
    met={}
    data={}
    for n,(raw,manifest) in blobs(src).items():
        ext='json' if n=='preflight' else 'jsonl.gz'
        path=EVIDENCE/f'r07-extension-{n}.{ext}'
        data[n]=raw
        path.write_bytes(raw if n=='preflight' else gzip.compress(raw,mtime=0))
        met[n]=dict(manifest,archive=path.name,archive_sha256=file_hash(path))
    EVIDENCE.joinpath('r07-extension-provenance.json').write_text(json.dumps(met,indent=2)+'\n',encoding='utf-8')


def verify(*, run_tail=False, replay=False, routes=False):
    met=read(EVIDENCE/'r07-extension-provenance.json')
    raw={}
    for n,m in met.items():
        p=EVIDENCE/m['archive']
        assert file_hash(p)==m['archive_sha256']
        raw[n]=p.read_bytes() if n=='preflight' else gzip.decompress(p.read_bytes())
        assert sha256(raw[n]).hexdigest()==m['output_sha256']
        for j,d in m['files'].items(): assert source_matches(ROOT/j,d,historical=True),j
    assert met['timed']['input_sha256']==sha256(raw['holdout']).hexdigest()
    cfg=read(CONFIG)
    preflight=json.loads(raw['preflight']); design=rows(raw['design']); holdout=rows(raw['holdout']); timed=rows(raw['timed'])
    validate(design,holdout,timed,preflight,cfg)
    if replay:
        for split, evidence in [('design',design),('holdout',holdout)]:
            for task,r in zip(cases(cfg,split),evidence,strict=True):
                assert json.loads(json.dumps(evaluate(task)))==r, r['key']
    for h,t in zip(holdout,timed,strict=True): h['timed']=t['timed']
    result=check_summary(holdout,cfg)
    result['hand_checked_cold_request']=manual_contract(cfg)
    if run_tail: result['tail_causal']=tail_witnesses(holdout,cfg)
    if routes: result['route_audit']=audit_routes(holdout,cfg)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-prefix',type=Path)
    p.add_argument('--verify-tail',action='store_true')
    p.add_argument('--verify-routes',action='store_true')
    p.add_argument('--replay',action='store_true',help='rerun all zero-overhead cases, not timing')
    a=p.parse_args()
    target=EVIDENCE/'r07-extension-summary.json'
    if a.archive_prefix:
        archive(a.archive_prefix)
        result=verify(run_tail=True,routes=True)
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        result=verify(run_tail=a.verify_tail,replay=a.replay,routes=a.verify_routes)
        expected=read(target)
        if not a.verify_tail: expected.pop('tail_causal')
        if not a.verify_routes:
            audited=expected.pop('route_audit')
            assert len(audited['table'])==sum(r['n'] for r in result['table'] if r['axis']=='all-traces')
            recovered=Counter()
            for row in audited['table']: recovered.update(row['routes'])
            assert dict(recovered)==audited['totals']
        assert result==expected
    print('R07 extension evidence verified:',len(result['table']),'strata')


if __name__=='__main__':
    main()
