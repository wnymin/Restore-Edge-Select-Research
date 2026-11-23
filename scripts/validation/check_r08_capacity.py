"""Recompute R08 summaries, validate frozen coverage, and package raw evidence."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
from statistics import mean, median

from scripts.cpu.replay.study_r08_capacity import CONFIG, MIB, POLICIES, R07_CONFIG, ROOT, budgets, models, read
from scripts.cpu.replay.study_r08_native import CFG as NATIVE, make_costs, model_cases
from scripts.cpu.replay.study_r08_replacement import AMENDMENT, corrected_config

from scripts.gpu.bench.measure_r08_paths import SIZES

E=ROOT/'docs/evidence'
MAIN='entry-replay'
PARTS=('micro','design','holdout','timed','replacement','replacement-timed',
       'native-prefill','native-io','native','native-timed')
WAVE=('native-prefill','native-io')


def arrange(raw):
    return [json.loads(line) for line in raw.splitlines()]


def opt(row):
    m,b=map(F,(row['main_exact'],row['base_exact'])); l,h=map(F,(row['oracle']['lower'],row['oracle']['upper_bound']))
    return m,b,l,h


def ident(k): return tuple(sorted(k.items()))


def source_matches(path, expected):
    # The Windows shell wrote a CRLF terminator on some initial source files;
    # .gitattributes normalizes checked-out sources to LF. Keep measured hashes.
    actual=path.read_bytes(); lf=actual.replace(b'\r\n',b'\n')
    variants={actual,lf,lf.replace(b'\n',b'\r\n')}
    if lf.endswith(b'\n'): variants.add(lf[:-1]+b'\r\n')
    return any(sha256(v).hexdigest()==expected for v in variants)


def check_row(r,cfg):
    m,b,l,h=opt(r)
    assert l<=h<=min(m,b) and r['oracle']['exact']==(l==h)
    if r['oracle']['nodes'] is not None:assert r['oracle']['nodes']<=cfg['node_budget']
    assert len(r['plan'])==len(r['oracle']['plan'])==r['n']
    assert r['zero']['entry']['sequence']==r['zero'][MAIN]['sequence']
    assert 0<=r['feasible_by_size']['cold']<=r['feasible_by_size']['total']
    assert max(v['peak'] for v in r['activity'].values())<=r['key']['capacity_bytes']


def timing(r,cfg):
    a=r['timed']
    assert len(a)==cfg['timing_repeats']
    assert {tuple(t['order']) for t in a}==set(permutations(POLICIES))
    for trial in a:
        assert set(trial['order'])==set(trial['policies'])==set(POLICIES)
        for t in trial['policies'].values():
            assert len(t['late_indices'])==t['losses_gt10'] and t['decision']>0
    return a


def memberships(models,src,config,native=False):
    pairs=[]
    for t,ws in src.items():
        for w in ws:
            for m,(model,_) in models.items():
                def key(r,s,l,b):return dict(trace=t,window=w,model=m,rate=r,section=s,level=l,capacity_bytes=b)
                for r,(s,l,b) in product(config['rates'],budgets(config,model)):
                    k=key(r,s,l,b)
                    pairs.append(dict(k,cost='measured-both') if native else k)
                if native:
                    for c in ('measured-compute','measured-io','archived-both'):
                        pairs.append(dict(key(16,'fixed','128',128*MIB),cost=c))
                else:
                    for b in config['negative_control_mib']:pairs.append(key(1,'control',str(b),b*MIB))
    return pairs


def verify_cases(cfg,ncfg,micro,design,holdout,timed,native,ntimed):
    mod={m:v for m,v in models(read(R07_CONFIG)).items() if m in cfg['main_models']}
    design_windows={(t,w) for t,ws in cfg['design'].items() for w in ws}
    main_windows={(t,w) for t,ws in cfg['holdout'].items() for w in ws}
    native_windows={(t,w) for t,ws in ncfg['heldout'].items() for w in ws}
    assert len(design_windows | main_windows | native_windows)==len(design_windows)+len(main_windows)+len(native_windows)
    expected_micro=set(product(mod,('same-time','stagger','shared-kv','late'),('one','union'),cfg['micro_offsets_bytes']))
    actual_micro=[tuple(r['key'][k] for k in ('model','fixture','target','offset_bytes')) for r in micro]
    assert len(actual_micro)==len(expected_micro) and set(actual_micro)==expected_micro
    for r in micro:
        check_row(r,cfg)
        k,w=r['key'],r['fixed_witness']; cap=k['capacity_bytes']
        q=mod[k['model']][0]['payload']
        assert w['one_bytes']==576*q['attention_bytes_per_token']+q['checkpoint_bytes']
        assert w['union_bytes']==2*w['one_bytes']-(576*q['attention_bytes_per_token'] if k['fixture']=='shared-kv' else 0)
        assert cap==w['threshold_bytes']+k['offset_bytes']
        assert w['threshold_bytes']==w['one_bytes' if k['target']=='one' else 'union_bytes']
        assert w['first_at_arrival']==(cap>=w['one_bytes'])
        assert w['second_at_arrival']==(k['fixture']!='late' and cap>=w['union_bytes'])
        assert w['peak']<=cap and r['oracle']['exact']
    native_input=dict(ncfg,reference_tokens=512,matched_padding_mib=1)
    for records,expected,config in (
        (design,memberships(mod,cfg['design'],cfg),cfg),
        (holdout,memberships(mod,cfg['holdout'],cfg),cfg),
        (native,memberships(model_cases(),ncfg['heldout'],native_input,True),ncfg)):
        assert [r['key'] for r in records]==expected and len(set(map(ident,expected)))==len(expected)
        inputs={}
        size_tests=defaultdict(list)
        for r in records:
            k=r['key']; check_row(r,config)
            x=(k['trace'],k['window']); tag=r['input']['order']
            assert inputs.setdefault(x,tag)==tag
            size_tests[x,k['model']].append((k['capacity_bytes'],r['feasible_by_size']['cold'],r['feasible_by_size']['total']))
        for v in size_tests.values():
            ranked=sorted(v)
            assert len({t for _cap,_n,t in ranked})==1
            assert [n for _,n,_ in ranked]==sorted(n for _,n,_ in ranked)
    for rows, clock,config in [(holdout,timed,cfg),(native,ntimed,ncfg)]:
        for r,t in zip(rows,clock,strict=True):
            assert r['key']==t['key'];r['timed']=timing(t,config)
            assert all(0<=ix<r['n'] for tr in r['timed'] for v in tr['policies'].values() for ix in v['late_indices'])



def amend_rows(data, cfg):
    patch=read(AMENDMENT)
    trace,old,new=patch['trace'],patch['exclude_window'],patch['replacement_window']
    counts=Counter()
    for label,supplement in [('holdout','replacement'),('timed','replacement-timed')]:
        by_key={ident(row['key']):row for row in data[supplement]}
        assert len(by_key)==len(data[supplement])
        replaced=[]
        for row in data[label]:
            k=row['key']
            if (k['trace'],k['window'])==(trace,old):
                row=by_key.pop(ident(dict(k,window=new)))
                counts[label]+=1
            replaced.append(row)
        assert counts[label]>0 and not by_key
        data[label]=replaced
    assert counts['holdout']==counts['timed']
    return corrected_config(cfg,patch),dict(**patch,replaced_cases=counts['holdout'])


def pair_summary(rows,cfg):
    groups=defaultdict(list); ct=Counter(); diags=Counter()
    for r in rows:
        k=r['key'];m,b,l,h=opt(r)
        ct['cases']+=1;ct['exact']+=l==h;ct['picked_optimum']+=m==l
        ct['no_room']+=b==l; ct['room_certified']+=b>h;ct['room_unknown']+=l<b==h
        groups[k['model'],k['rate'],k['section'],k['level'],k.get('cost','r07-proxy')].append(r)
        if d:=r.get('diagnostic'):
            if 'current' not in d:diags['isolated']+=1;continue
            cur=d['current']; f=d['closed_loop_forced_mean_ms']; valuable=lambda name:f[str(cur)]-f[str(d[name])]>1e-7
            if valuable('window4_choice'):diags['coarse']+=1
            elif valuable('all_seen_choice'):diags['visibility']+=1
            elif valuable('joint_choice'):diags['joint']+=1
            else:diags['unclassified']+=1
    table=[]
    for (model,rate,section,level,cost),ds in groups.items():
        cap,={r['key']['capacity_bytes'] for r in ds}
        polys=list(map(opt,ds)); m,b,l,h=map(mean,zip(*polys)); gap=(m-h,m-l); head=(b-h,b-l)
        if head[0]<=0: coverage=None
        else: coverage=sorted(float(100*(b-m)/v) for v in head)
        critical=[]; tail=[]; tmean=[];tp95=[];dec=[];dentry=[];max_t=0
        for r,(x,y,p,q) in zip(ds,polys,strict=True):
            if x-q>cfg['gap_ms'] and y-p>0 and (x-q)/(y-p)>F(str(cfg['gap_fraction'])):
                critical.append(dict(key=r['key'],gap_min=float(x-q),missed_min=float((x-q)/(y-p))*100))
            sim=r['timed']
            rep=Counter(i for v in sim for i in v['policies'][MAIN]['late_indices'])
            if hit:=[i for i,n in rep.items() if n>=cfg['request_loss_repeats']]:tail.append(dict(key=r['key'],requests=sorted(hit)))
            def diff(metric,baseline='local-cost'):
                return median(t['policies'][MAIN][metric]-t['policies'][baseline][metric] for t in sim)
            tmean.append(diff('mean'));tp95.append(diff('p95'));dentry.append(diff('decision','entry'))
            dec.append(median(t['policies'][MAIN]['decision'] for t in sim))
            max_t=max(max_t,*(t['policies'][MAIN]['max_loss'] for t in sim))
        status=('no-room' if head[1]==0 else 'material-gap' if len(critical)>=cfg['replication_windows']
                else 'robust-mean' if head[0]>0 and (gap[1]<=cfg['gap_ms'] or gap[1]/head[0]<=F(str(cfg['gap_fraction'])))
                else 'unresolved')
        table.append(dict(model=model,rate=rate,section=section,level=level,cost=cost,capacity_bytes=cap,n=len(ds),
                   exact=sum(p==q for x,y,p,q in polys),optimum=sum(x==p for x,y,p,q in polys),
                   no_room=sum(y==p for x,y,p,q in polys),space_cert=sum(y>q for x,y,p,q in polys),
                   gap=list(map(float,gap)),headroom=list(map(float,head)),capture=coverage,status=status,
                   critical=critical,persistent=tail,zero_tail=sum(r['zero'][MAIN]['losses_gt10']>0 for r in ds),
                   max_request_loss_ms=max_t,timed_mean_ms=mean(tmean),timed_p95_ms=mean(tp95),
                   decision_ms=mean(dec),decision_vs_entry_ms=mean(dentry),
                   cold_possible=mean(r['feasible_by_size']['cold'] for r in ds),
                   possible=mean(r['feasible_by_size']['total'] for r in ds),
                   observed_reserved=mean(r['activity'][MAIN]['reserved'] for r in ds)))
    return dict(totals=ct,groups=table,diagnostic_counts=diags)


def archive(paths):
    met={}
    for key,p in paths.items():
        val=p.read_bytes(); meta=None if key in WAVE else read(p.with_suffix('.manifest.json'))
        target=E/f'r08-{key}.{"json.gz" if key in WAVE else "jsonl.gz"}'
        target.write_bytes(gzip.compress(val,mtime=0))
        met[key]=dict(file=target.name,archive_sha256=sha256(target.read_bytes()).hexdigest(),
                      content_sha256=sha256(val).hexdigest(),manifest=meta)
    met['tools']={str(p.relative_to(ROOT).as_posix()):sha256(p.read_bytes()).hexdigest() for p in
                  (ROOT/'scripts/gpu/bench/measure_r08_prefill.py',ROOT/'scripts/gpu/bench/measure_r08_paths.py',
                   ROOT/'scripts/data/make_r08_calibration_gguf.py')}
    (E/'r08-provenance.json').write_bytes((json.dumps(met,indent=2)+'\n').encode())


def read_and_check():
    met=read(E/'r08-provenance.json');data={}
    for k in PARTS:
        x=met[k]; p=E/x['file']; b=p.read_bytes(); assert sha256(b).hexdigest()==x['archive_sha256']
        content=gzip.decompress(b)
        assert sha256(content).hexdigest()==x['content_sha256']
        data[k]=json.loads(content) if k in WAVE else arrange(content)
        if m:=x['manifest']:
            assert m['output_sha256']==x['content_sha256']
            for n,h in m.get('files',m.get('sources',{})).items():assert source_matches(ROOT/n,h),n
    assert met['timed']['manifest']['input_sha256']==met['holdout']['content_sha256']
    assert met['replacement-timed']['manifest']['input_sha256']==met['replacement']['content_sha256']
    assert met['native-timed']['manifest']['timed_from_sha256']==met['native']['content_sha256']
    for k in ('native','native-timed'):
        for ex in ('prefill','io'):assert met[k]['manifest']['input_sha256'][ex]==met['native-'+ex]['content_sha256']
    for n,h in met['tools'].items():assert source_matches(ROOT/n,h),n
    cfg,ncfg=read(CONFIG),read(NATIVE)
    prefill,io=data['native-prefill'],data['native-io']
    assert source_matches(ROOT/'scripts/gpu/bench/measure_r08_prefill.py',prefill['provenance']['tool_sha256'])
    assert source_matches(ROOT/'scripts/gpu/bench/measure_r08_paths.py',io['tool_sha256'])
    assert prefill['provenance']['local_model_sha256']==io['data_sha256']
    expected_io=set(product(('ram','ssd','copy'),SIZES,range(ncfg['io_repeats'])))
    assert io['rounds']==ncfg['io_repeats'] and len(io['samples'])==len(expected_io)
    assert {(r['kind'],r['bytes'],r['trial']) for r in io['samples']}==expected_io
    assert all(r['ns']>0 and (r['offset'] is not None)==(r['kind']=='ssd') for r in io['samples'])
    assert set(prefill['points'])==set(model_cases())
    for label,series in prefill['points'].items():
        assert set(series)=={'grid','observe'}
        for split,point in series.items():
            assert len(point)==len({(r['n_depth'],r['n_prompt']) for r in point})
            assert all(r['n_gen']==0 and r['n_gpu_layers']==99 and r['devices']=='Vulkan1'
                       and len(r['samples_ns'])==ncfg['prefill_repeats'] and min(r['samples_ns'])>0 for r in point)
    cfg,amendment=amend_rows(data,cfg)
    verify_cases(cfg,ncfg,*(data[k] for k in ('micro','design','holdout','timed','native','native-timed')))
    costs=make_costs(ncfg,prefill,io)
    return dict(holdout_amendment=amendment,main=pair_summary(data['holdout'],cfg),native=pair_summary(data['native'],cfg),
                calibration=dict(loss=costs['prefill_observe'],paths=costs['io_residuals'],fit=costs['io']),
                design=dict(cases=len(data['design']),exact=sum(r['oracle']['exact'] for r in data['design'])),
                micro=dict(cases=len(data['micro']),all_exact=all(r['oracle']['exact'] for r in data['micro']),
                           first_and_second_admission_changes=[dict(key=r['key'],witness=r['fixed_witness']) for r in data['micro']]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');a=p.parse_args()
    if a.archive:
        base=ROOT/'outputs'
        archive({k:base/(f'r08-pre-{k}.jsonl' if k in ('micro','design','holdout','timed') else
                        f'r08-{k}.json' if k in WAVE else f'r08-{k}.jsonl') for k in PARTS})
    result=read_and_check()
    target=E/'r08-summary.json'
    if a.archive:target.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    else:assert result==read(target)
    print('R08 verified:',result['micro']['cases'],'micro;',result['main']['totals'],'main;',result['native']['totals'],'native')


if __name__=='__main__':main()
