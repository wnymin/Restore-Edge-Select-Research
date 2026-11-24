"""Validate and summarize archived R07 real-server forced-boundary pairs."""
import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from scripts.gpu.experiments.study_r07_live_observer import ARMS as OBS_ARMS, REPEATS as OBS_N, SEED as OBS_SEED

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'; E=ROOT/'docs/evidence'
ARMS={'4b-cb512':(77651,1536),'4b-static512':(77651,1536),
      '4b-cb256-holdout':(138244,1792),'16l-cb256-holdout':(464277,1792)}
MODES=('idle','active','queue'); DEPTHS=(512,1024); REPEATS=8
MS=10**6


def latency(r):
    a=r['target']; p=r['peer']; f=r['follower']
    first=a['tokens_ns'][0]
    m=dict(target=(first-r['restore_start'])/MS,restore=(r['restore_end']-r['restore_start'])/MS,
           submit_to_first=(first-a['start'])/MS,prompt=a['timings']['prompt_ms'],
           restore_service=r['restore']['timings']['restore_ms'])
    if p:
        times=p['tokens_ns']; spans=list(zip(times,times[1:]))
        m.update(peer_max=max(b-a for a,b in spans)/MS,peer_itl=median(b-a for a,b in spans)/MS,
                 peer_stream=(times[-1]-times[0])/MS)
        for label,start,end in [('restore_window',r['restore_start'],r['restore_end']),
                                ('target_window',a['start'],first)]:
            overlap=[(b-x)/MS for x,b in spans if x<end and b>start]
            if overlap:m['peer_max_'+label]=max(overlap)
    if f:m['follower']=(f['tokens_ns'][0]-f['start'])/MS
    return m


def processing(snapshot,id):
    return next(r for r in snapshot['slots'] if r['id']==id)['is_processing']


def check(arm,raw):
    meta,*rows=map(json.loads,raw.splitlines()); seed,n=ARMS[arm]
    assert meta['meta'] and meta['condition']==arm and meta['seed']==seed and meta['props']['total_slots']==2
    assert '10923' in meta['props']['build_info'] and '8a56aedd6' in meta['props']['build_info'],meta['props']['build_info']
    assert len(rows)==(REPEATS+1)*len(MODES)*len(DEPTHS)
    pairs=defaultdict(dict); prompt_hashes={}; all_keys=set(); failures=[]
    for r in rows:
        k=(r['block'],r['mode'],r['depth']); assert k not in all_keys;all_keys.add(k)
        block,mode,depth=k
        assert len(r['prompt_hashes'])==3 and prompt_hashes.setdefault(block,r['prompt_hashes'])==r['prompt_hashes']
        if 'failure' in r:
            assert arm=='16l-cb256-holdout' and r['failure']
            failures.append(r);continue
        t,p,f=(r[x] for x in ('target','peer','follower'))
        assert t['timings']['cache_n']==r['restore']['n_restored']==depth and t['timings']['prompt_n']==n-depth
        assert r['restore']['n_read']>0
        assert (p is None)==(mode=='idle') and (f is None)==(mode!='queue')
        assert r['pre']['end']<=r['restore_start']<r['restore_end']<=r['post_restore']['start']<=r['post_restore']['end']<=t['start']
        for response,count in [(t,2),(p,64),(f,2)]:
            if response is None:continue
            ts=response['tokens_ns']
            assert response['start']<ts[0]<=ts[-1]<=response['end'] and ts==sorted(ts)
            assert count==len(ts)==response['length'],(k,count,len(ts))
        if p:assert p['tokens_ns'][0]<=r['pre']['start'] and p['timings']['prompt_n']==96
        if f:assert t['start']<=r['pre_follower']['start'] and r['pre_follower']['end']<=f['start'] and f['timings']['prompt_n']==128
        if block>=0:pairs[block,mode][depth]=r
    assert all_keys=={(b,m,d) for b in range(-1,REPEATS) for m in MODES for d in DEPTHS}
    return meta,pairs,prompt_hashes,failures


def summarize(pairs):
    out={}; anomalies=[]
    for mode in MODES:
        seen=defaultdict(list); blocks=[]
        for k,p in sorted(pairs.items()):
            if k[1]!=mode or tuple(sorted(p))!=DEPTHS:continue
            block=k[0];s,d=(p[x] for x in DEPTHS); ls,ld=latency(s),latency(d)
            delta={x:ld[x]-ls[x] for x in ls.keys() & ld.keys()}
            target_match=s['target']['first_token']==d['target']['first_token']
            peer_match=s['peer'] is None or s['peer']['chunk_hash']==d['peer']['chunk_hash']
            followers_match=s['follower'] is None or s['follower']['chunk_hash']==d['follower']['chunk_hash']
            def exposure(r):
                return dict(peer_at_restore=processing(r['pre'],0),peer_at_submit=processing(r['post_restore'],0),
                            target_at_follower=(processing(r['pre_follower'],1) and r['follower']['start']<r['target']['end']) if r['follower'] else None)
            flagged=sorted(x for x,v in delta.items() if (x=='target' and v>10) or (x in ('follower','peer_max','peer_stream','peer_max_restore_window') and delta['target']<-10 and v>10))
            if not target_match or not peer_match or not followers_match:flagged.append('greedy_output_difference')
            if flagged:anomalies.append(dict(block=block,mode=mode,flags=flagged,delta=delta))
            for x,y in delta.items():seen[x].append(y)
            blocks.append(dict(block=block,delta=delta,exposure=[exposure(s),exposure(d)],target_match=target_match,
                               peer_match=peer_match,followers_match=followers_match,
                               slot_probe_ms=[[(x['end']-x['start'])/MS for x in (a['pre'],a['post_restore'],a['pre_follower']) if x] for a in (s,d)]))
        stats={}
        for x,vals in sorted(seen.items()):
            n=len(vals); ordered=sorted(vals); neg=sum(v<0 for v in vals)
            # Exact order-statistic interval for median at n=8; nominal 92.97% under iid continuous blocks.
            interval=[ordered[1],ordered[-2]] if n==8 else None
            stats[x]=dict(n=n,median=median(vals),lo=min(vals),hi=max(vals),negative=neg,over_10=sum(v>10 for v in vals),median_interval_93=interval)
        out[mode]=dict(summary=stats,blocks=blocks)
    return dict(modes=out,anomalies=anomalies)


def visibility(pairs):
    observations=[]
    for row in pairs.values():
        for r in row.values():
            before=r['pre'];after=r['post_restore']
            slot=lambda cap:next(s for s in cap['slots'] if s['id']==1)
            if slot(before)==slot(after) and before['raw_keys'][1]==after['raw_keys'][1]:
                observations.append(dict(block=r['block'],mode=r['mode'],depth=r['depth']))
    return dict(slot_1_observational_alias=observations,
                snapshot_ms=median((x['end']-x['start'])/MS for p in pairs.values() for r in p.values()
                                   for x in (r['pre'],r['post_restore'],r['pre_follower']) if x))


def analyze_observer(raw):
    meta,*rows=map(json.loads,raw.splitlines()); assert meta['seed']==OBS_SEED and meta['props']['total_slots']==2
    assert '10923' in meta['props']['build_info'] and '8a56aedd6' in meta['props']['build_info']
    keys={(r['block'],r['mode'],r['depth']):r for r in rows}
    assert len(keys)==len(rows)==(OBS_N+1)*len(OBS_ARMS)
    assert set(keys)=={(b,m,d) for b in range(-1,OBS_N) for m,d in OBS_ARMS}
    result=[]
    for b in range(-1,OBS_N):
        pairs={(mode,d):keys[b,mode,d] for mode,d in OBS_ARMS};control=pairs['peer',0]
        assert len({row['prompt_hash'] for row in pairs.values()})==1
        by_arm={}
        for (mode,d),r in pairs.items():
            p=r['peer']; ts=p['tokens_ns']; assert 2<=len(ts)<=p['length']==64 and ts[0]<=r['trigger']<ts[-1]
            assert (r['restore'] is None)==(mode=='peer') and len(r['observe'])==(2 if mode=='probe' else 0)
            assert (r['target'] is not None)==(mode in ('clean','probe'))
            vals=dict(peer_max=max(y-x for x,y in zip(ts,ts[1:]))/MS,peer_stream=(ts[-1]-ts[0])/MS,
                      same_peer_output=p['chunk_hash']==control['peer']['chunk_hash'],peer_chunks=len(ts))
            if r['restore']:
                assert r['restore']['n_restored']==d and r['trigger']<=r['restore_start']<r['restore_end'] and r['restore_start']<ts[-1]
                max_interval=[(y-x)/MS for x,y in zip(ts,ts[1:]) if x<r['restore_end'] and y>r['restore_start']]
                assert max_interval
                vals.update(restore=(r['restore_end']-r['restore_start'])/MS,overlap=max(max_interval))
            if t:=r['target']:
                assert t['timings']['cache_n']==d and t['timings']['prompt_n']==1536-d and t['start']>=r['restore_end']
                vals['target']=(t['tokens_ns'][0]-r['trigger'])/MS
            if mode=='probe':
                b0,b1=r['observe']
                slot=lambda x:next(s for s in x['slots'] if s['id']==1)
                vals['target_alias']=slot(b0)==slot(b1) and b0['raw_keys'][1]==b1['raw_keys'][1]
                vals['live_prompt_tokens']=[slot(b0)['n_prompt_tokens'],slot(b1)['n_prompt_tokens']]
                assert vals['live_prompt_tokens']==[0,d]
                vals['poll_ms']=sum((o['end']-o['start'])/MS for o in r['observe'])
            by_arm[f'{mode}-{d}']=vals
        if b>=0:result.append(dict(block=b,arms=by_arm))
    deltas={}
    def paired(label,u,v,metrics):
        deltas[label]={}
        for name in metrics:
            a=[r['arms'][u][name]-r['arms'][v][name] for r in result]
            complete=[x for x,r in zip(a,result) if min(r['arms'][u]['peer_chunks'],r['arms'][v]['peer_chunks'])==64]
            deltas[label][name]=dict(median=median(a),over_10=sum(x>10 for x in a),range=[min(a),max(a)],
                                     all_chunks_median=median(complete) if complete else None)
    for d in (512,1024):
        paired('restore_vs_peer_'+str(d),f'restore-{d}','peer-0',('peer_max','peer_stream'))
        paired('clean_vs_restore_'+str(d),f'clean-{d}',f'restore-{d}',('peer_max','peer_stream'))
        paired('probe_vs_clean_'+str(d),f'probe-{d}',f'clean-{d}',('peer_max','peer_stream','target'))
    paired('deep_vs_shallow_restore','restore-1024','restore-512',('peer_max','overlap','peer_stream','restore'))
    paired('deep_vs_shallow_clean','clean-1024','clean-512',('peer_max','overlap','peer_stream','restore','target'))
    return dict(blocks=result,deltas=deltas)


def get_raw(name,archive,prov):
    gz=E/f'r07-live-{name}.jsonl.gz'
    if archive:
        raw=(OUT/f'r07-live-{name}.jsonl').read_bytes();gz.write_bytes(gzip.compress(raw,mtime=0))
    raw=gzip.decompress(gz.read_bytes())
    prov[name]=dict(archive_sha256=sha256(gz.read_bytes()).hexdigest(),raw_sha256=sha256(raw).hexdigest())
    return raw


def queue_evidence(archive,prov):
    name='r07-live-queue-interface.json'; dst=E/name
    if archive:dst.write_bytes((OUT/name).read_bytes())
    raw=dst.read_bytes()
    # Git normalizes JSON to LF. The original Windows capture used CRLF.
    captured=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    prov['queue_interface']=dict(raw_sha256=sha256(captured).hexdigest())
    e=json.loads(raw); p=e['probes']; req=e['requests']
    assert [x['expected_pending'] for x in p]==[0,1,2] and [r['length'] for r in req]==[160,160,2,2]
    assert all(processing(q['snapshot'],i) for i in (0,1) for q in p)
    assert req[2]['start']<req[1]['end'] and req[3]['start']<req[1]['end']
    assert [q['metrics']['llamacpp:requests_deferred'] for q in p]==[0,1,2]
    assert [q['metrics']['llamacpp:requests_processing'] for q in p]==[2,2,2]
    assert len({tuple(r['id_task'] for r in q['snapshot']['slots']) for q in p})==1
    return dict(metrics=[q['metrics'] for q in p],slot_task_ids=[[s['id_task'] for s in q['snapshot']['slots']] for q in p],
                queue_exposed_keys=[[r for row in q['snapshot']['raw_keys'] for r in row if 'queue' in r or 'defer' in r] for q in p])


def solo_evidence(archive,prov,hashes,failed_attempts):
    report={}
    for arm in ('4b','16l'):
        raw=get_raw('solo-'+arm,archive,prov); meta,*rows=map(json.loads,raw.splitlines())
        assert meta['arm']==arm and meta['props']['total_slots']==2
        assert [r['block'] for r in rows]==list(range(-1,REPEATS))
        assert all(r['prompt_hash']==hashes[r['block']][1] for r in rows)
        for r in rows:
            assert ('result' in r) != ('error' in r)
            if t:=r.get('result'):assert t['length']==64 and t['timings']['prompt_n']==96 and t['timings']['cache_n']==0
        errors=[dict(block=r['block'],message=r['error']) for r in rows if 'error' in r]
        report[arm]=dict(errors=errors,formal_completed=sum('result' in r and r['block']>=0 for r in rows),
                         failed_with_and_without_restore=sorted({r['block'] for r in failed_attempts if any(e['block']==r['block'] for e in errors)}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');a=p.parse_args()
    arms={}; prompts={}; prov={}; holdout_hashes=None
    for arm in ARMS:
        raw=get_raw(arm,a.archive,prov); _,pairs,hashes,failures=check(arm,raw)
        prompts[arm]=set(tuple(v) for v in hashes.values());arms[arm]=summarize(pairs)
        arms[arm]['visibility']=visibility(pairs)
        arms[arm]['failures']=failures
        if arm.startswith('16l-'):holdout_hashes=hashes
        print(arm,sum(tuple(sorted(p))==DEPTHS for p in pairs.values()),'valid paired trials;',len(failures),'failed attempts including warmup;',len(arms[arm]['anomalies']),'flagged pairs')
    assert prompts['4b-cb512']==prompts['4b-static512'] and len(set.union(*prompts.values()))==3*(REPEATS+1)
    target=E/'r07-live-scheduling-summary.json'; payload=dict(provenance=prov,arms=arms,observer=analyze_observer(get_raw('observer',a.archive,prov)),queue_interface=queue_evidence(a.archive,prov),
        solo=solo_evidence(a.archive,prov,holdout_hashes,arms['16l-cb256-holdout']['failures']))
    if a.archive:target.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    else:assert payload==json.loads(target.read_bytes())


if __name__=='__main__':main()
