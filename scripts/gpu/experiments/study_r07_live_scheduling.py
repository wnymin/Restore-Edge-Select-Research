"""Controlled paired probes against a local llama-server with slot save/restore enabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
from threading import Event
from time import perf_counter_ns, sleep
from urllib.request import Request, build_opener, ProxyHandler


def send(url, path, payload=None):
    body=None if payload is None else json.dumps(payload,separators=(',',':')).encode()
    return build_opener(ProxyHandler({})).open(Request(url+path,body,{'Content-Type':'application/json'}),timeout=180)


def query(url,path,payload=None):
    with send(url,path,payload) as response:return json.load(response)


def snapshot(url):
    start=perf_counter_ns()
    raw=query(url,'/slots')
    finish=perf_counter_ns()
    # Explicit allowlist of runtime state; no recorded prompt content or post-hoc features.
    keys=('id','id_task','is_processing','n_ctx','n_past','n_decoded','n_remaining','prompt_tokens','next_token')
    return dict(start=start,end=finish,slots=[{k:v for k,v in s.items() if k in keys} for s in raw],
                raw_keys=[sorted(s) for s in raw])


def stream(url, slot, tokens, n, first=None, submitted=None):
    start=perf_counter_ns()
    if submitted:submitted.set()
    stamps=[]; chunks=[]; final=None; final_received=None; terminal_events=[]
    payload=dict(id_slot=slot,prompt=tokens,n_predict=n,stream=True,cache_prompt=True,
                 temperature=0,ignore_eos=True,return_tokens=True)
    with send(url,'/completion',payload) as r:
        for line in r:
            if not line.startswith(b'data:'):continue
            content=line[5:].strip()
            if content==b'[DONE]':terminal_events.append('done');break
            j=json.loads(content); received=perf_counter_ns()
            if 'error' in j:
                error=j['error']; terminal_events.append({k:str(v)[:320] for k,v in error.items()} if isinstance(error,dict) else str(error)[:320])
            if j.get('stop') or 'timings' in j:
                final=j; final_received=received
            if j.get('tokens') or j.get('content'):
                stamps.append(received); chunks.append(j.get('tokens') or j.get('content'))
                if first:first.set()
    if final is None or not stamps:raise RuntimeError(f'incomplete SSE slot {slot}, count {len(stamps)}, terminal events {terminal_events}')
    timing=final['timings']
    if timing['predicted_n']!=n:raise RuntimeError(f'wrong generated length {timing}')
    return dict(start=start,end=final_received,tokens_ns=stamps,timings=timing,
                length=timing['predicted_n'],chunk_hash=sha256(json.dumps(chunks).encode()).hexdigest(),
                first_token=chunks[0])


def slot_api(url,i,action,name=None):
    return query(url,f'/slots/{i}?action={action}',{} if name is None else {'filename':name})


def trial(url,mode,depth,target,peer,later):
    for i in (0,1):slot_api(url,i,'erase')
    launch,first=Event(),Event();follower=None;s0=None;sent_at=None
    with ThreadPoolExecutor(3) as pool:
        if mode!='idle':
            f_peer=pool.submit(stream,url,0,peer,64,first)
            if not first.wait(45):raise RuntimeError('peer first token not received')
        s0=snapshot(url)
        start=perf_counter_ns()
        restored=slot_api(url,1,'restore',f'r07-live-{depth}.bin')
        ready=perf_counter_ns()
        assert restored['n_restored']==depth and restored['n_read']>0,restored
        verify=snapshot(url)
        f_target=pool.submit(stream,url,1,target,2,None,launch)
        s1=None
        if mode=='queue':
            assert launch.wait(10)
            until=perf_counter_ns()+200_000_000
            while True:
                s1=snapshot(url)
                target_slot=next(s for s in s1['slots'] if s['id']==1)
                if target_slot['is_processing'] or perf_counter_ns()>until:break
            sent_at=perf_counter_ns()
            follower=pool.submit(stream,url,1,later,2)
        a=f_target.result(); p=None if mode=='idle' else f_peer.result(); f=None if follower is None else follower.result()
    assert a['timings'].get('cache_n')==depth and a['timings']['prompt_n']==len(target)-depth,a['timings']
    if p:assert p['timings'].get('cache_n',0)==0,p['timings']
    if f:assert f['timings'].get('cache_n',0)==0,f['timings']
    return dict(mode=mode,depth=depth,pre=s0,post_restore=verify,pre_follower=s1,restore_start=start,
                restore_end=ready,restore=restored,target=a,peer=p,follower=f,follower_dispatch=sent_at)


def run(a):
    assert a.repeats>0
    order=random.Random(a.seed ^ 0xA495); rng=random.Random(a.seed)
    props=query(a.url,'/props'); init=snapshot(a.url)
    assert len(init['slots'])==2,init
    with a.out.open('w',encoding='utf-8',newline='\n') as out:
        def emit(row):out.write(json.dumps(row,separators=(',',':'))+'\n');out.flush()
        emit(dict(meta=True,seed=a.seed,condition=a.condition,props=props,initial_slots=init))
        for block in range(-a.warmup,a.repeats):
            def prompt(n):return [rng.randrange(1000,15000) for _ in range(n)]
            t,p,f=prompt(a.target_tokens),prompt(96),prompt(128)
            hashes=[sha256(json.dumps(x).encode()).hexdigest() for x in (t,p,f)]
            for depth in (512,1024):
                slot_api(a.url,1,'erase')
                query(a.url,'/completion',dict(id_slot=1,prompt=t[:depth],n_predict=1,cache_prompt=False,temperature=0))
                s=slot_api(a.url,1,'save',f'r07-live-{depth}.bin')
                assert s['n_saved']==depth and s['n_written']>0,s
            modes=['idle','active','queue'];order.shuffle(modes)
            for mode in modes:
                depths=[512,1024];order.shuffle(depths)
                for depth in depths:
                    try:
                        result=trial(a.url,mode,depth,t,p,f)
                    except (AssertionError, OSError, RuntimeError, KeyError, ValueError) as exc:
                        if not a.retain_failures:raise
                        emit(dict(block=block,prompt_hashes=hashes,mode=mode,depth=depth,failure=type(exc).__name__+': '+str(exc)))
                        print(a.condition,block,mode,depth,'FAILED',exc,flush=True)
                    else:
                        emit(dict(block=block,prompt_hashes=hashes,**result))
                        print(a.condition,block,mode,depth,'first ms',round((result['target']['tokens_ns'][0]-result['restore_start'])/1e6,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18297');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--repeats',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--warmup',type=int,default=0);p.add_argument('--target-tokens',type=int,default=1536)
    p.add_argument('--retain-failures',action='store_true')
    run(p.parse_args())
