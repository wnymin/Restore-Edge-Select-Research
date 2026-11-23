"""Randomized peer-only, restore-only, and observer-effect controls on a real server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
from threading import Event
from time import perf_counter_ns

from scripts.gpu.experiments.study_r07_live_scheduling import query, slot_api, stream, send

ARMS=[('peer',0)]+[(m,d) for m in ('restore','clean','probe') for d in (512,1024)]
SEED=774115; REPEATS=8


def snapshot(url):
    start=perf_counter_ns(); rows=query(url,'/slots'); finish=perf_counter_ns()
    keys=('id','id_task','is_processing','n_ctx','n_prompt_tokens','n_prompt_tokens_cache',
          'n_prompt_tokens_processed','next_token','speculative')
    return dict(start=start,end=finish,slots=[{k:r[k] for k in keys if k in r} for r in rows],
                raw_keys=[sorted(r) for r in rows])


def queue_visibility(url, destination):
    rng=random.Random(SEED ^ 0x1791)
    result=[]
    def probe(expected):
        held=snapshot(url)
        with send(url,'/metrics') as r:
            m={}
            for row in r.read().decode().splitlines():
                if not row or row.startswith('#'):continue
                pair=row.split()
                if len(pair)>=2 and any(k in pair[0] for k in ('request','slot','cache')):
                    m[pair[0]]=float(pair[1])
        return dict(expected_pending=expected,snapshot=held,metrics=m)
    for i in (0,1):slot_api(url,i,'erase')
    with ThreadPoolExecutor(4) as pool:
        def launch(slot,n):
            sent=Event();first=Event()
            prompt=[rng.randrange(1000,15000) for _ in range(96)]
            return pool.submit(stream,url,slot,prompt,n,first,sent),sent,first
        a,_,a_first=launch(0,160);b,_,b_first=launch(1,160)
        assert a_first.wait(45) and b_first.wait(45)
        result.append(probe(0))
        q1,s1,_=launch(1,2);assert s1.wait(5);result.append(probe(1))
        q2,s2,_=launch(1,2);assert s2.wait(5);result.append(probe(2))
        outcomes=[x.result() for x in (a,b,q1,q2)]
    payload=dict(probes=result,requests=outcomes)
    destination.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('queue visibility:',[x['metrics'] for x in result],flush=True)


def run(args):
    rng=random.Random(SEED); order=random.Random(SEED ^ 0x279b)
    with args.out.open('w',encoding='utf-8',newline='\n') as f:
        def write(r):f.write(json.dumps(r,separators=(',',':'))+'\n');f.flush()
        write(dict(meta=True,props=query(args.url,'/props'),seed=SEED))
        for b in range(-1,REPEATS):
            t,p=([rng.randrange(1000,15000) for _ in range(n)] for n in (1536,96))
            pair=sha256(json.dumps([t,p]).encode()).hexdigest()
            for d in (512,1024):
                slot_api(args.url,1,'erase')
                query(args.url,'/completion',dict(id_slot=1,prompt=t[:d],n_predict=1,cache_prompt=False,temperature=0))
                assert slot_api(args.url,1,'save',f'r07-live-{d}.bin')['n_saved']==d
            arms=ARMS.copy();order.shuffle(arms)
            for mode,d in arms:
                for i in (0,1):slot_api(args.url,i,'erase')
                first=Event()
                with ThreadPoolExecutor(1) as pool:
                    r=pool.submit(stream,args.url,0,p,64,first)
                    assert first.wait(45)
                    trigger=perf_counter_ns(); observations=[]; restore=target=begin=end=None
                    if mode=='probe':observations.append(snapshot(args.url))
                    if mode!='peer':
                        begin=perf_counter_ns(); restore=slot_api(args.url,1,'restore',f'r07-live-{d}.bin'); end=perf_counter_ns()
                        assert restore['n_restored']==d and restore['n_read']>0
                    if mode=='probe':observations.append(snapshot(args.url))
                    if mode in ('clean','probe'):
                        target=stream(args.url,1,t,2)
                        assert target['timings']['cache_n']==d and target['timings']['prompt_n']==1536-d
                    peer=r.result();assert peer['timings']['prompt_n']==96
                write(dict(block=b,mode=mode,depth=d,prompt_hash=pair,trigger=trigger,observe=observations,
                           restore=restore,restore_start=begin,restore_end=end,target=target,peer=peer))
                print('r07-observer',b,mode,d,flush=True)
    queue_visibility(args.url,args.out.with_name('r07-live-queue-interface.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:18298');p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
