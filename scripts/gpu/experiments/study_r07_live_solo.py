"""Replay the exact held-out peer prompts alone to discriminate SSE failures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

from scripts.gpu.experiments.study_r07_live_scheduling import query, stream


def run(a):
    rng=random.Random(464277); meta=dict(meta=True,arm=a.arm,props=query(a.url,'/props'))
    with a.out.open('w',encoding='utf-8',newline='\n') as f:
        def out(r):f.write(json.dumps(r,separators=(',',':'))+'\n');f.flush()
        out(meta)
        for b in range(-1,8):
            _,peer,_=([rng.randrange(1000,15000) for _ in range(n)] for n in (1792,96,128))
            row=dict(block=b,prompt_hash=sha256(json.dumps(peer).encode()).hexdigest())
            try:
                q=stream(a.url,0,peer,64)
                assert q['timings']['cache_n']==0 and q['timings']['prompt_n']==96, q['timings']
                row['result']=q
            except (AssertionError,OSError,RuntimeError,KeyError,ValueError) as exc:row['error']=str(exc)
            out(row);print(a.arm,b,row.get('error','ok'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:18299')
    p.add_argument('--arm',required=True);p.add_argument('--out',required=True,type=Path);run(p.parse_args())
