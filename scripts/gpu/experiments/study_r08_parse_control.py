"""R08-GPU-02: paired parsing-only control on the archived 16-layer solo prompts."""
import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import random
import socket
from subprocess import CREATE_NO_WINDOW, Popen
from time import monotonic, sleep

from scripts.gpu.experiments.study_r07_live_scheduling import query, send, slot_api


def digest(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def trial(url,prompt):
    slot_api(url,0,'erase')
    ids=[]; final=None; errors=[]; chunks=0
    payload=dict(id_slot=0,prompt=prompt,n_predict=64,stream=True,cache_prompt=True,
                 temperature=0,ignore_eos=True,return_tokens=True)
    with send(url,'/completion',payload) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            raw=line[5:].strip()
            if raw==b'[DONE]': break
            event=json.loads(raw)
            if event.get('error'):
                errors.append(event['error'])
            if event.get('stop') or 'timings' in event:
                final=event.get('timings')
            if event.get('tokens') or event.get('content'):
                chunks+=1
                if event.get('tokens'):
                    assert all(isinstance(t,int) for t in event['tokens'])
                    ids.extend(event['tokens'])
    if final is not None:
        assert final['cache_n']==0 and final['prompt_n']==96
    return dict(token_ids=ids,chunks=chunks,predicted_n=final['predicted_n'] if final else None,errors=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True,type=Path); p.add_argument('--model',required=True,type=Path)
    p.add_argument('--device',default='Vulkan1'); p.add_argument('--out',required=True,type=Path); a=p.parse_args()
    server=a.server.resolve(); model=a.model.resolve()
    pins={'model':digest(model),**{k:digest(server.parent/k) for k in
          ('llama-server.exe','llama-server-impl.dll','llama.dll','ggml-vulkan.dll')}}
    env=Path(__file__).resolve().parents[3]/'docs/evidence/r07-live-environment.json'
    old=json.loads(env.read_bytes())
    if pins['model']!=old['model_sha256']['r08-16-layer-calibration.gguf'] or any(
       pins[k]!=v for k,v in old['binary_sha256'].items()):
        p.error('archived model and server binary pins are required')
    rng=random.Random(464277)
    prompts=[]
    for _ in range(9):
        _,peer,_=([rng.randrange(1000,15000) for _ in range(n)] for n in (1792,96,128))
        prompts.append(peer)
    rows={}
    for label,option in (('default','--no-skip-chat-parsing'),('skip','--skip-chat-parsing')):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        cmd=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(port),'-np','2',
             '-c','8192','-b','256','-ub','128','-ngl','99','--device',a.device,'--cache-ram','0',
             '--slot-save-path',str(a.out.parent.resolve()),option]
        log=a.out.with_name(a.out.stem+'-'+label+'.log')
        with log.open('xb') as f:
            helper=Popen(cmd,cwd=server.parent,stdout=f,stderr=f,creationflags=CREATE_NO_WINDOW)
            try:
                deadline=monotonic()+90
                while monotonic()<deadline and helper.poll() is None:
                    try:
                        props=query(url,'/props'); break
                    except OSError:
                        sleep(0.5)
                else:
                    raise RuntimeError(f'server not ready: {log}')
                assert '8a56aedd6' in props['build_info'] and props['total_slots']==2
                rows[label]=[trial(url,p) for p in prompts]
            finally:
                helper.terminate(); helper.wait(timeout=10)
    pairs=[]
    for b,(prompt,default,skip) in enumerate(zip(prompts,rows['default'],rows['skip'],strict=True),-1):
        common=min(len(default['token_ids']),len(skip['token_ids']))
        pairs.append(dict(block=b,prompt_hash=sha256(json.dumps(prompt).encode()).hexdigest(),
            common_tokens=common,common_prefix_equal=bool(common) and default['token_ids'][:common]==skip['token_ids'][:common],
            default=default,skip=skip))
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(binary_sha256=pins,device=a.device,order=['default','skip'],pairs=pairs),f,indent=2); f.write('\n')


if __name__=='__main__':
    main()
