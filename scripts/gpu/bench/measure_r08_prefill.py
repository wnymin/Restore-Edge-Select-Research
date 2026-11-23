"""Measure paired R08 prefill grids on the same local GPU and runtime binary."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from hashlib import file_digest


def sha(path):
    with path.open('rb') as f:return file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bench',type=Path);ap.add_argument('anchor',type=Path);ap.add_argument('derived',type=Path)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    cfg=json.loads(Path('configs/r08-native-calibration.json').read_text(encoding='utf-8'))
    args=[]; blobs={}
    for split in ('grid','observe'):
        dims=('depth','suffix') if split=='grid' else ('observational_depth','observational_suffix')
        for label,weight in [('4b-native',a.anchor),('16l-native',a.derived)]:
            flags=['-p',','.join(map(str,cfg[dims[1]])),'-d',','.join(map(str,cfg[dims[0]])),
                   '-n','0','-t','24','-b','2048','-ub','512','-ngl','99','-dev','Vulkan1',
                   '-o','json','-r',str(cfg['prefill_repeats'])]
            command=[str(a.bench.resolve()),'-m',str(weight.resolve()),*flags]
            print(f'{label} {split} begin',flush=True)
            began=datetime.now(timezone.utc).isoformat()
            q=subprocess.run(command,capture_output=True,check=False)
            logfile=a.out.with_name(f'{a.out.stem}-{label}-{split}.log')
            logfile.write_bytes(q.stderr)
            q.check_returncode()
            data=json.loads(q.stdout)
            assert len(data)==len(cfg[dims[0]])*len(cfg[dims[1]])
            assert {(r['n_depth'],r['n_prompt']) for r in data}=={(i,j) for i in cfg[dims[0]] for j in cfg[dims[1]]}
            assert all(r['n_gen']==0 and r['devices']=='Vulkan1' and r['n_gpu_layers']==99
                       and len(r['samples_ns'])==cfg['prefill_repeats'] for r in data)
            for r in data:r['model_filename']=weight.name
            blobs.setdefault(label,{})[split]=data
            args.append(dict(model=label,kind=split,flags=flags,started_utc=began,
                             finished_utc=datetime.now(timezone.utc).isoformat(),stderr_sha256=sha(logfile)))
            print(f'{label} {split} {len(data)} shapes',flush=True)
    provenance=dict(bench_sha256=sha(a.bench),runtime_sha256={n:sha(a.bench.with_name(n)) for n in (
                          'llama-bench-impl.dll','llama.dll','ggml-vulkan.dll') if a.bench.with_name(n).is_file()},
                    local_model_sha256=sha(a.anchor),derived_model_sha256=sha(a.derived),calls=args,
                    tool_sha256=sha(Path(__file__)),driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
    with a.out.open('x',encoding='utf-8') as f:json.dump(dict(provenance=provenance,points=blobs),f,indent=2)


if __name__=='__main__':main()
