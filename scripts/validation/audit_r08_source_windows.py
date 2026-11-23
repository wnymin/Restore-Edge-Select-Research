"""Check R08 source-window identities against prior structured evidence and zip members."""
import csv
import gzip
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'docs/evidence'


def streams():
    for p in sorted(INPUT.iterdir()):
        if p.name.startswith('r08-') or p.suffix not in ('.csv','.json','.jsonl','.zip','.gz'):
            continue
        if p.suffix=='.zip':
            with ZipFile(p) as z:
                for n in sorted(z.namelist()):
                    if n.endswith(('.csv','.json','.jsonl')):
                        yield p.name+'::'+n,z.read(n)
        elif p.suffix=='.gz':
            yield p.stem,gzip.decompress(p.read_bytes())
        else:
            yield p.name,p.read_bytes()


def objects(x):
    if isinstance(x,dict):
        yield x
        for v in x.values(): yield from objects(v)
    elif isinstance(x,list):
        for v in x: yield from objects(v)


def keys(name,b):
    if name.endswith('.csv'):
        header=b.split(b'\n',1)[0]
        if b'window' not in header or not any(x in header for x in (b'trace',b'source')):
            return
        records=csv.DictReader(StringIO(b.decode('utf-8-sig')))
    elif name.endswith(('.json','.jsonl')):
        if not any(x in b for x in (b'qwen_trace',b'qwen_coder')):return
        roots=map(json.loads,b.splitlines()) if name.endswith('.jsonl') else (json.loads(b),)
        records=(row for root in roots for row in objects(root))
    else:
        return
    for r in records:
        w=r.get('window'); t=r.get('trace') or r.get('source')
        if isinstance(t,str) and t.startswith('qwen_') and str(w).isdigit():
            yield (t,int(w))


def audit():
    cfg=lambda filename:json.loads((ROOT/'configs'/filename).read_bytes())
    main=cfg('r08-capacity-matrix.json')['holdout']
    native=cfg('r08-native-calibration.json')['heldout']
    pairset=lambda x:{(t,w) for t,ws in x.items() for w in ws}
    old=pairset({t:[104,105,106,107,119] for t in ('qwen_traceA_blksz_16.jsonl','qwen_coder_blksz_16.jsonl')})
    amendment=cfg('r08-capacity-holdout-amendment.json')
    excluded=(amendment['trace'],amendment['exclude_window'])
    replacement=(amendment['trace'],amendment['replacement_window'])
    hold=pairset(main); calibration=pairset(native)
    assert excluded in hold and replacement not in (hold|calibration)
    hold.remove(excluded);hold.add(replacement)
    old.add(excluded)
    assert hold.isdisjoint(calibration)
    confirmed={}; overlaps=[]; n=0
    for name,content in streams():
        n+=1
        for k in set(keys(name,content)):
            if k in old:confirmed.setdefault(k,set()).add(name)
            if k in hold|calibration:overlaps.append((k,name))
    assert not overlaps, sorted(overlaps)
    assert len(confirmed)==len(old) and any('replay-inputs.zip::' in n for files in confirmed.values() for n in files)
    return dict(streams_scanned=n,main_windows=len(hold),native_windows=len(calibration),
                overlaps=len(overlaps),replaced_windows=len(confirmed))


if __name__=='__main__': print('R08 historical window audit:',audit())
