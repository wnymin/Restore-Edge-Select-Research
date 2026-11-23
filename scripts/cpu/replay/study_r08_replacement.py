"""Run the predeclared single-window R08 holdout correction, leaving old artifacts intact."""
import argparse
import json
from pathlib import Path
import platform

from scripts.cpu.replay.study_r08_capacity import (
    CONFIG, ROOT, cases, dependencies, examine, file_hash, open_exp, paired, read, write_rows,
)

AMENDMENT=ROOT/'configs/r08-capacity-holdout-amendment.json'


def corrected_config(cfg, patch):
    trace=patch['trace']; old=patch['exclude_window']; new=patch['replacement_window']
    assert cfg['holdout'][trace].count(old)==1 and new not in cfg['holdout'][trace]
    return dict(cfg,holdout={t:[new if t==trace and w==old else w for w in ws]
                             for t,ws in cfg['holdout'].items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('split',choices=['replacement','replacement-timed'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--untimed',type=Path)
    a=p.parse_args()
    if (a.split=='replacement-timed')!=(a.untimed is not None): p.error('timed requires --untimed')
    patch,cfg=read(AMENDMENT),read(CONFIG)
    trace,old,new=patch['trace'],patch['exclude_window'],patch['replacement_window']
    corrected_config(cfg,patch)
    window_index=list((t,w) for t,ws in cfg['holdout'].items() for w in ws).index((trace,old))
    per_window=len(cfg['main_models'])*(len(cfg['rates'])*(len(cfg['fixed_mib'])+4)+len(cfg['negative_control_mib']))
    replacement=dict(cfg,holdout={trace:[new]})

    def replay():
        for k,d,m,s in cases(replacement,'holdout'):
            yield examine(k,open_exp(d,m,k['rate'],k['capacity_bytes'],s),cfg['node_budget'],diagnosis=True)

    def timed():
        with a.untimed.open(encoding='utf-8') as f:
            for i,(task,record) in enumerate(zip(cases(replacement,'holdout'),map(json.loads,f),strict=True)):
                k,d,m,s=task
                assert k==record['key']
                yield dict(key=k,timed=paired(open_exp(d,m,k['rate'],k['capacity_bytes'],s),cfg['timing_repeats'],
                                              window_index*per_window+i))

    write_rows(timed() if a.untimed else replay(),a.out)
    meta=dict(split=a.split,platform=platform.platform(),python=platform.python_version(),
              input_sha256=file_hash(a.untimed) if a.untimed else None,output_sha256=file_hash(a.out),
              files={f.relative_to(ROOT).as_posix():file_hash(f) for f in
                     [*dependencies(),AMENDMENT,Path(__file__).resolve()]})
    a.out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
