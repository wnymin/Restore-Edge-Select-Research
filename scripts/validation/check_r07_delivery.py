"""Verify R07 frozen inputs, archived summaries, and optionally all exact results."""
import argparse
from hashlib import sha256
import gzip

from scripts.cpu.replay.study_r07_arrivals import CONFIG, POLICIES, ROOT, cases, file_hash, read, run, search, sequence
from scripts.validation.summarize_r07 import load, summary

EVIDENCE = ROOT/'docs/evidence'


def replay(cfg, split, rows):
    for (key, exp), rec in zip(cases(cfg, split),rows,strict=True):
        assert key == rec['key']
        z = {p:run(exp,p) for p in POLICIES}
        assert all(sequence(x)==rec['zero'][p]['sequence'] for p,x in z.items()), key
        oracle=rec['oracle']
        upper=exp.run('fixed',boundaries=oracle['plan'])['response_sum_ms']
        assert float(upper/rec['n'])==oracle['upper']
        if split=='short':
            lower=exp.offline()['response_sum_ms']; assert lower==upper
        else:
            lower, certified_upper, _, used=search(exp,list(z.values()),cfg['node_budget'],resource_floor=True,dominance=True)
            assert certified_upper==upper and used==oracle['nodes']
        assert float(lower/rec['n'])==oracle['lower'] and (lower==upper)==oracle['exact']
    print('Replayed',len(rows),split,'configurations and recalculated oracle certificates.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true'); args=p.parse_args(); cfg=read(CONFIG)
    splits={}
    for split,count in [('short',60),('design',42),('holdout',252)]:
        f=EVIDENCE/f'r07-{split}.jsonl.gz'; prov=read(EVIDENCE/f'r07-{split}-provenance.json')
        assert sha256(gzip.decompress(f.read_bytes())).hexdigest()==prov['output_sha256']
        if split!='design': # historical design used the pre-freeze matrix and driver
            assert all(file_hash(ROOT/p)==h for p,h in prov['files'].items()), split
        splits[split]=load(f)
        assert len(splits[split])==count
        if split=='design':
            assert {x['key']['window'] for x in splits[split]}==set(cfg['design_windows'])
    assert not set(cfg['design_windows']) & set(cfg['holdout_windows']) and cfg['locked']
    result=summary(splits['holdout'],cfg)
    assert result==read(EVIDENCE/'r07-summary.json')
    short,holdout=(read(EVIDENCE/f'r07-{s}-witness.json') for s in ('short','holdout'))
    for split,w in [('short',short),('holdout',holdout)]:
        assert len(w)==sum(r['zero']['entry-replay']['mean']-r['oracle']['upper']>1e-6 for r in splits[split])
        assert all(r['pruned_matches_exhaustive'] or r['coarse_return'] for r in w)
    assert any(x['future_pair']['same_snapshot'] for x in short if x['future_pair'])
    assert any(x['all_seen_choice']!=x['window4_choice'] for x in holdout)
    assert any(x['joint_choice']!=x['all_seen_choice'] for x in holdout)
    tail=read(EVIDENCE/'r07-tail-witness.json')
    assert tail['rounds_gt10']>=4 and tail['zero_extra_ms']>10 and any(t['request_loss_ms']==0 for t in tail['checks'])
    if args.replay:
        for split in ('short','holdout'):
            replay(cfg,split,splits[split])
    print('R07 verified: 60 short, 42 design, 252 holdout, summary, diagnostic coverage, tail intervention.')


if __name__=='__main__':
    main()
