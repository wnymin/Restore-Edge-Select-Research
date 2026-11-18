"""Reconstruct first-decision action selection and cross-budget invariants."""

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from scripts.validation.lookahead_evidence import csv_paths, summary_path


def check(path):
    groups=defaultdict(list)
    for row in csv.DictReader(path.open(encoding='utf-8')):
        groups[int(row['decision'])].append(row)
    counts=Counter()
    for key,rows in groups.items():
        first=rows[0]
        actions={int(r['candidate']):r for r in rows}
        fallback=int(first['fallback'])
        safe={c for c,r in actions.items() if F(r['individual_upper'])<=F(r['cap'])}
        for c,r in actions.items():
            assert bool(int(r['safe']))==(c in safe)
            assert F(r['regret'])>=0
        assert fallback in safe and F(actions[fallback]['individual_upper'])==0
        order=lambda c:(F(actions[c]['regret']),c!=fallback,-c)
        robust=min(actions,key=order)
        chosen=min(safe,key=order)
        assert int(first['robust'])==robust and int(first['constrained'])==chosen
        assert [c for c,r in actions.items() if int(r['selected'])]==[chosen]
        assert F(actions[chosen]['regret'])<=F(actions[fallback]['regret'])
        for r in rows:
            for field in first:
                if field not in {'candidate','regret','individual_upper','safe','selected'}:
                    assert r[field]==first[field],(path,key,field)
        counts['decisions']+=1
        counts['candidate_actions']+=len(rows)
        counts['cells']+=int(first['cells'])
        counts['probes']+=int(first['probes'])
        counts['running_decisions']+=int(first['running'])
        counts['future_target_decisions']+=int(first['distance'])>0
        counts['three_plus_candidates']+=len(rows)>=3
        counts['winner_changes']+=int(first['winners'])>1
        counts['robust_differs_nominal']+=robust!=int(first['nominal'])
        counts['constraint_changes_action']+=robust!=chosen
        counts['constrained_differs_fallback']+=chosen!=fallback
        counts['unsafe_actions']+=len(rows)-len(safe)
        if 'inflight' in first:
            counts['no_inflight_decisions']+=not int(first['inflight'])
            counts['parameter_sensitive_decisions']+=int(first['parameter_sensitive'])
    summary=json.loads(summary_path(path).read_text())
    for key,value in counts.items():
        assert summary.get(key,0)==value,(path,key,value,summary.get(key))
    if not groups:
        assert summary.get('decisions',0)==0
    print(path.name,dict(counts))
    return groups


def compare(base,other,*,wider):
    assert base.keys()==other.keys()
    for key,rows in base.items():
        ref={r['candidate']:r for r in rows}
        alt={r['candidate']:r for r in other[key]}
        assert ref.keys()==alt.keys()
        for c,r in ref.items():
            a=alt[c]
            for field in ('seed','case','target','head','fallback','residual'):
                assert r[field]==a[field]
            if wider:
                assert F(a['regret'])>=F(r['regret'])
                assert F(a['individual_upper'])>=F(r['individual_upper'])
                assert not int(a['safe']) or int(r['safe'])
            else:
                assert a['regret']==r['regret'] and a['individual_upper']==r['individual_upper']
                assert not int(r['safe']) or int(a['safe'])
        if not wider:
            chosen=lambda actions:next(F(r['regret']) for r in actions if int(r['selected']))
            assert chosen(other[key])<=chosen(rows)


if __name__=='__main__':
    groups={p.stem:check(p) for p in csv_paths('dynamic-lookahead-first-decision-*.csv')}
    prefix='dynamic-lookahead-first-decision-7710'
    for suffix,wider in (('-wide',True),('-cap1',False)):
        if prefix+suffix in groups:
            compare(groups[prefix],groups[prefix+suffix],wider=wider)
    print('Action reconstruction, safe-set nesting and regret monotonicity pass.')
