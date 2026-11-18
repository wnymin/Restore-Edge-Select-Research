"""Reconstruct budget frontiers; distinguish value and tie-safe dominance."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path


def frontier(actions,fallback):
    # For a nonnegative budget, negative H and zero H activate together.
    data={int(r['candidate']):(max(F(0),F(r['individual_upper'])),F(r['regret'])) for r in actions}
    priority=lambda c:(c!=fallback,-c)
    value_dominated={a for a,(h,r) in data.items() if any(
        a!=b and k<=h and s<=r and (k<h or s<r) for b,(k,s) in data.items())}
    # Strictly better regret dominates regardless of tie. With equal regret,
    # deleting an action requires the replacement's tie priority to be better.
    tie_safe_dominated={a for a,(h,r) in data.items() if any(
        a!=b and k<=h and (s<r or s==r and priority(b)<priority(a))
        for b,(k,s) in data.items())}
    retained=set(data)-tie_safe_dominated
    thresholds=sorted({F(0)}|{h for h,r in data.values()})
    rows=[]
    last=None
    for index,budget in enumerate(thresholds):
        upper=thresholds[index+1] if index+1<len(thresholds) else None
        safe={c for c,(h,r) in data.items() if h<=budget}
        rank=lambda c:(data[c][1],*priority(c))
        winner=min(safe,key=rank)
        assert min(safe & retained,key=rank)==winner
        if last is not None:
            assert data[winner][1]<=last
        last=data[winner][1]
        # Check interior and the point immediately below each activation.
        probes={budget}
        if upper is not None:
            probes.add((budget+upper)/2)
        for x in probes:
            eligible={c for c,(h,r) in data.items() if h<=x}
            assert min(eligible,key=rank)==winner
            assert min(eligible & retained,key=rank)==winner
        rows.append(dict(budget_lower=str(budget),budget_upper='' if upper is None else str(upper),
                         winner=winner,regret=str(data[winner][1]),safe_actions=len(safe),
                         retained_actions=len(retained),value_dominated=len(value_dominated),
                         tie_safe_dominated=len(tie_safe_dominated)))
    return rows,value_dominated,tie_safe_dominated


def self_check():
    actions=[dict(candidate=1,individual_upper='0',regret='2'),
             dict(candidate=2,individual_upper='1',regret='2'),
             dict(candidate=3,individual_upper='2',regret='0')]
    rows,value,tie=frontier(actions,3)
    assert 2 in value and 2 not in tie
    assert [r['winner'] for r in rows]==[1,2,3]
    assert [r['regret'] for r in rows]==['2','2','0']


def main(args):
    self_check()
    groups=defaultdict(list)
    for row in csv.DictReader(args.input.open(encoding='utf-8')):
        groups[int(row['decision'])].append(row)
    output=[]
    stats=Counter()
    for key,actions in groups.items():
        fallback=int(actions[0]['fallback'])
        rows,value,tie=frontier(actions,fallback)
        stats['decisions']+=1
        stats['actions']+=len(actions)
        stats['frontier_intervals']+=len(rows)
        stats['value_dominated_actions']+=len(value)
        stats['tie_safe_dominated_actions']+=len(tie)
        stats['tie_sensitive_dominance_decisions']+=bool(value-tie)
        stats['budget_changes_action_decisions']+=len({r['winner'] for r in rows})>1
        stats['budget_improves_regret_decisions']+=F(rows[-1]['regret'])<F(rows[0]['regret'])
        cap=F(actions[0]['cap'])
        selected=next(r for r in rows if F(r['budget_lower'])<=cap and
                      (not r['budget_upper'] or cap<F(r['budget_upper'])))
        assert selected['winner']==int(actions[0]['constrained'])
        output.extend(dict(decision=key,**r) for r in rows)
    with args.out.open('x',newline='',encoding='utf-8') as f:
        fields=tuple(output[0]) if output else ('decision','budget_lower','budget_upper','winner','regret')
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    main(parser.parse_args())
