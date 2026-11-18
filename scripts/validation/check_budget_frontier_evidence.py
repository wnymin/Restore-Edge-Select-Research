"""Audit archived frontiers against exhaustive budget-qualified action scans."""

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from scripts.validation.lookahead_evidence import csv_path, summary_path


def groups(path):
    result=defaultdict(list)
    for row in csv.DictReader(path.open(encoding='utf-8')):
        result[int(row['decision'])].append(row)
    return result


def check(source,path):
    actions=groups(source)
    frontier=groups(path)
    assert actions.keys()==frontier.keys()
    stats=Counter()
    for key,candidates in actions.items():
        data={int(r['candidate']):(max(F(0),F(r['individual_upper'])),F(r['regret'])) for r in candidates}
        fallback=int(candidates[0]['fallback'])
        priority=lambda c:(c!=fallback,-c)
        rank=lambda c:(data[c][1],*priority(c))
        budgets=sorted({F(0)}|{h for h,r in data.values()})
        recorded=frontier[key]
        assert [F(r['budget_lower']) for r in recorded]==budgets
        # Exhaustive pairwise scan, then recompute winners for all activation
        # thresholds and interval interiors without calling the producer.
        value=set()
        tie_safe=set()
        for c,(h,r) in data.items():
            for b,(k,s) in data.items():
                if c==b or k>h:
                    continue
                if s<=r and (s<r or k<h):
                    value.add(c)
                if s<r or s==r and priority(b)<priority(c):
                    tie_safe.add(c)
        winners=[]
        for i,row in enumerate(recorded):
            budget=budgets[i]
            upper=budgets[i+1] if i+1<len(budgets) else None
            assert row['budget_upper']==('' if upper is None else str(upper))
            probes=(budget,(budget+upper)/2) if upper is not None else (budget,budget+1)
            for probe in probes:
                safe={c for c,(h,r) in data.items() if h<=probe}
                chosen=min(safe,key=rank)
                assert chosen==int(row['winner'])
                assert min(safe-tie_safe,key=rank)==chosen
                assert data[chosen][1]==F(row['regret'])
            assert int(row['safe_actions'])==sum(h<=budget for h,r in data.values())
            assert int(row['retained_actions'])==len(data)-len(tie_safe)
            assert int(row['value_dominated'])==len(value)
            assert int(row['tie_safe_dominated'])==len(tie_safe)
            winners.append(int(row['winner']))
        stats['decisions']+=1
        stats['actions']+=len(data)
        stats['frontier_intervals']+=len(recorded)
        stats['value_dominated_actions']+=len(value)
        stats['tie_safe_dominated_actions']+=len(tie_safe)
        stats['tie_sensitive_dominance_decisions']+=bool(value-tie_safe)
        stats['budget_changes_action_decisions']+=len(set(winners))>1
        stats['budget_improves_regret_decisions']+=F(recorded[-1]['regret'])<F(recorded[0]['regret'])
    summary=json.loads(summary_path(path).read_text())
    assert dict(stats)==summary,(path,stats,summary)
    print(path.name,dict(stats))


if __name__=='__main__':
    for suffix in ('7721','7722','witness-7720'):
        check(csv_path(f'dynamic-lookahead-first-decision-ssd-{suffix}.csv'),
              csv_path(f'dynamic-lookahead-budget-frontier-{suffix}.csv'))
