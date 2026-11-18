"""Independently reconstruct archived common-partition evidence counts."""

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction as F
from scripts.validation.lookahead_evidence import csv_paths, summary_path
from scripts.cpu.lookahead.robustness.study_lookahead_candidate_difference import Affine, Domain, classify, extrema


def check(path):
    groups = defaultdict(list)
    for row in csv.DictReader(path.open(encoding='utf-8')):
        groups[int(row['pair'])].append(row)
    counts = Counter()
    for pair, rows in groups.items():
        cells = [(Domain(F(r['lo']),F(r['hi']),bool(int(r['left'])),bool(int(r['right']))),
                  Affine(F(r['intercept']),F(r['slope']))) for r in rows]
        assert cells[0][0].left and cells[-1][0].right
        for a,b in zip(cells,cells[1:]):
            assert a[0].hi==b[0].lo and a[0].right != b[0].left
        low,high=extrema(cells)
        baseline=rows[0]
        for row,(domain,delta) in zip(rows,cells):
            assert domain.nonempty()
            if domain.lo<domain.hi:
                assert not domain.left and not domain.right
            assert classify(delta,domain)==row['sign']
            for field in ('case','snapshot','head','left_candidate','right_candidate',
                          'pair_lower','pair_upper','independent_lower','independent_upper',
                          'workload_lower','workload_upper'):
                assert row[field]==baseline[field],(pair,field)
            counts['cell_'+row['sign']]+=1
            points={domain.lo} if domain.lo==domain.hi else {
                (3*domain.lo+domain.hi)/4,(domain.lo+domain.hi)/2,
                (domain.lo+3*domain.hi)/4}
            if delta.b and domain.contains(-delta.a/delta.b):
                points.add(-delta.a/delta.b)
            counts['audited_points']+=len(points)
        assert (low,high)==(F(baseline['pair_lower']),F(baseline['pair_upper']))
        il,iu=F(baseline['independent_lower']),F(baseline['independent_upper'])
        wl,wu=F(baseline['workload_lower']),F(baseline['workload_upper'])
        assert il<=low<=high<=iu and wl<=low<=high<=wu
        cert=low>0 or high<0
        counts['pairs']+=1
        counts['cells']+=len(cells)
        counts['paired_certified']+=cert
        counts['independent_certified']+=il>0 or iu<0
        counts['workload_certified']+=wl>0 or wu<0
        counts['paired_only_vs_independent']+=cert and not(il>0 or iu<0)
        counts['paired_only_vs_workload']+=cert and not(wl>0 or wu<0)
        counts['strictly_tighter']+=low>il or high<iu
        counts['winner_changes']+=low<0<high
        counts['singleton_cells']+=sum(d.lo==d.hi for d,_ in cells)
    summary=json.loads(summary_path(path).read_text())
    for field,value in counts.items():
        assert summary.get(field,0)==value,(path,field,value,summary.get(field))
    print(path.name, dict(counts))


if __name__=='__main__':
    for path in csv_paths('dynamic-lookahead-candidate-difference-*.csv'):
        check(path)
