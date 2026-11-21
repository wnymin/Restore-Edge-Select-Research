"""Compare single and joint decision-cost errors on unchanged execution."""
from fractions import Fraction as F
import json

from scripts.validation.study_r03_estimate import evaluate as replay, read


def evaluate():
    key = lambda row: (row['id'], row['multiplier'])
    ssd = {key(row): row for row in read('r03-cpu-estimate.json')}
    compute = {key(row): row for row in read('r03-cpu-compute-estimate.json')}
    rows = []
    for multiplier in ('1/2', '2'):
        for row in replay(companion=F(multiplier)):
            s = ssd[key(row)]
            c = compute[row['id'], multiplier]
            assert s['baseline_actions'] == c['baseline_actions'] == row['baseline_actions']
            assert s['baseline_responses'] == c['baseline_responses'] == row['baseline_responses']
            excess = F(row['mean_loss_ms']) - F(s['mean_loss_ms']) - F(c['mean_loss_ms'])
            rows.append(dict(**row, compute_multiplier=multiplier,
                             ssd_only_actions=s['estimate_actions'], compute_only_actions=c['estimate_actions'],
                             joint_only_change=row['changed'] and not s['changed'] and not c['changed'],
                             mean_loss_above_additive_ms=str(excess)))
    return rows


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
