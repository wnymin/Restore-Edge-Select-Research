"""Report finite-grid ambiguity of fixed-plan winners under score error."""
from fractions import Fraction as F
import json

from scripts.validation.study_r03_capacity import read


def evaluate():
    results = {}
    for split, rows in read('r03-cpu-interaction.json').items():
        unique = {}
        for row in rows:
            for cell in row['cells']:
                key = row['id'], cell['delta_bytes'], cell['ssd_payload_multiplier']
                assert key not in unique or unique[key] == cell
                unique[key] = cell
        result = []
        for (case, delta, scale), cell in sorted(unique.items()):
            scores = {p['boundary']: F(p['response_sum_ms']) / len(cell['order'])
                      for p in cell['plans']}
            best = scores[cell['winner']]
            alternatives = sorted((s, boundary) for boundary, s in scores.items()
                                  if boundary != cell['winner'])
            gap = alternatives[0][0] - best
            assert gap >= 0
            # Independent additive errors bounded by eps on every candidate
            # mean can change a pairwise score difference by at most 2 eps.
            result.append(dict(id=case, capacity_delta=delta, multiplier=scale,
                               best=cell['winner'], exact_tie=gap == 0, margin_ms=str(gap),
                               critical_error_ms=str(gap / 2),
                               ambiguous=[str(eps) for eps in (F(1,20), F(1,4), F(1))
                                          if gap <= 2 * eps]))
        results[split] = result
    return results


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
