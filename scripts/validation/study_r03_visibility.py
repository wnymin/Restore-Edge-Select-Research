"""Extract R03 decision and response contrasts from exact R02 visibility runs."""
from fractions import Fraction as F
import json

from scripts.validation.study_r03_capacity import read


def ranking(plans):
    scores = {p['boundary']: F(p['response_sum_ms']) for p in plans}
    return sorted(scores, key=lambda b: (scores[b], -b)), sorted(b for b in scores if scores[b] == min(scores.values()))


def evaluate():
    result = {}
    for split, records in read('r02-cpu-visibility.json').items():
        rows = []
        for record in records:
            for kind, old, new in (('arrival', 'original', 'same_fifo'),
                                   ('fifo', 'same_fifo', 'file_order')):
                a, b = record[old], record[new]
                prior = {p['boundary'] for p in a['plans']}
                plans = {p['boundary']: p for p in b['plans']}
                assert prior == plans.keys()
                ar, at = ranking(a['plans'])
                br, bt = ranking(b['plans'])
                best, frozen = plans[b['winner']], plans[a['winner']]
                # Both perturbation comparison endpoints have all arrivals at 0.
                first, second = [list(map(F, p['completion_ms'])) for p in (best, frozen)]
                loss = (sum(second)-sum(first))/len(first)
                rows.append(dict(id=record['id'], kind=kind, reference_action=a['winner'],
                                 perturbed_action=b['winner'], rank_changed=ar != br,
                                 tie_set_changed=at != bt, legal_set_changed=False,
                                 mean_loss_ms=str(loss), mean_loss_fraction=str(loss/(sum(first)/len(first))),
                                 p95_loss_ms=str(max(second)-max(first)),
                                 maximum_individual_harm_ms=str(max(F(0), *(s-f for s,f in zip(second,first,strict=True))))))
        result[split] = rows
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
