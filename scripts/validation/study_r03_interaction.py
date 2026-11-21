"""Measure capacity × SSD sensitivity with fixed anchored request traces."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json

from scripts.validation.study_r03_capacity import read, replay


def evaluate():
    result = {}
    for split in ('design', 'holdout'):
        rows = []
        for anchor in read('r02-cpu-inputs.json')['cases']:
            if anchor['split'] != split or anchor['id'].split('-')[2] != '+0':
                continue
            cells = {}
            for delta, multiplier in product((-1, 0, 1), (F(1, 2), F(1), F(2))):
                case = deepcopy(anchor)
                case['capacity_bytes'] += delta
                cost = case['trace']['cost']['ssd_to_device']
                cost['bytes_per_ms'] = str(F(cost['bytes_per_ms']) / multiplier)
                values, = replay({'cases': [case]}, split)
                cells[delta, multiplier] = {**values, 'delta_bytes': delta,
                                             'ssd_payload_multiplier': str(multiplier)}
            for delta, multiplier in product((-1, 1), (F(1, 2), F(2))):
                control = cells[0, F(1)]
                capacity_only = cells[delta, F(1)]
                cost_only = cells[0, multiplier]
                joint = cells[delta, multiplier]
                # Pair within each cost stratum to see whether the capacity
                # perturbation is decision-relevant at that cost.
                capacity_flip_nominal = capacity_only['winner'] != control['winner']
                capacity_flip_shifted = joint['winner'] != cost_only['winner']
                score = {p['boundary']: F(p['response_sum_ms']) for p in joint['plans']}
                loss = (score[control['winner']] - score[joint['winner']]) / len(joint['order'])
                rows.append(dict(id=anchor['id'], delta_bytes=delta, multiplier=str(multiplier),
                                 capacity_flip_nominal=capacity_flip_nominal,
                                 capacity_flip_shifted=capacity_flip_shifted,
                                 joint_only_flip=(capacity_only['winner'] == cost_only['winner']
                                                  == control['winner'] != joint['winner']),
                                 frozen_mean_loss_ms=str(loss),
                                 cells=[control, capacity_only, cost_only, joint]))
        result[split] = rows
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
