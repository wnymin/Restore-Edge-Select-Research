"""Recompute the archived R03 collaborator evidence without external inputs."""
from fractions import Fraction as F
from hashlib import sha256
import subprocess
import sys

from scripts.validation import study_r03_capacity as capacity
from scripts.validation import study_r03_clock_witness as clock_witness
from scripts.validation import study_r03_estimate as estimate
from scripts.validation import study_r03_estimate_timing as timing
from scripts.validation import study_r03_estimate_interaction as joint_estimate
from scripts.validation import study_r03_interaction as interaction
from scripts.validation import study_r03_margin as margin
from scripts.validation import study_r03_oracle as oracle
from scripts.validation import study_r03_proxy as proxy
from scripts.validation import study_r03_sharing as sharing
from scripts.validation import study_r03_visibility as visibility


def check_inputs():
    manifest = capacity.read('r03-cpu-input-manifest.json')
    for name, expected in manifest['inputs'].items():
        content = (capacity.EVIDENCE / name).read_bytes().replace(b'\r\n', b'\n')
        if sha256(content).hexdigest() != expected:
            raise ValueError(f'R03 source archive changed: {name}')


def check_timing_archive(name):
    archive = capacity.read(f'r03-cpu-{name}-timing.json')
    rows = archive['rows']
    groups = {}
    for row in rows:
        key = row['id'], row['multiplier']
        groups.setdefault(key, []).append(row)
        for field in ('net_mean_loss_ms', 'max_individual_loss_ms'):
            row[field] = F(row[field])
        old, new = (list(map(F, row[field])) for field in ('baseline_response', 'biased_response'))
        if row['net_mean_loss_ms'] != (sum(new)-sum(old))/len(old):
            raise ValueError(f'Timing response mismatch: {key}')
    for key, group in groups.items():
        if sorted(r['trial'] for r in group) != list(range(archive['repeats'])):
            raise ValueError(f'Timing repetitions mismatch: {key}')
        if 2 * sum(r['baseline_first'] for r in group) != len(group):
            raise ValueError(f'Timing order imbalance: {key}')
    if timing.jsonable(timing.summarize(rows)) != archive['summary']:
        raise ValueError('Timing summary differs from archived pairs')
    print(f'Checked R03 timing archive ({len(rows)} pairs)', flush=True)


def main():
    check_inputs()
    for name in ('estimate', 'target'):
        check_timing_archive(name)
    studies = [(name, capacity.evaluate(name)) for name in ('capacity', 'ssd', 'release')]
    studies.extend((name, module.evaluate()) for name, module in (
        ('clock-witness', clock_witness), ('estimate', estimate), ('estimate-interaction', joint_estimate), ('interaction', interaction),
        ('margin', margin), ('oracle', oracle), ('proxy', proxy), ('sharing', sharing), ('visibility', visibility),
    ))
    studies.append(('compute-estimate', estimate.evaluate('compute')))
    for name, actual in studies:
        if actual != capacity.read(f'r03-cpu-{name}.json'):
            raise ValueError(f'R03 evidence differs: {name}')
        print(f'Matched R03 {name}', flush=True)
    subprocess.run([sys.executable, 'scripts/check_fixture.py'], cwd=capacity.ROOT, check=True)


if __name__ == '__main__':
    main()
