"""Recompute the deterministic R02 handoff from the pinned reference checkout."""
import subprocess
import sys

from scripts.cpu.lookahead.staging import study_r02_reference as reference
from scripts.cpu.lookahead.staging import study_r02_suffix_keys as suffix


ROOT = reference.ROOT
EVIDENCE = ROOT / 'docs/evidence'
STAGING = 'scripts.cpu.lookahead.staging.'


def require_match(label, actual, expected):
    if actual != expected:
        raise ValueError(f'R02 archive mismatch: {label}')
    print(f'Matched {label} ({len(actual)} records)', flush=True)


def main():
    for module, file in (
        ('study_lookahead_dynamic_release', 'dynamic-lookahead-dynamic-release-7884.json'),
        ('study_lookahead_transfer_order', 'dynamic-lookahead-transfer-order-7820.json'),
        ('study_r02_release_tie', 'r02-cpu-release-tie.json'),
        ('study_r02_shared_retry', 'r02-cpu-shared-retry.json'),
        ('study_r02_checkpoint_retry', 'r02-cpu-checkpoint-retry.json'),
        ('study_r02_exchange', 'r02-cpu-exchange.json'),
        ('study_r02_visibility', 'r02-cpu-visibility.json'),
        ('study_r02_guarded', 'r02-cpu-guarded.json'),
    ):
        subprocess.run([sys.executable, '-m', STAGING + module, '--check', str(EVIDENCE / file)],
                       cwd=ROOT, check=True)
    inputs = reference.read(EVIDENCE / 'r02-cpu-inputs.json')
    require_match('frozen input cases', reference.freeze()['cases'], inputs['cases'])
    for split in ('design', 'holdout'):
        require_match(f'{split} fixed plans', reference.evaluate(inputs, split),
                      reference.read(EVIDENCE / f'r02-cpu-{split}-reference.json'))
        require_match(f'{split} suffix keys', suffix.evaluate(inputs, split),
                      reference.read(EVIDENCE / f'r02-cpu-{split}-suffix.json'))
    subprocess.run([sys.executable, 'scripts/check_fixture.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()



