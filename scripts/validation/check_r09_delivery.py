"""Verify the R09 artifacts against the pre-run input and timing locks."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from scripts.cpu.replay.study_r09_budget import (
    EVIDENCE, capacity, check_inputs, verify,
)
from scripts.validation.archive_source import source_matches
from scripts.validation.check_r09_gpu_observability import inspect
from scripts.validation.check_r09_quality import main as check_quality
from scripts.validation.check_r09_slack import main as check_slack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=capacity.ROOT/'outputs')
    p.add_argument('--deep-slack',action='store_true')
    a=p.parse_args()
    check_slack()
    check_quality()
    inputs=EVIDENCE/'r09-inputs.json'
    lock=EVIDENCE/'r09-budget-lock.json'
    manifest=capacity.read(inputs)
    commit=capacity.read(lock)
    check_inputs(manifest)
    assert commit['inputs_sha256']==capacity.file_hash(inputs)
    assert commit['calibration_sha256']==capacity.file_hash(EVIDENCE/'r09-slack-calibration.jsonl.gz')
    budget=capacity.read(EVIDENCE/'r09-budget-results.json')
    assert budget['budget_lock_sha256']==capacity.file_hash(lock)
    assert budget['sample']['cases']==len(manifest['holdout'])
    rawslack=EVIDENCE/'r09-slack-holdout.jsonl.gz'
    assert budget['holdout_sha256']==capacity.file_hash(rawslack)
    if a.deep_slack:
        assert verify(manifest,rawslack,SimpleNamespace(inputs=inputs,lock=lock))==budget
    else:
        for path, pin in commit['pins'].items():
            assert source_matches(capacity.ROOT/path, pin, historical=True), path
    for split,stem in [('calibration','calibration'),('holdout','results')]:
        report=capacity.read(EVIDENCE/f'r09-quality-{stem}.json')
        assert report['lock_sha256']==capacity.file_hash(lock)
        assert len(report['cases'])==len(manifest[split])==report['total']['configurations']
        assert report['total']['exact_oracle']<=len(report['cases'])
        raw=a.raw_dir/f'r09-quality-{split}.jsonl.gz'
        if raw.exists():
            assert report['raw_sha256']==capacity.file_hash(raw)
        else:
            print(f'Per-request {split} quality archive not present at {raw}; summary verified only')
    observed=capacity.read(EVIDENCE/'r09-gpu-observability.json')
    assert source_matches(EVIDENCE/'r07-live-scheduling-summary.json', observed['summary_sha256'])
    recomputed=inspect()
    recomputed['summary_sha256']=observed['summary_sha256']
    assert observed==recomputed and not observed['budget_calibration_qualified']
    critical=capacity.read(EVIDENCE/'r09-critical-points.json')
    assert critical['holdout_sha256']==capacity.file_hash(rawslack)
    assert not critical['independent_source_claim'] and len(critical['samples'])>3
    sensitivity=capacity.read(EVIDENCE/'r09-anchor-sensitivity.json')
    assert sensitivity['lock_sha256']==capacity.file_hash(lock)
    assert sensitivity['anchor_rounds']==6 and sensitivity['calibration_configs']==24
    print('R09: input and code seals, fixed-action replay, budget/quality outputs and GPU scope verified')


if __name__=='__main__':
    main()
