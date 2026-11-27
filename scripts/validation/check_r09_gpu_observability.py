"""R09-GPU-01: what the existing paired live probe can and cannot time."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from scripts.validation.check_r07_live_scheduling import analyze_observer, get_raw
from scripts.validation.run_r08_sealed_holdout import capacity, EVIDENCE


def inspect():
    source=EVIDENCE/'r07-live-scheduling-summary.json'
    archive=EVIDENCE/'r07-live-observer.jsonl.gz'
    data=capacity.read(source)
    checked={}
    observation=analyze_observer(get_raw('observer',False,checked))
    assert data['observer']==observation
    with gzip.open(archive,'rt',encoding='utf-8') as f:
        header,*rows=map(json.loads,f)
    inventory=set().union(*(r.keys() for r in rows))
    spans={}
    for d in (512,1024):
        blocks=observation['blocks']
        poll=[b['arms'][f'probe-{d}']['poll_ms'] for b in blocks]
        shift=[b['arms'][f'probe-{d}']['target']-b['arms'][f'clean-{d}']['target'] for b in blocks]
        spans[str(d)]=dict(blocks=len(blocks),poll_total_ms_median=median(poll),
                           probe_minus_clean_http_target_ms_median=median(shift),
                           target_shift_range_ms=[min(shift),max(shift)])
    q=data['queue_interface']
    assert all(not row for row in q['queue_exposed_keys'])
    return dict(archive_sha256=capacity.file_hash(archive),summary_sha256=capacity.file_hash(source),
                archived_fields=sorted(inventory),paired_poll=spans,
                request_queue_identity_visible=False,gpu_busy_boundary_visible=False,
                exact_candidate_and_object_transfer_issue_visible=False,
                decision_ready_and_publish_on_common_execution_clock_visible=False,
                budget_calibration_qualified=False,
                scope='Recompute paired HTTP observer evidence; no new GPU policy or trace. CPU clocks are not GPU kernel/transfer event markers.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    val=inspect()
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(val,f,indent=2)
        f.write('\n')


if __name__=='__main__':
    main()
