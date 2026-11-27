"""R09 fixed-action release-delay intervention on the existing serial executor."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

from scripts.cpu.replay.study_r07_arrivals import run
from scripts.validation.run_r08_sealed_holdout import workloads, capacity

BASE = 'entry-replay'
GRID = tuple(map(F, ('0', '.05', '.1', '.25', '.5', '1', '2', '4', '8', '16', '32')))


def measured(state):
    return F(state['measured_ns'], 1_000_000)


def starts(exp, result):
    return [a + w + d for a,w,d in zip(exp.arrivals,result['queue_wait_ms'],result['head_to_compute_ms'],strict=True)]


def fixed(exp, plan, durations, extra):
    """Replay the measured baseline cost plus a controlled additional delay."""
    def release(state):
        i = state['index']
        return durations[i] + extra.get(i, 0)
    return exp.run('fixed', boundaries=plan, decision_timing=release)


def effects(exp, base, trial):
    """Exact matched-event plus compute/service differences (not just the mean)."""
    def events(run):
        # Do not silently drop repeated attempts, or equate demand with prefetch.
        out = defaultdict(list)
        for t in run['transfers']:
            key = tuple(t[k] for k in ('request','object','mode','path'))
            out[key].append(t['start_ms'])
        return out

    def missing_or_later(old, new):
        return sum(b is None or b>a for a,b in zip(old,new) if a is not None) + max(0,len(old)-len(new))

    old, new = events(base), events(trial)
    reserve = missing_or_later(base['prefetch_reserved_at_ms'],trial['prefetch_reserved_at_ms'])
    issue = sum(missing_or_later(v,new[k]) for k,v in old.items())
    starts_late = [j-i for i,j in zip(starts(exp,base),starts(exp,trial),strict=True)]
    ends_late = [j-i for i,j in zip(base['completion_ms'],trial['completion_ms'],strict=True)]
    extra_peak = trial['peak_prefetch_bytes'] - base['peak_prefetch_bytes']
    no_perf_loss = max(starts_late) <= 0 and max(ends_late) <= 0 and extra_peak <= 0
    return dict(events_safe=no_perf_loss and not (reserve or issue), perf_safe=no_perf_loss,
                reservation_late=reserve, transfer_late=issue, compute_late=sum(t>0 for t in starts_late),
                completion_late=sum(t>0 for t in ends_late), peak_delta=extra_peak,
                worst_start_ms=str(max(0,*starts_late)), worst_completion_ms=str(max(0,*ends_late)),
                mean_delta_ms=str(trial['response_mean_ms']-base['response_mean_ms']))


def anchor(exp, sampler=None):
    baseline = (sampler or (lambda: run(exp,BASE,extra={'profile_decisions':True,'decision_timing':measured})))()
    states = baseline['decision_timing']
    assert [s['index'] for s in states] == list(range(len(exp.requests)))
    plan = baseline['boundaries_tokens']
    durations = [measured(s) for s in states]
    match = fixed(exp,plan,durations,{})
    for k in ('boundaries_tokens','prefetch_reserved_at_ms','head_to_compute_ms','queue_wait_ms',
              'completion_ms','transfers','peak_prefetch_bytes','decision_intervals_ms'):
        assert baseline[k] == match[k], k
    return baseline, plan, durations


def stratum(state):
    gpu = 'busy' if state['running_until_ms'] is not None else 'idle'
    if not state['missing_objects']:
        return gpu+'/ready'
    if state['added_bytes'] > state['free_bytes']:
        return gpu+'/cold-capacity'
    if state['transfer_until_ms'] is not None:
        return gpu+'/cold-lane'
    return gpu+'/cold-issuable'


def sweep(exp, key, grid=GRID, *, all_states=True):
    baseline,plan,duration = anchor(exp)
    states = baseline['decision_timing']
    eligible = [s['index'] for s in states if s['eligible_count']>1]
    def one(indices, d):
        run = fixed(exp,plan,duration,dict.fromkeys(indices,d))
        return effects(exp,baseline,run)
    def scan(indices):
        rows = [dict(delay_ms=str(d),**one(indices,d)) for d in grid]
        assert rows[0]['perf_safe'] and rows[0]['events_safe']
        return rows
    busy = [s['index'] for s in states if s['eligible_count']>1 and s['running_until_ms'] is not None]
    return dict(key=key, plan=plan, sample_ns=[s['measured_ns'] for s in states],
                decisions=[dict(state=s,stratum=stratum(s),grid=scan([s['index']]))
                           for s in states if all_states or s['index'] in eligible],
                joint=scan(eligible), joint_busy=scan(busy), eligible=eligible, anchor_busy=busy)


def prefix_cap(rows, field):
    """Last tested passing point before first failure; no interpolation."""
    for i,row in enumerate(rows):
        if not row[field]:
            return F(rows[i-1]['delay_ms']), F(row['delay_ms'])
    return F(rows[-1]['delay_ms']), None



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('split',choices=('calibration','holdout'))
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    manifest = capacity.read(args.manifest)
    for tr in {a['trace'] for a in manifest[args.split]}:
        assert capacity.file_hash(capacity.TRACE_DIR/tr) == manifest['trace_sha256'][tr]
    cases = manifest[args.split]
    grid = tuple(F(x) for x in manifest['grid_ms'])
    assert grid == tuple(sorted(set(grid))) and grid[0] == 0
    with gzip.open(args.out,'xt',encoding='utf-8',newline='\n') as stream:
        for i,(key,exp) in enumerate(workloads(dict(cases=cases,strata=manifest['strata'])),1):
            row = sweep(exp,key,grid)
            stream.write(json.dumps(row,default=str,separators=(',',':'))+'\n')
            stream.flush()
            print(i,len(cases),key['trace'],key['window'],key['model'],key['stratum'],flush=True)
    assert i == len(cases)


if __name__ == '__main__':
    main()
