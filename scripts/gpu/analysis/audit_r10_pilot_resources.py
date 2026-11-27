"""Audit allocation tiers or a monitored calibration block's WDDM, Vulkan, and host memory."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints, resource_rows
from scripts.validation.check_r10_native_catalog import digest


def arm(prefix, schedule=None):
    run_path, raw, stderr = endpoints(prefix)
    run = json.loads(run_path.read_bytes())
    assert run['allocation_only'] == (schedule is None) and run['resources'] and run['warmup'] and not run['qualification']
    if schedule is None:
        assert raw.stat().st_size == 0
        audit = None
    else:
        assert run['warmup_grid'] and run['policy'] == 'local-cost' and run['native_events']
        assert not run.get('holdout_lock_sha256') and not run.get('ssd_hold_ms') and run['publish_rest_on'] == 'open-loop'
        audit = check(run_path, raw, schedule)
    samples, host = defaultdict(lambda: defaultdict(dict)), defaultdict(dict)
    with prefix.with_suffix('.wddm.csv').open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            ts, pid, src, key, value = row['host_utc'], int(row['pid']), row['gpu_instance'], row['counter'], int(row['bytes'])
            assert value >= 0
            if src == 'system':
                assert pid == 0 and key not in host[ts]
                host[ts][key] = value
            else:
                assert pid == run['worker_pid'] and src.startswith(f'pid_{pid}_') and key not in samples[ts][src]
                samples[ts][src][key] = value
    fields = {'dedicated_usage', 'shared_usage', 'total_committed'}
    gpu = {ts: {key: sum(c[key] for c in instances.values()) for key in fields}
           for ts, instances in samples.items() if all(c.keys() == fields for c in instances.values())}
    host_fields = {'available_bytes', 'cache_bytes', 'committed_bytes'}
    system = {ts: x for ts, x in host.items() if x.keys() == host_fields}
    res = resource_rows(stderr)
    assert ('allocation-idle' if schedule is None else 'after-drain') in res
    assert run['monitor_exit_code'] == 0 and len(gpu) >= 2 and len(system) >= 2
    result = dict(run=run_path.name, block=run['block'], capacity_bytes=run['capacity_bytes'], resident_ram_bytes=run['resident_ram_bytes'],
                  pinned_bytes=run['pinned_bytes'], native_resources=res, gpu_samples=len(gpu), host_samples=len(system),
                  wddm_sampled_max_bytes={k: max(x[k] for x in gpu.values()) for k in fields},
                  system_sampled_range_bytes={k: [min(x[k] for x in system.values()), max(x[k] for x in system.values())] for k in host_fields},
                  seal_sha256=run['seal_sha256'], worker_sha256=run['worker_sha256'])
    if audit is not None:
        result.update(event_audit=audit, schedule_sha256=digest(schedule))
        if run.get('arrival_epoch_utc'):
            begin, end = (datetime.fromisoformat(run[key]) for key in ('arrival_epoch_utc', 'drain_observed_utc'))
            assert begin < end
            active_gpu, active_system = ([x for t, x in kind.items() if begin <= datetime.fromisoformat(t) <= end]
                                         for kind in (gpu, system))
            assert len(active_gpu) >= 2 and len(active_system) >= 2
            result.update(active_gpu_samples=len(active_gpu), active_system_samples=len(active_system),
                          active_wddm_sampled_max_bytes={k: max(x[k] for x in active_gpu) for k in fields},
                          active_system_sampled_range_bytes={k: [min(x[k] for x in active_system), max(x[k] for x in active_system)]
                                                             for k in host_fields})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, action='append', required=True)
    ap.add_argument('--busy-schedule', type=Path, help='audit one monitored arm or a three-capacity sweep of the same block')
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new evidence filenames')
    if a.busy_schedule and len(a.prefix) not in (1, 3):
        ap.error('supply one prefix or all three registered capacity tiers')
    arms = sorted((arm(x, a.busy_schedule) for x in a.prefix), key=lambda row: row['capacity_bytes'])
    if len(a.prefix) == 3 or not a.busy_schedule:
        assert [r['capacity_bytes'] for r in arms] == [67108864, 86245376, 150994944]
        for k in ('worker_sha256', 'seal_sha256', 'resident_ram_bytes', 'pinned_bytes'):
            assert len({r[k] for r in arms}) == 1
        if a.busy_schedule:
            assert all(r['active_gpu_samples'] >= 2 and r['active_system_samples'] >= 2 for r in arms)
    with ZipFile(a.archive, 'x') as z:
        for prefix in a.prefix:
            for path in (*endpoints(prefix), prefix.with_suffix('.wddm.csv'), prefix.with_suffix('.wddm.stderr')):
                z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    with a.out.open('x', encoding='utf8', newline='\n') as out:
        scope = 'allocation idle only'
        if a.busy_schedule:
            scope = 'three monitored tiers of one calibration block' if len(arms) == 3 else 'one monitored calibration block through drain'
        print(json.dumps(dict(scope=scope + '; sampled WDDM/host peaks are lower bounds on continuous peaks',
                              archive_sha256=digest(a.archive), arms=arms), indent=2, sort_keys=True), file=out)
    print(f'Audited {len(arms)} monitored calibration arm(s).' if a.busy_schedule else 'Audited three local allocated capacity tiers.')


if __name__ == '__main__':
    main()
