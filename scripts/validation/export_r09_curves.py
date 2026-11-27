"""Export R09 figure-ready data for the timing, quality and budget curves."""
import argparse
import csv
from fractions import Fraction as F
from pathlib import Path

from scripts.cpu.replay.study_r09_budget import capacity


def csv_out(target, records):
    rows=list(records)
    with target.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def quality_points(cal, hold):
    for doc in (cal,hold):
        for grouping,group in [('all',doc['total']),*doc['groups'].items()]:
            for a,v in group['snapshot'].items():
                yield dict(split=doc['split'],group=grouping,arm=a,configurations=group['configurations'],
                           source_windows=group['source_windows'],median_added_search_ms=v['median_total_extra_ns']/1e6,
                           p95_added_search_ms=v['p95_total_extra_ns']/1e6,changed_configs=v['changed_cases'],
                           improved_mean_configs=v['zero_mean_improves'],degraded_mean_configs=v['zero_mean_degrades'],
                           best_observed_mean_gain_ms=v['top_gain_ms'])


def response_points(cal, hold):
    for doc in (cal,hold):
        for grouping,group in [('all',doc['total']),*doc['groups'].items()]:
            for a,v in group['timed'].items():
                routes=v['routes']
                budget=0 if a=='entry-replay' else float(F(a[:-2])) if a.endswith('ms') else None
                yield dict(split=doc['split'],group=grouping,arm=a,nominal_extra_budget_ms=budget,
                           configurations=group['configurations'],source_windows=group['source_windows'],
                           median_paired_mean_delta_ms=v['median_case_mean'],median_paired_p95_delta_ms=v['median_case_p95'],
                           worst_paired_mean_delta_ms=v['worst_case_mean'],worst_paired_p95_delta_ms=v['worst_case_p95'],
                           config_median_decision_p95_ms=v['median_case_decision_p95_ms'],
                           source_worst_delta_mean_min_ms=v['source_worst_mean_observed_range_ms'][0],
                           source_worst_delta_mean_max_ms=v['source_worst_mean_observed_range_ms'][1],
                           mean_improvements_gt_1ms=v['mean_faster_over_1ms'],
                           known_good_mean_failures=v['case_failures'].get('mean',0),
                           known_good_p95_failures=v['case_failures'].get('p95',0),
                           known_good_retention_failures=v['case_failures'].get('retention',0),
                           new_repeat_tail_configs=v['case_failures'].get('new_repeat_tail',0),
                           busy_eligible=routes.get('budget_eligible'),soft_overrun=routes.get('budget_overrun'))


def slack_points(doc):
    for group in doc['descriptive']['joint']:
        for v in group['grid']:
            yield dict(kind='per-config-with-anchor-busy-assignment',model=group['model'],condition=group['condition'],
                       extra_delay_ms=float(F(v['delay_ms'])),denominator=v['total'],
                       performance_failures_all=v['all_bad'],performance_failures_busy=v['busy_bad'],
                       event_failures_all=v['all_event_bad'],event_failures_busy=v['busy_event_bad'])
    for cap,probe in doc['primary'].items():
        v=probe['windows']
        yield dict(kind='per-source-window-with-dynamic-busy-gate',model='4b-native',condition='pooled-within-window',
                   extra_delay_ms=float(F(cap)),denominator=len(v),performance_failures_all=None,
                   performance_failures_busy=sum(x['failed'] for x in v),event_failures_all=None,
                   event_failures_busy=sum(x['events_failed'] for x in v))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('calibration','holdout','budget','outdir'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    c,h,d=(capacity.read(p) for p in (a.calibration,a.holdout,a.budget))
    assert c['lock_sha256']==h['lock_sha256']==d['budget_lock_sha256']
    csv_out(a.outdir/'r09-search-time-quality.csv',quality_points(c,h))
    csv_out(a.outdir/'r09-budget-net-response.csv',response_points(c,h))
    csv_out(a.outdir/'r09-fixed-action-budget.csv',slack_points(d))


if __name__=='__main__':
    main()
