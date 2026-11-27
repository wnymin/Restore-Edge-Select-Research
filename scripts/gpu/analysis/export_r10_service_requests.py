"""Export and audit all registered service requests from the closed R10 archive."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.analysis.audit_r10_service_pipe import distribution
from scripts.gpu.experiments.r10_holdout_lock import POLICIES
from scripts.gpu.experiments.r10_service_lock import load
from scripts.validation.check_r10_native_catalog import digest


FIELDS = ('case', 'block', 'family', 'profile', 'registered_anchor', 'policy', 'request', 'arm_failed', 'status',
          'finish_reason', 'selected_boundary', 'output_frames', 'visible_ttft_ms', 'request_frame_tpot_ms',
          'max_inter_token_frame_ms', 'client_end_to_end_ms', 'native_prefill_to_first_token_ms', 'slo_pass')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def request_rows(record, failed, expected, ttft, tpot):
    if failed:
        partial = record.get('partial_sse')
        needs_none = dict(status='unobserved', started_ms=None, accepted_ms=None, events=[])
        clients = partial['requests'] if partial is not None else [dict(needs_none, request=i) for i in range(expected)]
        native, chosen = {}, {}
    else:
        service = record['service']
        clients = [dict(c, status='terminal', started_ms=start, accepted_ms=c['server_accept_ms']) for start, c in
                   zip(record['client_arrivals_ms'], service['http_sse_requests'], strict=True)]
        native = {r['request']: r for r in service['requests']}
        chosen = {r['snapshot']['request']: r['action'] for r in record['decisions']}
        need(list(native) == list(range(expected)) and len(chosen) == len(record['decisions']) == expected,
             'missing or duplicated native request IDs')
    need([r['request'] for r in clients] == list(range(expected)), 'client requests differ from the registered population')

    rows, intervals, signatures = [], [], []
    for i, client in enumerate(clients):
        status, events = client['status'], client['events']
        terminal = status == 'terminal'
        kinds = [e['event'] for e in events]
        count = len(events) - terminal
        need(count >= 0 and kinds == ['token'] * count + (['end'] if terminal else []), f'invalid SSE frame order for request {i}')
        when = [Fraction(e['client_received_ms']) for e in events]
        sent = client.get('started_ms')
        accepted = client.get('accepted_ms')
        start = Fraction(sent) if sent is not None else None
        accept = Fraction(accepted) if accepted is not None else None
        need(not events or start is not None and accept is not None and [start, accept, *when] == sorted([start, accept, *when]),
             f'invalid transport clocks for request {i}')
        visible = next((t for t, event in zip(when, events, strict=True) if event['text']), None)
        delay = visible - start if visible is not None else None
        # End can flush visible text, but it does not participate in ITL or TPOT.
        gaps = [float(b - a) for a, b in zip(when[:count], when[1:count])]
        per_token = (when[count - 1] - when[0]) / (count - 1) if count >= 2 else None
        compare_delay, compare_tpot = ((delay, per_token) if failed else
                                      tuple(float(x) if x is not None else None for x in (delay, per_token)))
        hit = terminal and compare_delay is not None and compare_delay <= ttft and (compare_tpot is None or compare_tpot <= tpot)
        reason = events[-1]['reason'] if terminal else None
        tokens = tuple(e['id'] for e in events[:count])
        prefill_gap = None
        if not failed:
            n = native[i]
            need(tokens == tuple(e['id'] for e in n['tokens']) and reason == n['finish']['reason'], 'client differs from the native output')
            if count:
                prefill_gap = (n['tokens'][0]['native_us'] - n['prefill_done']['native_us']) / 1000
            signatures.append((reason, tokens))
        intervals.extend(gaps)
        rows.append(dict(request=i, status=status, finish_reason=reason, output_frames=count,
                         selected_boundary=chosen[i] if not failed else None, visible_ttft_ms=float(delay) if delay is not None else None,
                         request_frame_tpot_ms=float(per_token) if per_token is not None else None,
                         max_inter_token_frame_ms=max(gaps) if gaps else None, client_end_to_end_ms=float(when[-1] - start) if terminal else None,
                         native_prefill_to_first_token_ms=prefill_gap, slo_pass=int(hit)))
    return rows, intervals, signatures


def check_arm(arm, rows, intervals):
    def dist(field):
        return distribution([row[field] for row in rows if row[field] is not None])

    client = arm['client']
    eligible = sum(row['slo_pass'] for row in rows)
    status = Counter(row['status'] for row in rows)
    if arm['failed']:
        need(dict(status) == client['statuses'] and len(rows) == client['expected'] and eligible == client['slo_success_count'],
             'failed arm status or SLO denominator disagrees with the frozen summary')
    else:
        need((status['terminal'], eligible) == (client['completed'], client['slo']['success_count']) and
             (dist('visible_ttft_ms'), dist('request_frame_tpot_ms'), distribution(intervals), dist('client_end_to_end_ms'),
              dist('native_prefill_to_first_token_ms')) == (client['ttft_visible_ms'], client['token_frame_tpot_ms'],
                  client['inter_token_frame_ms'], client['client_end_to_end_ms'], arm['worker']['native_prefill_done_to_first_emission']),
             'request timing or SLO denominator disagrees with the frozen summary')


def collect(lock, summary, archive):
    blocks = {r['id']: r for r in lock['blocks']}
    target = lock['experimental_slo']
    complete, all_rows = [], []
    for case, result in zip(lock['cases'], summary['pairs'], strict=True):
        ident = case['id']
        ledger = json.loads(archive.read(f'{ident}/pair.json'))
        need(ledger['service_lock_sha256'] == summary['lock_sha256'] and result['family'] == blocks[case['block']]['family'] and
             all(ledger[k] == result[k] == v for k, v in case.items()),
             'service case differs from the registered order')
        per_arm, outputs = {}, {}
        for i, (policy, arm) in enumerate(zip(case['arm_order'], ledger['arms'], strict=True), 1):
            prefix = f'{i}-{policy}'
            official = result['arms'][policy]
            failed = bool(arm['return_code'])
            need((arm['prefix'], arm['order'], arm['policy'], official['failed']) == (prefix, i, policy, failed), 'arm identity changed')
            suffix = '.failure.json' if failed else '.run.json'
            record = {}
            if not failed or suffix in arm['files']:
                payload = archive.read(f'{ident}/{prefix}{suffix}')
                need(sha256(payload).hexdigest() == arm['files'][suffix], 'arm receipt hash changed')
                record = json.loads(payload)
            rows, intervals, signatures = request_rows(record, failed, blocks[case['block']]['requests'],
                                                        target['visible_ttft_ms'], target['client_request_tpot_ms'])
            check_arm(official, rows, intervals)
            if not failed:
                per_arm[policy], outputs[policy] = rows, signatures
            m = dict(case=ident, block=case['block'], family=result['family'], profile=case['profile'],
                     registered_anchor=case['anchor'], policy=policy, arm_failed=int(failed))
            all_rows.extend(dict(m, **row) for row in rows)
        if set(per_arm) == set(POLICIES):
            a, b = ([r['visible_ttft_ms'] for r in per_arm[p]] for p in POLICIES)
            diff = [y - x for x, y in zip(a, b, strict=True) if x is not None and y is not None]
            changes = sum(x != y for x, y in zip(*(outputs[p] for p in POLICIES), strict=True))
            comparison = result['paired']
            need((comparison['paired_visible_ttft_ms'], comparison['max_individual_visible_ttft_regression_ms'],
                  comparison['different_greedy_outputs'], comparison['valid_performance_comparison']) == (
                      distribution(diff), max(diff, default=None), changes, changes == 0), 'paired tail or output equality changed')
            complete.append(ident)
        else:
            need(result['paired'] is None, 'aborted arm was included in paired latency')
    return all_rows, complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'summary', 'archive', 'delivery', 'csv', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    if a.csv.exists() or a.out.exists() or a.csv.resolve() == a.out.resolve():
        raise FileExistsError('choose distinct new request detail and audit paths')
    lock, summary, delivery = load(a.lock), read(a.summary), read(a.delivery)
    identity = digest(a.lock), digest(a.summary), digest(a.archive)
    need((delivery['lock_sha256'], delivery['summary_sha256'], delivery['archive_sha256']) == identity and
         (summary['lock_sha256'], summary['archive_sha256'], delivery['completed_pairs']) == (identity[0], identity[2], len(lock['cases'])),
         'requires the complete service archive, summary, and delivery audit')
    with ZipFile(a.archive) as z:
        rows, complete = collect(lock, summary, z)
    with a.csv.open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    by_policy = {p: [row for row in rows if row['policy'] == p] for p in POLICIES}
    result = dict(scope='request-level HTTP/SSE evidence; no prompt or output content; failed arms retain the full registered population',
                  lock_sha256=identity[0], summary_sha256=identity[1], archive_sha256=identity[2], delivery_sha256=digest(a.delivery),
                  request_csv_sha256=digest(a.csv), requests=len(rows), terminal_statuses=dict(Counter(row['status'] for row in rows)),
                  by_policy={p: dict(requests=len(rr), statuses=dict(Counter(r['status'] for r in rr)), slo_pass=sum(r['slo_pass'] for r in rr))
                             for p, rr in by_policy.items()},
                  completed_pairs_with_paired_tail_audit=complete, slo_pass_count=sum(row['slo_pass'] for row in rows))
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
