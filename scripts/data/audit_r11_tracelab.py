"""Measure tool-loop frequency and timing in the sanitized TraceLab v0.0.1 release."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import gzip
import json
import math
from pathlib import Path


SHA256 = '9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b'


def quantiles(values):
    values = sorted(values)
    return {f'p{p}': values[math.ceil(p * len(values) / 100) - 1]
            for p in (10, 50, 90, 95, 99)} if values else {}


def audit(path):
    with path.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != SHA256:
            raise ValueError('expected the pinned TraceLab v0.0.1 JSONL')
    counts = defaultdict(Counter)
    fields = defaultdict(set)
    samples = defaultdict(lambda: defaultdict(list))
    sessions = defaultdict(set)
    tool_sessions = defaultdict(set)
    previous = {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            provider = row['provider']
            key = (provider, row['user'], row['session_id'])
            counters = counts[provider]
            values = samples[provider]
            fields[provider].update(row)
            sessions[provider].add(key)
            is_return = row['current_tool_result_count'] > 0
            tools = row['tools']
            counters.update(rounds=1, tool_input=is_return, tool_output=bool(tools),
                            tool_only_input=is_return and not row['current_user_message_count'],
                            tool_records=len(tools))
            if is_return:
                tool_sessions[provider].add(key)
            input_tokens = row['input_tokens_total']
            if isinstance(input_tokens, int) and input_tokens > 0:
                values['input_tokens'].append(input_tokens)
            for tool in tools:
                latency = tool['tool_wall_latency_ms']
                if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
                    values['tool_wall_ms'].append(latency)
            returned_ids = {e.get('tool_call_id') for e in row['timing_events']
                            if e['event_type'] == 'tool_result'} - {None}
            prior = previous.get(key)
            if prior is not None and row['round_index'] == prior[0] + 1:
                counters['adjacent_rows'] += 1
                matched = returned_ids & prior[1]
                if matched:
                    counters['linked_tool_returns'] += 1
                    if type(prior[2]) is int and prior[2] >= 0:
                        values['linked_previous_output_tokens'].append(prior[2])
                        counters['linked_previous_output_le128'] += prior[2] <= 128
                    if isinstance(input_tokens, int) and isinstance(prior[3], int):
                        values['linked_input_token_delta'].append(input_tokens - prior[3])
            previous[key] = (row['round_index'], {t['tool_call_id'] for t in tools} - {None},
                             row['output_tokens'], input_tokens)
    return {'sha256': SHA256, 'providers': {
        p: {'counts': dict(counts[p], sessions=len(sessions[p]), sessions_with_tool_input=len(tool_sessions[p])),
            'fields': sorted(fields[p]),
            'distributions': {k: {'n': len(v), **quantiles(v)} for k, v in samples[p].items()}}
        for p in sorted(counts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.path), indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
