"""Run and summarize paired resident Qwen continuations at Weka-derived positions."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
from hashlib import file_digest
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import monotonic

from scripts.data.study_r11_weka import read
from scripts.gpu.experiments.run_r10_prefill_grid import gpu_environment


SHAPES = {
    'short': [(19264, 256, 11100, 20480), (40896, 512, 11101, 43008), (55808, 256, 11102, 57344)],
    'trajectory': [(19264, 4800, 11103, 24576), (56768, 6656, 11104, 65536)],
}
VK = {'GGML_VK_VISIBLE_DEVICES': '1', 'GGML_VK_ASYNC_USE_TRANSFER_QUEUE': '1',
      'GGML_VK_R10_CONCURRENT_BUFFERS': '1'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def run(probe, model, runtime, out, shape, repeats):
    deep, tail, seed, context = shape
    stem = out / f'resident-{deep}-{tail}'
    env = os.environ | VK | {'PATH': str(runtime) + os.pathsep + os.environ['PATH']}
    env.pop('VK_INSTANCE_LAYERS', None)
    env.pop('VK_ADD_LAYER_PATH', None)
    argv = [str(probe), str(model), str(runtime), 'Vulkan0', *map(str, (deep, tail, repeats, seed, context))]
    before = gpu_environment()
    start = monotonic()
    with stem.with_suffix('.csv').open('wb') as stdout, stem.with_suffix('.stderr').open('wb') as stderr:
        process = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW, timeout=480)
    elapsed = monotonic() - start
    info = {'boundary': deep, 'tail': tail, 'seed': seed, 'context': context, 'reps': repeats,
            'exit_code': process.returncode, 'elapsed_s': elapsed, 'gpu_env': VK,
            'gpu_before': before, 'gpu_after': gpu_environment(), 'probe_sha256': digest(probe),
            'stdout_sha256': digest(stem.with_suffix('.csv')), 'stderr_sha256': digest(stem.with_suffix('.stderr'))}
    stem.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8', newline='\n')
    process.check_returncode()
    print(deep, tail, round(elapsed, 1), flush=True)


def summarize(out, shape):
    deep, tail, seed, context = shape
    stem = out / f'resident-{deep}-{tail}'
    info = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
    if (info['exit_code'] or (info['boundary'], info['tail'], info['seed'], info['context']) != shape
            or digest(stem.with_suffix('.csv')) != info['stdout_sha256']):
        raise ValueError(f'incompatible resident-pair runs: {stem}')
    with stem.with_suffix('.csv').open(newline='', encoding='utf-8') as stream:
        rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    grouped = defaultdict(dict)
    for row in rows:
        group = grouped[row['rep']]
        if row['boundary'] in group:
            raise ValueError('duplicate pair member')
        group[row['boundary']] = row
    depths = (deep - 128, deep - 64, deep)
    if set(grouped) != set(range(info['reps'])) or any(set(group) != set(depths) for group in grouped.values()):
        raise ValueError(f'incomplete random-order pairs: {stem}')
    deltas = {deep - d: [(group[d]['prefill_us'] - group[deep]['prefill_us']) / 1000
                        for group in grouped.values()] for d in depths[:-1]}
    greedy_pairs_matched = sum(len({r['greedy'] for r in group.values()}) == 1 for group in grouped.values())
    return {'receipt': info, 'prefill_median_ms_by_depth': {d: median(g[d]['prefill_us'] for g in grouped.values()) / 1000
                                                          for d in depths},
            'paired_extra_prefill_ms_by_gap': {gap: {'min': min(ds), 'median': median(ds), 'max': max(ds)}
                                               for gap, ds in deltas.items()},
            'greedy_pairs_matched': greedy_pairs_matched, 'samples': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('raw', type=Path)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--group', choices=SHAPES)
    parser.add_argument('--repeats', type=int, default=6)
    args = parser.parse_args()
    lock = read(Path('configs/r10-local-holdout-cost-v1.json'))
    model_digest = digest(args.model)
    if model_digest != lock['model_sha256']:
        raise ValueError('Qwen model differs from the R10 calibration')
    if args.group:
        for shape in SHAPES[args.group]:
            run(args.probe.resolve(), args.model.resolve(), args.runtime.resolve(), args.raw, shape, args.repeats)
    shapes = {name: [summarize(args.raw, shape) for shape in selected
                     if (args.raw / f'resident-{shape[0]}-{shape[1]}.json').is_file()]
              for name, selected in SHAPES.items()}
    cost = lock['cost']
    state_bytes = read(Path('configs/qwen3.5-4b.json'))['models'][0]['payload']['checkpoint_bytes']
    staged = cost['ssd_to_device']
    ssd_ms = Fraction(staged['startup_ms']) + state_bytes / Fraction(staged['bytes_per_ms'])
    result = {'scope': 'resident continuation, synthetic token IDs; Weka context lengths only',
              'model_sha256': model_digest, 'llama_dll_sha256': digest(args.runtime / 'libllama.dll'),
              'state_bytes': state_bytes, 'R10_reference_SSD_state_ms': float(ssd_ms), 'groups': shapes}
    args.summary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
