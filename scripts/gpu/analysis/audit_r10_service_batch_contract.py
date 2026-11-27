"""Record source-level evidence for the boundary between R10 single-slot and upstream batching."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def source(path, needles):
    raw = path.read_bytes()
    lines = raw.decode('utf8').splitlines()
    found = {}
    for label, needle in needles.items():
        matches = [i for i, line in enumerate(lines, 1) if needle in line]
        if not matches:
            raise ValueError(f'{path.name}: source marker missing for {label}')
        found[label] = dict(line=matches[0], text=lines[matches[0] - 1].strip())
    return dict(sha256=sha256(raw).hexdigest(), basename=path.name, matches=found)


def audit(repo, llama):
    worker = repo / 'scripts/gpu/experiments/r10_catalog_worker.cpp'
    fifo = repo / 'src/restore_edge_select/live_snapshot.py'
    engine = llama / 'tools/server'
    server = engine / 'server-context.cpp'
    stock = source(server, dict(slot_selection='server_slot * get_available_slot(',
                                mixing_gate='if (params_base.cont_batching || batch.size() == 0)',
                                batching='ret = llama_decode(ctx_tgt, batch_view)',
                                prefix='n_past = slot.prompt.tokens.get_common_prefix(input_tokens)',
                                compatibility='slot_batched && !slot_batched->can_batch_with(slot)'))
    backend = source(worker, dict(context_slots='cp.n_seq_max = 1;',
                                  native_restore='ledger_.recurrent_must_preserve(',
                                  current_target='Target current_;', active_decode='generation token failed',
                                  prefill='native prompt chunk failed'))
    snapshot = source(fifo, dict(active_dependencies='active_deps: tuple',
                                 transfer_object='transfer_object: str | None',
                                 serial_projection='Invalid active head or active dependency identity'))
    files = sorted(engine.glob('server-*.cpp')) + sorted(engine.glob('server-*.h'))
    if not files or server not in files:
        raise ValueError('expected the pinned upstream server implementation')
    markers = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    hooks = [p.name for p in files if b'llama_r10_' in p.read_bytes() or b'ggml_backend_vk_r10_' in p.read_bytes()]
    return dict(scope='source inspection only; no runtime throughput evidence',
                stock_server=stock, r10_worker=backend, r10_snapshot=snapshot,
                upstream_checked_unit_sha256=markers, upstream_units_exposing_r10_hooks=hooks,
                qualification=dict(native_single_slot_confirmed=True,
                                   stock_continuous_batch_path_found=True,
                                   existing_switch_is_policy_compatible=False if not hooks else None,
                                   reason='stock scheduling/prompt-cache path has no R10 object-lease and choice handoff in checked units'
                                   if not hooks else 'found a potential hook; inspect before qualification'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('repo', 'llama', 'out'):
        ap.add_argument('--' + arg, type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.repo, args.llama)
    with args.out.open('x', encoding='utf8') as f:
        print(json.dumps(result, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
