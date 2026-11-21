"""R06 gate using decision-time costs and the original entry-replay fallback."""


def allow_serial(rows, base, budget, margin):
    """Certify that entry-replay's screening would keep the serial choice.

    Decline on shared work or a resource boundary. In the remaining states
    the projected scores below reduce to the original entry screening scores.
    """
    if any(c and (r['ready_bytes'] or r['shared_reservation_bytes']
                  or r['active_new_kv_bytes'] or r['queue_debt_ms'] or r['inflight_ms']
                  or (r['head_distance'] and not r['prefetch_fits']
                      and r['extra_reserve_bytes'] <= budget)) for c, r in rows.items()):
        return False
    base_row = rows[base]
    coverage = base_row['head_slack_ms'] + base_row['committed_ahead_compute_ms']
    base_compute = base_row['serial_compute_ms'] + base_row['serial_copy_ms']
    base_io = base_row['incremental_ram_ms'] + base_row['incremental_ssd_ms']
    base_exposed = max(0, base_io - coverage) if base_row['prefetch_fits'] else base_io
    for c, r in rows.items():
        if c == base:
            continue
        compute = r['serial_compute_ms'] + r['serial_copy_ms']
        transfer = r['incremental_ram_ms'] + r['incremental_ssd_ms']
        exposed = max(0, transfer - coverage) if r['prefetch_fits'] else transfer
        savings = base_compute - compute
        if exposed - base_exposed - savings <= margin * abs(savings):
            return False
    return True
