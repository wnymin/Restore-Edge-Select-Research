"""Score candidate boundaries from decision-time queue state."""

def predict(name, row):
    compute = row["serial_compute_ms"] + row["serial_copy_ms"]
    io = row["serial_io_ms"]
    slack = row["head_slack_ms"]
    if name == "serial" or (name == "slack_capacity" and not row["prefetch_fits"]):
        return compute + io
    if name in {"slack", "slack_capacity"}:
        return compute + max(0, io - slack)
    if not row["missing_bytes"]:
        return compute
    io = row["incremental_ram_ms"] + row["incremental_ssd_ms"]
    inflight = row["inflight_ms"]
    if name == "fifo_snapshot":
        slack += row["committed_ahead_compute_ms"]
    if row["prefetch_fits"]:
        return compute + max(0, io + inflight + row["queue_debt_ms"] - slack)
    return compute + io + max(0, inflight - slack)
