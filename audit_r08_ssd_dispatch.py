"""Fixed-boundary SSD dispatch counterfactual for the B481 tail witness."""

from fractions import Fraction
import gzip
import inspect
from itertools import product
import json
from textwrap import dedent

from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run
from restore_edge_select.simulator import Experiment


def diagnostic_executor():
    """Compile an isolated copy of the frozen executor with a dispatch gate."""
    source = dedent(inspect.getsource(Experiment._run))
    signature = "pause_at_head=None, early_portfolio=False):"
    dispatch = "            for i, obj, mode in order:\n"
    assert source.count(signature) == source.count(dispatch) == 1
    event_loop = "    while head < n:\n"
    event_min = "end = min(t for t in (running_until,"
    assert source.count(event_loop) == source.count(event_min) == 1
    source = source.replace(signature, "pause_at_head=None, early_portfolio=False, ssd_gate=None):")
    source = source.replace(event_loop, event_loop + "        dispatch_wakeup = None\n")
    source = source.replace(event_min, event_min + " dispatch_wakeup,")
    source = source.replace(dispatch, dispatch + """
                if mode == "prefetch" and i in ssd_gate and self.objects[obj]["tier"] == "SSD":
                    gate_head, delay = ssd_gate[i]
                    if head < gate_head:
                        continue
                    if head == gate_head and now < head_since + delay:
                        deadline = head_since + delay
                        dispatch_wakeup = deadline if dispatch_wakeup is None else min(dispatch_wakeup, deadline)
                        continue
""")
    namespace = dict(Experiment._run.__globals__)
    exec(compile(source, "<r08-ssd-dispatch-diagnostic>", "exec"), namespace)
    return namespace["_run"]


def audit(exp, archived):
    base = run(exp, "local-cost")
    frozen = run(exp, "entry-replay")
    plan = frozen["boundaries_tokens"]
    assert plan == archived["plan"]
    assert base["response_mean_ms"] == Fraction(archived["base_exact"])
    assert frozen["response_mean_ms"] == Fraction(archived["main_exact"])
    executor = diagnostic_executor()

    def execute(gate):
        return executor(exp, "fixed", 0, plan, 10000, len(plan), ssd_gate=gate)

    control = execute({})
    for field in ("completion_ms", "transfers", "prefetch_reserved_at_ms", "boundaries_tokens"):
        assert frozen[field] == control[field], field

    def evaluate(gate):
        result = execute(gate)
        assert result["boundaries_tokens"] == plan
        losses = [a - b for a, b in zip(result["completion_ms"], base["completion_ms"], strict=True)]
        return dict(gate=gate, mean_gain_ms=float(base["response_mean_ms"] - result["response_mean_ms"]),
                    worst_loss_ms=float(max(losses)), late_indices=[i for i, loss in enumerate(losses) if loss > 10],
                    target_losses_ms=[float(losses[i]) for i in (15, 16)])

    divergences = [i for i, (a, b) in enumerate(zip(base["boundaries_tokens"], plan, strict=True)) if a != b]
    # Delay dispatch by queue head, using the frozen executor window.
    choices = [(0, *range(max(1, i - exp.window + 1), i + 1)) for i in divergences]
    trials = [evaluate({i: (h, 0) for i, h in zip(divergences, heads) if h})
              for heads in product(*choices)]
    # Sweep wall-clock offsets measured from predecessor 16 reaching head.
    # This is a witness-specific intervention, using a fixed grid.
    trials.extend(evaluate({17: (16, delay)}) for delay in range(25, 1251, 25))
    coarse_pass = [t for t in trials if not t["late_indices"] and set(t["gate"]) == {17}
                   and t["gate"][17][1] > 0]
    if coarse_pass:
        best_delay = max(coarse_pass, key=lambda t: t["mean_gain_ms"])["gate"][17][1]
        trials.extend(evaluate({17: (16, delay)}) for delay in range(best_delay - 24, best_delay + 25)
                      if delay % 25)
    frontier = []
    for row in sorted(trials, key=lambda t: (t["worst_loss_ms"], -t["mean_gain_ms"])):
        if not frontier or row["mean_gain_ms"] > frontier[-1]["mean_gain_ms"]:
            frontier.append(row)
    return dict(divergences=divergences, frontier=frontier,
                best_within_10ms=max((t for t in trials if not t["late_indices"]),
                                    key=lambda t: t["mean_gain_ms"], default=None),
                trials=trials)


def main():
    cfg = read(CONFIG)
    cfg["holdout"] = {"qwen_traceB_blksz_16.jsonl": [481]}
    match = dict(model="27b", rate=16, section="fixed", level="512")
    with gzip.open(ROOT / "docs/evidence/r08-holdout.jsonl.gz", "rt", encoding="utf-8") as source:
        archived, = (r for r in map(json.loads, source) if r["key"]["window"] == 481
                     and r["key"].items() >= match.items())
    key, data, model, scale = next(case for case in cases(cfg, "holdout") if case[0] == archived["key"])
    exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
    print(json.dumps(audit(exp, archived), indent=2))


if __name__ == "__main__":
    main()
