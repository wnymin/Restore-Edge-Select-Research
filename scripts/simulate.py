"""Run a Phase 1 trace and print JSON results."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restore_edge_select.simulator import Experiment, PAIR_POLICIES
from restore_edge_select.adaptive import AdaptiveConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "fixtures/phase1-hand-trace.json")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/qwen3.5-4b.json")
    parser.add_argument("--recompute-config", type=Path,
                        help="override the trace's recompute cost model with a JSON object")
    parser.add_argument("--prefetch-mib", type=int, nargs="+")
    parser.add_argument("--policy", nargs="+",
                        choices=["recompute", "longest", "threshold", "local-cost", "capacity",
                                 "remaining-wait", "remaining-wait-live", "remaining-wait-fast", "arrived-lookahead",
                                 "adaptive", "offline",
                                 *sorted(PAIR_POLICIES)],
                        default=["local-cost", "capacity", "offline"])
    parser.add_argument("--threshold-tokens", type=int, default=576)
    parser.add_argument("--max-combinations", type=int, default=10000)
    parser.add_argument("--guard-ms", default="0", help="near-tie window for remaining-wait-live")
    parser.add_argument("--transfer-trigger-ms", default="0", help="minimum transfer signal spread for pair lookahead")
    parser.add_argument("--screen-ms", default="0", help="extra screening delay on the dispatcher")
    parser.add_argument("--pair-ms", default="0", help="extra delay per triggered pair forecast")
    parser.add_argument("--probe-ms", default="0", help="extra delay per I/O floor probe")
    parser.add_argument("--telemetry-lag-ms", default="0", help="lag in the observed I/O speed")
    parser.add_argument("--stable-ms", default="0", help="post-change holdoff for two-request-stable-io")
    parser.add_argument("--adaptive-later", type=int, default=4)
    parser.add_argument("--adaptive-light-replays", type=int, default=2)
    parser.add_argument("--adaptive-coupled-replays", type=int, default=3)
    parser.add_argument("--adaptive-loss-ms", type=Fraction, default=Fraction(10))
    parser.add_argument("--adaptive-mean-only", action="store_true", help="omit the per-request loss constraint")
    parser.add_argument("--adaptive-gain-ms", type=Fraction, default=Fraction(1),
                        help="minimum predicted gain per visible request")
    parser.add_argument("--adaptive-error-ms", type=Fraction, default=Fraction(0),
                        help="assumed absolute error bound per predicted completion")
    parser.add_argument("--adaptive-screen", action="store_true", help="use the optional heuristic skip")
    parser.add_argument("--adaptive-exhaustive", action="store_true",
                        help="allow replay of every eligible candidate; disable heuristic skips")
    parser.add_argument("--adaptive-audit", action="store_true", help="verify by exhaustive visible forecasts")
    parser.add_argument("--estimate", nargs="*", default=[], metavar="PATH=FACTOR",
                        help="duration multipliers for local-cost and remaining-wait-live")
    args = parser.parse_args()
    try:
        estimate = dict(item.split("=", 1) for item in args.estimate)
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        if args.recompute_config:
            trace["cost"]["recompute"] = json.loads(args.recompute_config.read_text(encoding="utf-8"))
        models = json.loads(args.model.read_text(encoding="utf-8"))["models"]
        model = next((m for m in models if m["model_id"] == trace["model"]
                      and f"/{trace['revision']}/" in m["config_url"]), None)
        if model is None:
            raise ValueError("No model matches the trace model ID and revision")
        results = []
        for budget in args.prefetch_mib or trace["memory"]["prefetch_mib_cases"]:
            exp = Experiment(trace, model, budget)
            exhaustive_limit = max(map(len, exp.eligible)) if args.adaptive_exhaustive else None
            for policy in args.policy:
                run = (exp.offline(args.max_combinations) if policy == "offline" else
                       exp.run(policy, threshold_tokens=args.threshold_tokens,
                               max_combinations=args.max_combinations, estimate=estimate,
                               guard_ms=args.guard_ms, transfer_trigger_ms=args.transfer_trigger_ms,
                               screen_ms=args.screen_ms, pair_ms=args.pair_ms, probe_ms=args.probe_ms,
                               telemetry_lag_ms=args.telemetry_lag_ms, stable_ms=args.stable_ms,
                               adaptive=AdaptiveConfig(
                                   later=args.adaptive_later, light_replays=exhaustive_limit or args.adaptive_light_replays,
                                   coupled_replays=exhaustive_limit or args.adaptive_coupled_replays,
                                   loss_ms=None if args.adaptive_mean_only else args.adaptive_loss_ms,
                                   gain_per_request_ms=args.adaptive_gain_ms, audit=args.adaptive_audit,
                                   screen=args.adaptive_screen and not args.adaptive_exhaustive,
                                   forecast_error_ms=args.adaptive_error_ms) if policy == "adaptive" else None))
                results.append({"prefetch_mib": budget, "policy": policy, **run})
        print(json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, Fraction) else x))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
