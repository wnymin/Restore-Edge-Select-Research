"""Bounded entry refinement for stationary, single-lane exact recovery."""
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import prod


@dataclass(frozen=True)
class Refinement:
    short: bool = True
    wide: bool = True
    joint: bool = True
    guard: bool = True
    follower_limit: int = 24
    joint_followers: int = 2
    replay_limit: int = 40
    gain_per_request_ms: int = 1
    new_tail_ms: int = 10

    def __post_init__(self):
        if (not 4 <= self.follower_limit <= 32 or not 0 <= self.joint_followers <= 2
                or not 1 <= self.replay_limit <= 64 or self.gain_per_request_ms < 0 or self.new_tail_ms < 0):
            raise ValueError('Invalid refinement budget')


class EntryRefiner:
    """Only returns a boundary for the current request; continuation is a forecast."""

    def __init__(self, experiment, config, audit=None):
        self.exp = experiment
        self.cfg = config
        self.audit = audit
        self.stats = Counter()
        self.trials = Counter()
        # Per-object data is already indexed in the simulator. No offline
        # decisions, future timestamps, or workload labels are provided.

    def __call__(self, i, head, arrived, frozen, coarse, running, charged, ready, serial, replay):
        cfg, exp = self.cfg, self.exp
        self.stats['eligible'] += 1
        if not running or head >= i:
            self.stats['no-active-predecessor'] += 1
            return frozen
        choices = exp.eligible[i]
        cold = [set(exp.candidates[i][c]) - exp.initial for c in choices if c]
        if not any(ds - ready and exp.size(ds) <= exp.budget for ds in cold):
            self.stats['no-cold-fitting-action'] += 1
            return frozen
        available = min(arrived - i - 1, cfg.follower_limit if cfg.wide else 4)
        if not available or not ((cfg.short and coarse) or (cfg.wide and available > 4) or cfg.joint):
            self.stats['no-supported-extension'] += 1
            return frozen
        n = i + 1 + available
        # A competing request must have a legal staged alternative on the
        # shared lane. Its serial choice can itself be zero.
        branches = []
        for j in range(i+1,n):
            if any(c and exp.size(set(exp.candidates[j][c]) - exp.initial) <= exp.budget
                   and not ready.issuperset(exp.candidates[j][c]) for c in exp.eligible[j]):
                branches.append(j)
                if len(branches) == max(1,cfg.joint_followers):
                    break
        if not branches and not charged:
            self.stats['no-visible-competition'] += 1
            return frozen
        self.stats['triggered'] += 1
        memo = {}

        def evaluate(tail):
            if tail not in memo:
                memo[tail] = tuple(replay(tail))
            return memo[tail]

        def plans(later, flex=()):
            options = [choices, *(exp.eligible[j] if j in flex else (c,)
                                  for j,c in enumerate(later,i+1))]
            return list(product(*options)) if prod(map(len,options)) <= cfg.replay_limit else []

        later = tuple(serial(j) for j in range(i+1,n))
        stages = []
        if cfg.short and coarse:
            stages.append(('short',plans(later[:4])))
        if cfg.wide and len(later) > 4:
            stages.append(('wide',plans(later)))
        if cfg.joint and branches and cfg.joint_followers:
            stages.append(('joint',plans(later,tuple(branches))))
        proposed = frozen
        outcome = 'no-predicted-gain'
        # Every stage requires all its plans or does not start. Touching the
        # cap never silently promotes a partially evaluated challenger.
        for label, group in stages:
            if not group or len(set(group) - memo.keys()) + len(memo) > cfg.replay_limit:
                self.stats['capped-stage'] += 1
                self.trials[len(memo)] += 1
                return frozen
            best_by_action = {}
            for plan in group:
                ends = evaluate(plan)
                rank = (sum(ends), plan)
                current = best_by_action.get(plan[0])
                if current is None or rank < current[0]:
                    best_by_action[plan[0]] = (rank, ends)
            champ = min(choices,key=lambda c:(best_by_action[c][0][0], c != frozen, -c))
            frozen_rank, _ = best_by_action[frozen]
            champ_rank, champ_ends = best_by_action[champ]
            gain, champ_plan = frozen_rank[0]-champ_rank[0], champ_rank[1]
            self.stats[f'evaluated-{label}'] += 1
            if champ != frozen and gain > cfg.gain_per_request_ms * len(champ_ends):
                # A local prediction, not a guarantee for the full trajectory.
                # Veto a material loss that persists with both the proposed
                # continuation and the ordinary serial continuation.
                local = (serial(i), *later[:len(champ_plan)-1])
                matched = (frozen, *champ_plan[1:])
                serial_new = (champ, *local[1:])
                serial_old = (frozen, *local[1:])
                if cfg.guard and len({local, matched, serial_new, serial_old} - memo.keys()) + len(memo) > cfg.replay_limit:
                    outcome = 'guard-budget'
                elif cfg.guard:
                    blocked = [head + k for k,(new,old,base,snew,sold) in enumerate(
                        zip(champ_ends,evaluate(matched),evaluate(local),evaluate(serial_new),
                            evaluate(serial_old),strict=True))
                        if min(new-max(old,base), snew-max(sold,base)) > cfg.new_tail_ms]
                    if blocked:
                        outcome = 'tail-guard'
                    else:
                        proposed, outcome = champ, 'changed'
                    if self.audit is not None:
                        self.audit.append(dict(index=i, outcome=outcome, blocked=blocked, head=head,
                                               predicted_gain=float(gain), replay_count=len(memo),
                                               continuation_end=i+len(champ_plan)-1))
                else:
                    proposed, outcome = champ, 'changed'
                if proposed != frozen:
                    self.stats[f'accepted-{label}'] += 1
                    break
        self.stats[outcome] += 1
        self.trials[len(memo)] += 1
        return proposed
