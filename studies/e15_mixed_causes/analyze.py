"""E15 analysis: what does the frozen diagnosis rule say about mixed and unnamed faults? Rules are in PLAN.md.

Usage: python -m studies.e15_mixed_causes.analyze [results.json ...]
"""

import collections
import json
import sys

from studies.e9_longer_probes.analyze import diagnose, supported

# The causes actually present in each condition, in the rule's vocabulary. A fault the rule has no name for
# contributes nothing, so for `heavy_decay` every named cause is a misdiagnosis.
PRESENT = {
    "healthy": set(),
    "low_and_noisy": {"lr_low", "eval_noise"},
    "high_and_noisy": {"lr_high", "eval_noise"},
    "heavy_decay": set(),
    "low_and_decay": {"lr_low"},
}
MIXED, UNNAMED = ("low_and_noisy", "high_and_noisy"), ("heavy_decay",)
MIN_MIXED, MAX_WRONG, MAX_FALSE_ALARMS = 0.80, 0.20, 0.10


def verdicts(cases):
    """condition -> Counter of the rule's answers."""
    out = collections.defaultdict(collections.Counter)
    for case in cases:
        out[case["fault"]][diagnose(case)[0]] += 1
    return out


def rate(counter, wanted):
    return sum(n for answer, n in counter.items() if wanted(answer)) / sum(
        counter.values()
    )


def report(cases_by_task):
    h1 = h2 = h3 = True
    for task, cases in cases_by_task.items():
        said = verdicts(cases)
        print(f"\n{task}")
        for fault in PRESENT:
            print(f"  {fault:15s} {dict(said[fault])}")
        mixed = sum((said[f] for f in MIXED), collections.Counter())
        present = set().union(*(PRESENT[f] for f in MIXED))
        # per condition, so a cause present in one mixed condition does not excuse naming it in the other
        named = sum(n for f in MIXED for a, n in said[f].items() if a in PRESENT[f])
        h1 &= named / sum(mixed.values()) >= MIN_MIXED
        wrong = rate(
            sum((said[f] for f in UNNAMED), collections.Counter()),
            lambda a: a != "healthy",
        )
        h2 &= wrong <= MAX_WRONG
        # as E11's H4: an alarm is "LR too low" or "noisy evaluation"; "LR too high" shows in the counts above
        alarms = rate(said["healthy"], lambda a: a in ("lr_low", "eval_noise"))
        h3 &= alarms <= MAX_FALSE_ALARMS
        both = said["low_and_decay"]
        print(
            f"  mixed: names a cause that is present in {named} of {sum(mixed.values())} (vocabulary {sorted(present)}); "
            f"unnamed fault: names some cause in {wrong:.0%}; healthy: false alarms {alarms:.0%}; "
            f"low LR with heavy decay: says lr_low in {both['lr_low']} of {sum(both.values())}"
        )
    print(
        f"\nH1 in mixed cases the rule names a cause that is present in at least {MIN_MIXED:.0%}, every task: {supported(h1)}"
    )
    print(
        f"H2 for a fault it has no name for, the rule names a cause in at most {MAX_WRONG:.0%}, every task: {supported(h2)}"
    )
    print(
        f"H3 healthy runs are told LR too low or noisy evaluation in at most {MAX_FALSE_ALARMS:.0%}, every task: {supported(h3)}"
    )
    return h1, h2, h3


if __name__ == "__main__":
    # the counting, on answers alone: 3 of 4 right is 0.75
    assert (
        rate(collections.Counter(healthy=1, lr_low=3), lambda a: a == "lr_low") == 0.75
    )
    paths = sys.argv[1:] or [
        f"results/e15/e15_{task}.json" for task in ("lm", "vision")
    ]
    report({(d := json.load(open(p)))["config"]["task"]: d["cases"] for p in paths})
