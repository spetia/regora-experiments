"""E9 analysis: the label-free rule with 300-step LR probes and no capacity stage. Plan and decision rules: PLAN.md.

Usage: python -m studies.e9_longer_probes.analyze [selftest | results/e9/FILE.json ...]   (default: the test files)
"""

import glob
import json
import statistics
import sys
from collections import Counter

from studies.e8_label_free.analyze import NOISE_AT, T_CRIT, TRUTH

# Fixed in PLAN.md before the test run.
LR_AT = 0.4  # |half-LR minus double-LR| after 300 steps, in units of the run's progress over steps 200..300
LR_CHUNK, FOLLOW_CHUNK = (
    2,
    8,
)  # indices into the 100-step readings: 300 steps for the rule, 900 for the advice
LR_STAGE_COST = 1800  # 3 replicates x 2 branches x 300 steps
ADVICE = {"lr_low": "double_lr", "lr_high": "halve_lr"}
MILD = tuple(TRUTH)
OUT_OF_SCOPE = "capacity"


def floor(value, case):
    return max(abs(value), 0.01 * case["metric"][2])


def same_sign(lean):
    return len({x > 0 for x in lean}) == 1


def t_resolved(lean):
    """E8's criterion, reported for comparison."""
    return (
        abs(statistics.mean(lean)) > T_CRIT * statistics.stdev(lean) / len(lean) ** 0.5
    )


def diagnose(case, lr_chunk=LR_CHUNK, agree=same_sign):
    """Returns (diagnosis, cost in baseline training steps). Stops at the first stage that fires."""
    scale = floor(case["metric"][2] - case["metric"][1], case)
    cost = len(case["noise_reads"]) * case["eval_batches"] / 3
    if 2 * statistics.stdev(case["noise_reads"]) > NOISE_AT * scale:
        return "eval_noise", cost

    cost += LR_STAGE_COST * (lr_chunk + 1) / (LR_CHUNK + 1)
    # positive: lowering the LR hurts and raising it helps, so the LR is too low. The control branch cancels out.
    lean = [
        h[lr_chunk] - d[lr_chunk] for h, d in zip(case["halve_lr"], case["double_lr"])
    ]
    mean = statistics.mean(lean)
    if agree(lean) and abs(mean) / scale > LR_AT:
        return ("lr_low" if mean > 0 else "lr_high"), cost
    return "healthy", cost


def advice_helped(case, diagnosis):
    """Did the LR change the diagnosis implies beat the control branch 900 steps later? None if no LR advice."""
    if diagnosis not in ADVICE:
        return None
    return case[ADVICE[diagnosis]][0][FOLLOW_CHUNK] < case["control"][0][FOLLOW_CHUNK]


def wider_lead(case):
    """How far ahead a 2x wider model is at step 100, in units of the run's progress over steps 100..300."""
    at_100, _, at_300 = case["metric"]
    return (case["wider"]["1.0"][0] - at_100) / floor(at_100 - at_300, case)


def truth(case):
    return TRUTH.get(case["fault"], case["fault"])


def right(cases, **kwargs):
    return sum(truth(c) == diagnose(c, **kwargs)[0] for c in cases)


def supported(ok):
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def report(cases):
    core = [c for c in cases if c["fault"] not in (*MILD, OUT_OF_SCOPE)]
    mild = [c for c in cases if c["fault"] in MILD]
    faults = sorted({c["fault"] for c in core})
    worst = 1.0
    for model in sorted({c["model"] for c in cases}):
        subset = [c for c in core if c["model"] == model]
        confusion = Counter((truth(c), diagnose(c)[0]) for c in subset)
        cost = statistics.mean(diagnose(c)[1] for c in subset)
        worst = min(worst, right(subset) / len(subset))
        print(f"{model}: {right(subset)} of {len(subset)} (mean cost {cost:.0f})")
        print(" " * 11 + " ".join(f"{f[:9]:>9s}" for f in faults))
        for t in faults:
            print(f"{t:10s} " + " ".join(f"{confusion[t, pick]:9d}" for pick in faults))
        print()
    for fault in MILD:
        subset = [c for c in mild if c["fault"] == fault]
        picks = dict(Counter(diagnose(c)[0] for c in subset))
        by_length = [right(subset, lr_chunk=k) for k in range(LR_CHUNK + 1)]
        print(
            f"{fault}: {picks}; right at 100/200/300-step probes: {by_length} of {len(subset)}"
        )
    healthy = [c for c in core if c["fault"] == "healthy"]
    alarms = sum(diagnose(c)[0] in ADVICE for c in healthy)
    outcomes = [advice_helped(c, diagnose(c)[0]) for c in cases]
    helped, advised = outcomes.count(True), len(outcomes) - outcomes.count(None)

    h1 = right(mild) / len(mild) >= 0.7 and alarms / len(healthy) <= 0.1
    print(
        f"\nH1 mild LR faults (>= 0.70 right, <= 0.10 LR false alarms on healthy): "
        f"{right(mild)} of {len(mild)}, {alarms} of {len(healthy)} false alarms: {supported(h1)}"
    )
    h2 = right(core) / len(core) >= 0.85 and worst >= 0.75
    print(
        f"H2 four conditions (>= 0.85 pooled, >= 0.75 per model): "
        f"{right(core)} of {len(core)}, worst model {worst:.0%}: {supported(h2)}"
    )
    h3 = advised > 0 and helped / advised >= 0.8
    print(
        f"H3 LR advice lowers loss 900 steps later (>= 0.80): {helped} of {advised}: {supported(h3)}"
    )

    print("\nReported, not decisive")
    both = core + mild
    print(
        f"E8's t-test in place of sign agreement: {right(both, agree=t_resolved)} of {len(both)} (rule: {right(both)})"
    )
    narrow = [c for c in cases if c["fault"] == OUT_OF_SCOPE]
    print(f"capacity cases read as: {dict(Counter(diagnose(c)[0] for c in narrow))}")
    for name, subset in (("healthy", healthy), ("capacity", narrow)):
        leads = sorted(wider_lead(c) for c in subset)
        print(
            f"wider-model lead at step 100, {name}: min {leads[0]:+.2f} median {statistics.median(leads):+.2f} max {leads[-1]:+.2f}"
        )


def selftest():
    def branch(*ends):
        return [[2.8, 2.8, end] + [end] * 6 for end in ends]

    base = dict(
        fault="healthy",
        metric=[3.2, 3.0, 2.9],
        eval_batches=20,
        noise_reads=[2.9] * 4,
        control=branch(2.8, 2.8, 2.8),
        halve_lr=branch(2.801, 2.799, 2.8),
        double_lr=branch(2.8, 2.801, 2.799),
        wider={"1.0": [3.1]},
    )
    assert diagnose(base) == ("healthy", 80 / 3 + LR_STAGE_COST)
    assert abs(wider_lead(base) + 1 / 3) < 1e-9
    low = base | dict(
        halve_lr=branch(2.85, 2.86, 2.84), double_lr=branch(2.75, 2.76, 2.74)
    )
    assert diagnose(low)[0] == "lr_low" and advice_helped(low, "lr_low")
    high = base | dict(
        halve_lr=branch(2.75, 2.76, 2.74), double_lr=branch(2.85, 2.86, 2.84)
    )
    assert diagnose(high)[0] == "lr_high" and advice_helped(high, "lr_high")
    assert (
        advice_helped(high, "lr_low") is False
        and advice_helped(base, "healthy") is None
    )
    # replicates that disagree in direction never fire, however large their mean
    split = base | dict(
        halve_lr=branch(3.2, 3.2, 2.79), double_lr=branch(2.8, 2.8, 2.8)
    )
    assert diagnose(split)[0] == "healthy"
    # a consistent but uneven effect fires under sign agreement and not under E8's t-test
    uneven = base | dict(
        halve_lr=branch(2.81, 2.9, 3.4), double_lr=branch(2.8, 2.8, 2.8)
    )
    assert (
        diagnose(uneven)[0] == "lr_low"
        and diagnose(uneven, agree=t_resolved)[0] == "healthy"
    )
    # the effect only shows at 300 steps: a 100-step reading of the same branches sees nothing
    assert diagnose(low, lr_chunk=0)[0] == "healthy"
    noisy = diagnose(base | dict(noise_reads=[2.8, 3.0, 2.9, 3.1], eval_batches=1))
    assert noisy[0] == "eval_noise" and noisy[1] < 10
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        paths = sys.argv[1:] or [
            f"results/e9/e9_{model}.json" for model in ("squat", "broad", "tall")
        ]
        report([case for path in paths for case in json.load(open(path))["cases"]])
