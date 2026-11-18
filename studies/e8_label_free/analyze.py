"""E8 analysis: a diagnosis rule that needs no labelled examples. The plan and decision rules are in PLAN.md.

Usage: python -m studies.e8_label_free.analyze [results/e8/e8.json | selftest]
"""

import json
import statistics
import sys
from collections import Counter

from studies.e7b_confirmation.analyze import empty_rows, score

# Thresholds, in units of the run's own progress over its last 100 steps. Fixed in PLAN.md before the test run.
LR_LOW_AT, LR_HIGH_AT, CAPACITY_AT, NOISE_AT = 0.1, 1.0, 0.5, 0.25
T_CRIT = 2.92  # one-sided 95% Student-t, 2 degrees of freedom (3 replicates)
TRUTH = {"mild_high": "lr_high", "mild_low": "lr_low"}
LR_STAGE_COST, WIDER_COST = (
    900,
    1600,
)  # 3 replicates x (control + halve + double) x 100 steps; 400 steps at 4x


def diagnose(case):
    """Returns (diagnosis, cost in baseline training steps). Stops at the first stage that fires."""
    scale = max(abs(case["metric_300"] - case["metric_200"]), 0.01 * case["metric_300"])
    cost = len(case["noise_reads"]) * case["eval_batches"] / 3
    if 2 * statistics.stdev(case["noise_reads"]) > NOISE_AT * scale:
        return "eval_noise", cost

    cost += LR_STAGE_COST
    # positive: lowering the LR hurts and raising it helps, so the LR is too low
    lean = [
        halve - double for halve, double in zip(case["halve_lr"], case["double_lr"])
    ]
    mean = statistics.mean(lean)
    standard_error = statistics.stdev(lean) / len(lean) ** 0.5
    resolved = abs(mean) > T_CRIT * standard_error
    if resolved and mean / scale > LR_LOW_AT:
        return "lr_low", cost
    if resolved and mean / scale < -LR_HIGH_AT:
        return "lr_high", cost

    cost += WIDER_COST
    return ("capacity" if case["wider"] / scale < -CAPACITY_AT else "healthy"), cost


def as_e7_case(case):
    """The same case in E7's single-replicate format, for the fitted-likelihood baseline."""
    probes = dict(
        halve_lr=case["halve_lr"][0],
        double_lr=case["double_lr"][0],
        reeval_4x=case["reeval_4x"],
        wider=case["wider"],
    )
    return case | dict(probes=probes)


def tally(cases):
    rows = [(TRUTH.get(c["fault"], c["fault"]), *diagnose(c)) for c in cases]
    right = sum(truth == pick for truth, pick, _ in rows)
    return (
        right,
        len(rows),
        statistics.mean(cost for *_, cost in rows),
        Counter((t, p) for t, p, _ in rows),
    )


def report(cases):
    core = [c for c in cases if c["fault"] not in TRUTH]
    mild = [c for c in cases if c["fault"] in TRUTH]
    small = json.load(open("results/e7/e7.json"))["cases"]
    faults = sorted({c["fault"] for c in core})
    pooled_right = pooled_fitted = 0
    worst = 1.0
    for model in sorted({c["model"] for c in core}):
        subset = [c for c in core if c["model"] == model]
        right, total, cost, confusion = tally(subset)
        fitted = empty_rows()
        score(small, [as_e7_case(c) for c in subset], fitted, passive=())
        fitted_right = sum(truth == pick for truth, pick, *_ in fitted["ALL"])
        pooled_right, pooled_fitted, worst = (
            pooled_right + right,
            pooled_fitted + fitted_right,
            min(worst, right / total),
        )
        print(
            f"{model}: label-free rule {right} of {total} (mean cost {cost:.0f}); fitted on E7 {fitted_right} of {total}"
        )
        print(" " * 11 + " ".join(f"{f[:9]:>9s}" for f in faults))
        for truth in faults:
            print(
                f"{truth:10s} "
                + " ".join(f"{confusion[truth, pick]:9d}" for pick in faults)
            )
        print()
    total = len(core)
    print(
        f"pooled: label-free rule {pooled_right} of {total}; fitted on E7 {pooled_fitted} of {total}"
    )
    if mild:
        right, count, _, confusion = tally(mild)
        print(
            f"mild LR faults (reported, not decisive): {right} of {count}; {dict(confusion)}"
        )
    h1 = pooled_right / total >= 0.8 and worst >= 0.7
    h2 = (pooled_right - pooled_fitted) / total >= 0.10
    print(
        f"\nH1 label-free diagnosis works (>= 0.80 pooled, >= 0.70 per model): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}"
    )
    print(
        f"H2 beats likelihoods fitted on another model by >= 0.10: {'SUPPORTED' if h2 else 'NOT SUPPORTED'}"
    )


def selftest():
    base = dict(
        metric_200=3.0,
        metric_300=2.9,
        eval_batches=20,
        noise_reads=[2.9] * 4,
        wider=0.0,
    )
    flat = dict(halve_lr=[0.001, -0.002, 0.001], double_lr=[0.0, 0.001, -0.001])
    assert diagnose(base | flat)[0] == "healthy"
    assert diagnose(base | flat | dict(wider=-0.1))[0] == "capacity"
    assert (
        diagnose(
            base | dict(halve_lr=[0.03, 0.04, 0.035], double_lr=[-0.03, -0.02, -0.03])
        )[0]
        == "lr_low"
    )
    assert (
        diagnose(
            base | dict(halve_lr=[-0.1, -0.08, -0.09], double_lr=[0.2, 0.15, 0.25])
        )[0]
        == "lr_high"
    )
    # a large but inconsistent lean is not resolved, so the LR stage must not fire
    assert (
        diagnose(base | dict(halve_lr=[0.2, -0.2, 0.1], double_lr=[-0.2, 0.3, 0.0]))[0]
        == "healthy"
    )
    noisy = diagnose(
        base | flat | dict(noise_reads=[2.8, 3.0, 2.9, 3.1], eval_batches=1)
    )
    assert noisy[0] == "eval_noise" and noisy[1] < 10
    print("selftest ok")


if __name__ == "__main__":
    argument = sys.argv[1] if len(sys.argv) > 1 else "results/e8/e8.json"
    selftest() if argument == "selftest" else report(json.load(open(argument))["cases"])
