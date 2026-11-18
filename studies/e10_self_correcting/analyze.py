"""E10 analysis: a self-correcting run against schedules picked by successive halving. Plan and rules: PLAN.md.

Usage: python -m studies.e10_self_correcting.analyze [selftest | results/e10/FILE.json ...]   (default: the test files)
"""

import json
import statistics
import sys
from collections import Counter

from scipy import stats

from studies.e10_self_correcting.run import (
    DEFAULT_START,
    GRID,
    MARGIN,
    SEGMENT,
    STARTS,
    STEPS,
)

# Successive halving over the grid: (step at which runs are ranked, how many survive). It spends
# 6 x 600 + 3 x 600 + 2 x 600 + 1 x 600 = 7200 steps, the same as the controller's 3 x 2400.
RUNGS = ((600, 3), (1200, 2), (1800, 1))


def halving(runs):
    """The run successive halving ends up with, ranking survivors by validation loss at each rung."""
    alive = list(runs)
    for step, keep in RUNGS:
        alive = sorted(alive, key=lambda run: run["val"][step // SEGMENT - 1])[:keep]
    return alive[0]


def interval(differences):
    """Mean and 95% t-interval half-width of paired differences."""
    n = len(differences)
    half = stats.t.ppf(0.975, n - 1) * statistics.stdev(differences) / n**0.5
    return statistics.mean(differences), half


def ours(case, start, margin):
    return next(
        run
        for run in case["controlled"]
        if run["start"] == start and run["margin"] == margin
    )


def compare(cases, margin, comparator, start=DEFAULT_START):
    """Controller minus comparator on test loss, paired by seed. Negative means the controller is better."""
    theirs = [comparator(case) for case in cases]
    mean, half = interval(
        [ours(case, start, margin)["test"] - loss for case, loss in zip(cases, theirs)]
    )
    return dict(mean=mean, half=half, noise=statistics.stdev(theirs))


def verdicts(cases, margin):
    """Per model: is the controller no worse (H1) and better (H2) than each halving baseline, and is it robust (H3)?"""
    out = {}
    for model in sorted({c["model"] for c in cases}):
        subset = [c for c in cases if c["model"] == model]
        against = {
            kind: compare(
                subset, margin, lambda case, kind=kind: halving(case[kind])["test"]
            )
            for kind in ("constant", "cosine")
        }
        default = lambda case: ours(case, DEFAULT_START, margin)["test"]
        extremes = {
            start: compare(subset, margin, default, start)
            for start in (STARTS[0], STARTS[-1])
        }
        out[model] = dict(
            against=against,
            extremes=extremes,
            no_worse=all(c["mean"] + c["half"] < c["noise"] for c in against.values()),
            better=all(c["mean"] + c["half"] < 0 for c in against.values()),
            robust=all(c["mean"] + c["half"] < c["noise"] for c in extremes.values()),
        )
    return out


def supported(ok):
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def report(cases, margin=MARGIN):
    result = verdicts(cases, margin)
    for model, v in result.items():
        subset = [c for c in cases if c["model"] == model]
        print(
            f"{model} ({len(subset)} seeds), margin {margin}; test loss, controller minus baseline (negative is better)"
        )
        for kind, c in v["against"].items():
            print(
                f"  vs halving over {kind:8s} {c['mean']:+.4f} +/- {c['half']:.4f}   (baseline seed SD {c['noise']:.4f})"
            )
        for start, c in v["extremes"].items():
            print(
                f"  start {start:<7g} vs default start {c['mean']:+.4f} +/- {c['half']:.4f}"
            )
        for kind in ("constant", "cosine"):
            means = [
                statistics.mean(case[kind][i]["test"] for case in subset)
                for i in range(len(GRID))
            ]
            best = means.index(min(means))
            c = compare(subset, margin, lambda case: case[kind][best]["test"])
            picks = Counter(halving(case[kind])["lr"] for case in subset)
            print(
                f"  vs best {kind} in hindsight (lr {GRID[best]:g}, 6x compute) {c['mean']:+.4f} +/- {c['half']:.4f}; halving picked {dict(picks)}"
            )
        runs = [ours(case, DEFAULT_START, margin) for case in subset]
        short = sum(
            run["steps"] < STEPS
            for case in subset
            for run in case["controlled"]
            if run["margin"] == margin
        )
        print(
            f"  final LR from default start: {sorted(run['trace'][-1]['lr'] * (run['trace'][-1]['kept'] or 1) for run in runs)}"
        )
        print(
            f"  median LR by segment: {[statistics.median(run['trace'][i]['lr'] for run in runs if i < len(run['trace'])) for i in range(STEPS // SEGMENT)]}"
        )
        print(f"  runs that ended short of {STEPS} steps (any start): {short}\n")
    models = result.values()
    print(
        f"H1 no worse than halving, on every model: {supported(all(v['no_worse'] for v in models))}"
    )
    print(
        f"H2 better than halving, on at least 2 of 3 models: {supported(sum(v['better'] for v in models) >= 2)}"
    )
    print(
        f"H3 the starting LR does not matter, on every model: {supported(all(v['robust'] for v in models))}"
    )


def selftest():
    def run(lr, val, test):
        return dict(lr=lr, val=val, test=test)

    # a slow starter: best at the end, dropped at the first rung
    slow = run(1, [9, 9, 9, 9, 9, 9, 9, 0.5], 0.5)
    steady = run(2, [3, 2, 2, 2, 2, 2, 2, 1], 1.0)
    others = [run(k, [4 + k] * 8, 4.0 + k) for k in range(3, 7)]
    assert halving([slow, steady, *others])["lr"] == 2

    def case(seed, controlled_test):
        flat = [run(lr, [2.0] * 8, 2.0 + 0.01 * seed) for lr in GRID]
        runs = [
            dict(
                start=start,
                margin=MARGIN,
                test=controlled_test + 0.01 * seed,
                steps=STEPS,
                trace=[dict(lr=start, kept=1.0)] * 8,
            )
            for start in STARTS
        ]
        return dict(model="m", seed=seed, constant=flat, cosine=flat, controlled=runs)

    better = verdicts(
        [case(seed, 1.9 + 0.001 * (seed % 3)) for seed in range(8)], MARGIN
    )["m"]
    assert better["no_worse"] and better["better"] and better["robust"]
    worse = verdicts(
        [case(seed, 2.5 + 0.001 * (seed % 3)) for seed in range(8)], MARGIN
    )["m"]
    assert not worse["no_worse"] and not worse["better"]
    mean, half = interval([1.0, 2.0, 3.0])
    assert mean == 2.0 and abs(half - 4.303 / 3**0.5) < 1e-3
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        arguments = sys.argv[1:]
        margin = (
            float(arguments.pop(0).split("=")[1])
            if arguments and arguments[0].startswith("margin=")
            else MARGIN
        )
        paths = arguments or [
            f"results/e10/e10_{model}.json" for model in ("slim", "stout", "lanky")
        ]
        report(
            [case for path in paths for case in json.load(open(path))["cases"]], margin
        )
