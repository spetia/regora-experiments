"""E12 analysis: is a difference between two fine-tunes real? Decision rules are in PLAN.md.

Usage: python -m studies.e12_paired_eval.analyze [selftest | results.json ...]
"""

import itertools
import json
import statistics
import sys

from studies.e9_longer_probes.analyze import supported
from studies.e12_paired_eval.run import ARMS, TASKS

MIN_ITEM_SAVING, MIN_FALSE_ALARMS, MIN_SEED_SAVING = 2.0, 0.10, 2.0
Z, T19 = (
    1.96,
    2.093,
)  # T19: two-sided 95% t for the 20 seeds of the full run; other counts print no interval


def bits(run):
    return [int(c) for c in run["correct"]]


def accuracy(run):
    return statistics.mean(bits(run))


def item_stats(a, b):
    """For two runs on the same items: (variance of a, of b, of the per-item difference, mean difference)."""
    d = [x - y for x, y in zip(a, b)]
    return (
        statistics.variance(a),
        statistics.variance(b),
        statistics.variance(d),
        statistics.mean(d),
    )


def item_saving(pairs):
    """How many times fewer items a paired comparison needs than one on separate item samples, pooled over pairs."""
    stats = [item_stats(a, b) for a, b in pairs]
    return sum(va + vb for va, vb, _, _ in stats) / sum(vd for _, _, vd, _ in stats)


def significant(a, b):
    """Does the paired 95% interval over items exclude zero? This is the interval an evaluation tool would print."""
    _, _, vd, mean = item_stats(a, b)
    return abs(mean) > Z * (vd / len(a)) ** 0.5


def seed_saving(arm, control):
    """How many times fewer seeds a comparison on shared seeds needs than one on separate seeds."""
    paired = statistics.variance([a - c for a, c in zip(arm, control)])
    return (statistics.variance(arm) + statistics.variance(control)) / paired


def report(runs_by_task):
    h1 = h2 = h3 = True
    for task, runs in runs_by_task.items():
        seeds = sorted({r["seed"] for r in runs})
        get = {(r["arm"], r["seed"]): r for r in runs}
        correct = {key: bits(r) for key, r in get.items()}
        acc = {key: statistics.mean(v) for key, v in correct.items()}
        control = [acc["control", s] for s in seeds]
        n_items = len(correct["control", seeds[0]])
        print(f"\n{task}: {len(seeds)} seeds, {n_items} items")
        print(
            f"  control accuracy {statistics.mean(control):.4f}, SD across seeds {statistics.stdev(control):.4f}; "
            f"SE from {n_items} items {(statistics.mean(control) * (1 - statistics.mean(control)) / n_items) ** 0.5:.4f}"
        )

        twins = list(itertools.combinations(seeds, 2))
        alarms = sum(
            significant(correct["control", s], correct["control", t]) for s, t in twins
        )
        rate = alarms / len(twins)
        h2 &= rate >= MIN_FALSE_ALARMS
        twin_saving = item_saving(
            [(correct["control", s], correct["control", t]) for s, t in twins]
        )
        print(
            f"  same settings, different seed: item interval excludes zero in {alarms} of {len(twins)} pairs "
            f"({rate:.0%}; 5% expected if training noise did not matter); item pairing saves {twin_saving:.1f}x"
        )

        print(
            "  arm              items saved  seeds saved  mean diff  95% interval (seeds)   1-seed verdicts"
        )
        for arm in ARMS:
            if arm == "control":
                continue
            pairs = [(correct[arm, s], correct["control", s]) for s in seeds]
            items = item_saving(pairs)
            scores = [acc[arm, s] for s in seeds]
            seedx = seed_saving(scores, control)
            h1 &= items >= MIN_ITEM_SAVING
            h3 &= seedx >= MIN_SEED_SAVING
            diffs = [a - c for a, c in zip(scores, control)]
            mean = statistics.mean(diffs)
            half = (
                T19 * statistics.stdev(diffs) / len(diffs) ** 0.5
                if len(seeds) == 20
                else float("nan")
            )
            flagged = sum(significant(a, c) for a, c in pairs)
            agree = sum((d > 0) == (mean > 0) for d in diffs)
            print(
                f"  {arm:15s}  {items:9.1f}x  {seedx:10.1f}x  {mean:+.4f}    [{mean - half:+.4f}, {mean + half:+.4f}]"
                f"     {flagged} of {len(seeds)} significant, {agree} match the sign of the mean"
            )

    print(
        f"\nH1 scoring both models on the same items saves at least {MIN_ITEM_SAVING:g}x items for every change "
        f"on every task: {supported(h1)}"
    )
    print(
        f"H2 two fine-tunes that differ only in seed look different to the item interval in at least "
        f"{MIN_FALSE_ALARMS:.0%} of pairs on every task: {supported(h2)}"
    )
    print(
        f"H3 sharing the training seed saves at least {MIN_SEED_SAVING:g}x seeds for every change on every task: "
        f"{supported(h3)}"
    )
    return h1, h2, h3


def selftest():
    import random

    rng = random.Random(0)
    n, seeds = 400, range(20)
    hard = [
        rng.random() for _ in range(n)
    ]  # item difficulty shared by every run: this is what item pairing uses
    runs = []
    for seed in seeds:
        luck = rng.gauss(
            0, 0.08
        )  # training noise shared by both arms on one seed: what seed pairing uses
        for arm, lift in (("control", 0.0), ("half_lr", 0.05)):
            # an item is right when it is easy enough for this run, with 5% of answers flipped
            answer = lambda h: (h < 0.63 + luck + lift) != (rng.random() < 0.05)
            correct = "".join(str(int(answer(h))) for h in hard)
            runs.append(dict(arm=arm, seed=seed, correct=correct))
    others = [
        dict(r, arm=arm)
        for r in runs
        if r["arm"] == "half_lr"
        for arm in ARMS
        if arm not in ("control", "half_lr")
    ]
    assert report({"synthetic": runs + others}) == (True, True, True)
    a, b = [1, 1, 0, 0], [1, 0, 0, 0]
    assert item_stats(a, b)[3] == 0.25 and not significant(a, a)
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        paths = sys.argv[1:] or [f"results/e12/e12_{task}.json" for task in TASKS]
        loaded = [json.load(open(path)) for path in paths]
        report({d["config"]["task"]: d["runs"] for d in loaded})
