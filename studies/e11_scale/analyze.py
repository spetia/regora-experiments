"""E11 analysis. Plan and decision rules: PLAN.md. The diagnosis rule is E9's, imported and not changed.

Usage: python -m studies.e11_scale.analyze [selftest | results/e11/FILE.json ...]   (default: the test files)
"""

import json
import math
import statistics
import sys
from collections import Counter

from studies.e9_longer_probes.analyze import (
    ADVICE,
    LR_CHUNK,
    advice_helped,
    diagnose,
    supported,
    truth,
)

TASKS = ("lm", "vision")
INTERVENTIONS = ("halve_lr", "double_lr", "weight_decay")
IN_SCOPE = ("lr_low", "mild_low", "eval_noise")  # the diagnoses E9 supported
TRUSTED = ("lr_low", "eval_noise")
# Fixed in PLAN.md before the test run.
MIN_SEED_SAVING, MAX_FLOOR = 2.0, 0.1
MIN_RIGHT, MIN_RIGHT_PER_TASK, MAX_FALSE_ALARMS, MIN_ADVICE = 0.85, 0.75, 0.1, 0.8
MIN_EARLY_WINS, MAX_LASTING = 0.5, 0.5


def pooled_variance(groups):
    """Variance within groups, pooled: here, across data orders within each checkpoint."""
    squares = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
    return squares / sum(len(g) - 1 for g in groups)


def at_probe(case, arm):
    """The arm's reading on every data order at the probe horizon (300 steps)."""
    return [branch[LR_CHUNK] for branch in case[arm]]


def finite(case, arms):
    return all(math.isfinite(x) for arm in arms for x in at_probe(case, arm))


def seed_saving(cases, arm):
    """How many times fewer seeds a paired comparison needs than an unpaired one, and the cases it could use."""
    cases = [c for c in cases if finite(c, (arm, "control"))]
    paired = pooled_variance(
        [
            [a - b for a, b in zip(at_probe(c, arm), at_probe(c, "control"))]
            for c in cases
        ]
    )
    unpaired = 2 * pooled_variance([at_probe(c, "control") for c in cases])
    return unpaired / paired, len(cases)


def pairing_spread(cases, arm):
    """Reported only. Correlation between arm and control readings across data orders, pooled within checkpoints,
    and the per-checkpoint saving (min, median, max). Each checkpoint has only a few data orders, so the
    per-checkpoint savings are noisy; the pooled correlation is the stable number."""
    cases = [c for c in cases if finite(c, (arm, "control"))]
    pairs = [(at_probe(c, "control"), at_probe(c, arm)) for c in cases]
    centred = [
        ([x - statistics.mean(k) for x in k], [y - statistics.mean(a) for y in a])
        for k, a in pairs
    ]
    sxy = sum(x * y for k, a in centred for x, y in zip(k, a))
    sxx = sum(x * x for k, _ in centred for x in k)
    syy = sum(y * y for _, a in centred for y in a)
    savings = sorted(
        2 * statistics.variance(k) / statistics.variance([y - x for x, y in zip(k, a)])
        for k, a in pairs
    )
    return sxy / (sxx * syy) ** 0.5, savings[0], statistics.median(savings), savings[-1]


def floor_ratio(cases, chunk=LR_CHUNK):
    """GPU non-determinism (RMS gap between two identical branches) over the SD across data orders."""
    cases = [c for c in cases if finite(c, ("repeat", "control"))]
    gaps = [
        r[chunk] - b[chunk]
        for c in cases
        for r, b in zip(c["repeat"], c["control"])
        if len(r) > chunk
    ]
    rms = statistics.mean(g * g for g in gaps) ** 0.5
    return rms / pooled_variance([at_probe(c, "control") for c in cases]) ** 0.5


def ahead(case, arm, chunk):
    """Is the arm's own-data-order branch below the control at this reading?"""
    return case[arm][0][chunk] < case["control"][0][chunk]


def share(hits, total):
    return f"{hits} of {total}" + (f" ({hits / total:.0%})" if total else "")


def pairing(healthy, tasks):
    """Print the pairing table for healthy runs; returns whether the seed-saving and non-determinism limits hold."""
    print("Pairing at 300 steps (seeds saved = unpaired variance / paired variance)")
    saves = quiet = True
    for task in tasks:
        subset = [c for c in healthy if c["model"] == task]
        for arm in INTERVENTIONS:
            saving, used = seed_saving(subset, arm)
            saves &= saving >= MIN_SEED_SAVING
            rho, low, mid, high = pairing_spread(subset, arm)
            print(
                f"  {task:6s} {arm:12s} {saving:8.1f}x  ({used} of {len(subset)} runs finite)"
                f"  rho {rho:.2f}; per-checkpoint {low:.1f}x / {mid:.1f}x / {high:.1f}x (min / median / max)"
            )
        ratio = floor_ratio(subset)
        quiet &= ratio <= MAX_FLOOR
        by_horizon = " ".join(
            f"{floor_ratio(subset, k):.3f}" for k in range(len(subset[0]["repeat"][0]))
        )
        print(
            f"  {task:6s} non-determinism / data-order SD: {ratio:.4f}; own data order, every 100 steps: {by_horizon}"
        )
    return saves, quiet


def repaired(case):
    """E9's suggested repair, reported only: the advised branch itself must beat the control on every data order."""
    label = diagnose(case)[0]
    if label == "eval_noise":
        return label
    wins = {
        fault: all(
            a < b for a, b in zip(at_probe(case, arm), at_probe(case, "control"))
        )
        for fault, arm in ADVICE.items()
    }
    if wins["lr_low"] != wins["lr_high"]:
        return "lr_low" if wins["lr_low"] else "lr_high"
    return "healthy"


def report_exact(cases):
    tasks = [t for t in TASKS if any(c["model"] == t for c in cases)]
    print("\nDeterministic kernels, healthy runs")
    saves, quiet = pairing(cases, tasks)
    print(
        f"H7 with deterministic kernels both limits hold on every task (>= {MIN_SEED_SAVING:g}x, <= {MAX_FLOOR:g}): "
        f"{supported(saves and quiet)}"
    )


def report(cases):
    tasks = [t for t in TASKS if any(c["model"] == t for c in cases)]
    healthy = [c for c in cases if c["fault"] == "healthy"]
    picks = {id(c): diagnose(c)[0] for c in cases}

    h1, h2 = pairing(healthy, tasks)
    print(
        f"H1 pairing saves at least {MIN_SEED_SAVING:g}x seeds for every intervention on every task: {supported(h1)}"
    )
    print(
        f"H2 GPU non-determinism at most {MAX_FLOOR:g} of data-order noise on every task: {supported(h2)}"
    )

    print("\nDiagnosis (E9 rule, unchanged)")
    faults = sorted({c["fault"] for c in cases})
    labels = sorted({picks[id(c)] for c in cases})
    worst, right, total = 1.0, 0, 0
    for task in tasks:
        subset = [c for c in cases if c["model"] == task]
        confusion = Counter((c["fault"], picks[id(c)]) for c in subset)
        print(
            f"  {task}" + " " * 9 + " ".join(f"{label[:10]:>10s}" for label in labels)
        )
        for fault in faults:
            print(
                f"  {fault:10s} "
                + " ".join(f"{confusion[fault, label]:10d}" for label in labels)
            )
        scoped = [c for c in subset if c["fault"] in IN_SCOPE]
        hits = sum(truth(c) == picks[id(c)] for c in scoped)
        worst, right, total = (
            min(worst, hits / len(scoped)),
            right + hits,
            total + len(scoped),
        )
        cost = statistics.mean(diagnose(c)[1] for c in subset)
        print(
            f"  right on {', '.join(IN_SCOPE)}: {share(hits, len(scoped))}; mean cost {cost:.0f} steps\n"
        )
    h3 = right / total >= MIN_RIGHT and worst >= MIN_RIGHT_PER_TASK
    print(
        f"H3 LR-too-low and noisy-evaluation cases right (>= {MIN_RIGHT:.0%} pooled, >= {MIN_RIGHT_PER_TASK:.0%} per task): "
        f"{share(right, total)}, worst task {worst:.0%}: {supported(h3)}"
    )
    alarms = sum(picks[id(c)] in TRUSTED for c in healthy)
    h4 = alarms / len(healthy) <= MAX_FALSE_ALARMS
    print(
        f"H4 healthy runs given either diagnosis (<= {MAX_FALSE_ALARMS:.0%}): {share(alarms, len(healthy))}: {supported(h4)}"
    )
    advised = [c for c in cases if picks[id(c)] == "lr_low"]
    helped = sum(advice_helped(c, "lr_low") for c in advised)
    h5 = bool(advised) and helped / len(advised) >= MIN_ADVICE
    print(
        f"H5 doubling the LR, when advised, is ahead 900 steps later (>= {MIN_ADVICE:.0%}): "
        f"{share(helped, len(advised))}: {supported(h5)}"
    )

    early = [c for c in healthy if ahead(c, "halve_lr", LR_CHUNK)]
    lasting = sum(ahead(c, "halve_lr", -1) for c in early)
    h6 = (
        len(early) / len(healthy) >= MIN_EARLY_WINS
        and lasting / len(early) <= MAX_LASTING
    )
    print(
        f"\nH6 on healthy runs a halved LR is ahead at 300 steps (>= {MIN_EARLY_WINS:.0%}) and that lead mostly does not "
        f"last to the end (<= {MAX_LASTING:.0%}): ahead {share(len(early), len(healthy))}, "
        f"still ahead {share(lasting, len(early))}: {supported(h6)}"
    )

    print("\nReported, not decisive")
    fixed = Counter((truth(c), repaired(c)) for c in cases if c["fault"] != "mild_high")
    print(
        f"E9's suggested repair (advised branch must beat control on all 3 data orders), truth -> reading: {dict(fixed)}"
    )
    high = sum(picks[id(c)] == "lr_high" for c in healthy)
    print(f"healthy runs told 'LR too high': {share(high, len(healthy))}")
    mild_high = [c for c in cases if c["fault"] == "mild_high"]
    print(f"LR 3x read as: {dict(Counter(picks[id(c)] for c in mild_high))}")
    final_helped = sum(ahead(c, ADVICE["lr_low"], -1) for c in advised)
    print(
        f"doubling the LR, when advised, ahead at the final reading: {share(final_helped, len(advised))}"
    )
    steady = [c for c in cases if c["fault"] != "eval_noise"]
    chunks = len(steady[0]["control"][0])
    for arm in ("halve_lr", "double_lr"):
        agree = [
            sum(ahead(c, arm, k) == ahead(c, arm, -1) for c in steady)
            for k in range(chunks)
        ]
        print(
            f"{arm}: runs whose sign at each 100 steps matches the final sign, of {len(steady)}: {agree}"
        )
    for task in tasks:
        subset = [
            c
            for c in healthy
            if c["model"] == task and finite(c, ("weight_decay", "control"))
        ]
        effects = [
            statistics.mean(
                a - b
                for a, b in zip(at_probe(c, "weight_decay"), at_probe(c, "control"))
            )
            for c in subset
        ]
        if len(effects) < 2:
            continue
        spread = statistics.stdev(effects) / len(effects) ** 0.5
        print(
            f"{task}: weight-decay effect at 300 steps {statistics.mean(effects):+.5f} +/- {2.26 * spread:.5f} "
            f"(mean of {len(effects)} checkpoints)"
        )


def selftest():
    def branches(*ends, length=15):
        return [[end] * (length if k == 0 else 3) for k, end in enumerate(ends)]

    def case(shift, **fields):
        base = dict(
            model="lm",
            fault="healthy",
            metric=[3.2, 3.0, 2.9],
            eval_batches=40,
            noise_reads=[2.9] * 4,
            control=branches(2.80 + shift, 2.84 + shift, 2.76 + shift),
            halve_lr=branches(2.81 + shift, 2.85 + shift, 2.772 + shift),
            double_lr=branches(2.80 + shift, 2.86 + shift, 2.74 + shift),
            repeat=branches(2.80 + shift, 2.84 + shift, 2.76 + shift + 0.004),
            weight_decay=branches(2.801 + shift, 2.84 + shift, 2.76 + shift),
        )
        return base | fields

    cases = [case(0.0), case(0.3)]
    assert abs(pooled_variance([[1, 3], [10, 12, 14]]) - 10 / 3) < 1e-12
    # a nearly constant offset pairs almost perfectly; an effect that varies with the data order does not
    assert seed_saving(cases, "halve_lr")[0] > 1000
    saving, used = seed_saving(cases, "double_lr")
    assert used == 2 and abs(saving - 2 * 0.0016 / 0.0004) < 1e-6
    # one branch in three differs by 0.004, against a data-order SD of 0.04
    rho, low, _, high = pairing_spread(cases, "halve_lr")
    assert rho > 0.99 and low > 1000 <= high
    assert abs(floor_ratio(cases) - (0.004**2 / 3) ** 0.5 / 0.04) < 1e-9
    diverged = case(0.0, double_lr=branches(math.inf, 2.86, 2.74))
    assert seed_saving([*cases, diverged], "double_lr")[1] == 2
    assert not ahead(cases[0], "halve_lr", -1)
    assert ahead(case(0.0, halve_lr=branches(2.7, 2.7, 2.7)), "halve_lr", LR_CHUNK)
    report(
        cases
        + [
            case(
                0.0,
                fault="lr_low",
                halve_lr=branches(2.9, 2.9, 2.9),
                double_lr=branches(2.7, 2.7, 2.7),
            )
        ]
    )
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        load = lambda paths: [
            c for path in paths for c in json.load(open(path))["cases"]
        ]
        if sys.argv[1:]:
            report(load(sys.argv[1:]))
        else:
            report(load(f"results/e11/e11_{task}.json" for task in TASKS))
            report_exact(load(f"results/e11/exact_{task}.json" for task in TASKS))
