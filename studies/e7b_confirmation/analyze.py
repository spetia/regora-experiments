"""E7b analysis: diagnosis with continuous likelihoods, fitted on E7 and tested on fresh seeds. Rules are in run.py.

Usage: python -m studies.e7b_confirmation.analyze            confirmation on results/e7/e7b.json
       python -m studies.e7b_confirmation.analyze posthoc    leave-one-seed-out on E7 itself (post hoc, not evidence)
"""

import itertools
import json
import math
import sys
from collections import Counter

import numpy as np
from scipy import stats

PASSIVE = ("level", "trend")
PROBES = ("halve_lr", "double_lr", "reeval_4x", "wider")
NEEDS_CONTROL = {"halve_lr", "double_lr", "wider"}
TRAIN_COST = {"halve_lr": 100, "double_lr": 100, "wider": 1600}
CONTROL_COST, STOP_AT, QUANTILES = 100, 0.9, (np.arange(25) + 0.5) / 25


def readings(case):
    passive = {
        "level": case["metric_300"],
        "trend": case["metric_300"] - case["metric_200"],
    }
    return passive | case["probes"]


def fit(cases):
    """Student-t predictive density per (reading, fault), plus the value that stands in for non-finite readings."""
    faults = sorted({c["fault"] for c in cases})
    density, ceiling = {}, {}
    for name in PASSIVE + PROBES:
        ceiling[name] = (
            max(v for c in cases if math.isfinite(v := readings(c)[name])) + 1
        )
        for fault in faults:
            values = [
                min(readings(c)[name], ceiling[name])
                for c in cases
                if c["fault"] == fault
            ]
            values = [v if math.isfinite(v) else ceiling[name] for v in values]
            n = len(values)
            scale = max(np.std(values, ddof=1), 1e-6) * math.sqrt(1 + 1 / n)
            density[name, fault] = stats.t(df=n - 1, loc=np.mean(values), scale=scale)
    return dict(faults=faults, density=density, ceiling=ceiling)


def update(belief, model, name, value):
    value = value if math.isfinite(value) else model["ceiling"][name]
    log_post = {
        f: math.log(p) + model["density"][name, f].logpdf(value)
        for f, p in belief.items()
        if p > 0
    }
    top = max(log_post.values())
    weights = {f: math.exp(v - top) for f, v in log_post.items()}
    total = sum(weights.values())
    return {f: weights.get(f, 0.0) / total for f in belief}


def entropy(belief):
    return -sum(p * math.log2(p) for p in belief.values() if p > 0)


def info_gain(belief, model, name):
    """Expected entropy reduction (bits), averaging over quantiles of each fault's predictive density."""
    expected = 0.0
    for fault, p in belief.items():
        outcomes = model["density"][name, fault].ppf(QUANTILES)
        expected += p * np.mean(
            [entropy(update(belief, model, name, x)) for x in outcomes]
        )
    return entropy(belief) - expected


def cost_of(name, case, control_paid):
    if name in PASSIVE:
        return 0.0
    if name not in TRAIN_COST:
        return 4 * case["eval_batches"] / 3
    return TRAIN_COST[name] + (0 if control_paid else CONTROL_COST)


def by_gain_per_cost(belief, remaining, model, case, paid):
    return max(
        remaining, key=lambda n: info_gain(belief, model, n) / cost_of(n, case, paid)
    )


def cheapest(belief, remaining, model, case, paid):
    return min(remaining, key=lambda n: cost_of(n, case, paid))


def fixed(order):
    return lambda belief, remaining, model, case, paid: next(
        n for n in order if n in remaining
    )


def diagnose(case, model, choose, probes=PROBES, stop_at=STOP_AT, passive=PASSIVE):
    """Read the passive signals, then probes chosen by `choose`. Returns (pick, confidence, cost, probes used)."""
    belief = {fault: 1 / len(model["faults"]) for fault in model["faults"]}
    for name in passive:
        belief = update(belief, model, name, readings(case)[name])
    remaining, spent, control_paid = list(probes), 0.0, False
    while remaining and max(belief.values()) < stop_at:
        name = choose(belief, remaining, model, case, control_paid)
        remaining.remove(name)
        spent += cost_of(name, case, control_paid)
        control_paid |= name in NEEDS_CONTROL
        belief = update(belief, model, name, readings(case)[name])
    pick = max(belief, key=belief.get)
    return pick, belief[pick], spent, len(probes) - len(remaining)


def score(train, test, rows, passive=PASSIVE):
    """Fit on `train`, diagnose every case in `test` under each policy, append to `rows`."""
    model = fit(train)
    policies = {"EIG_PER_COST": [by_gain_per_cost], "CHEAPEST": [cheapest]}
    policies["RANDOM"] = [fixed(order) for order in itertools.permutations(PROBES)]
    for case in test:
        for name, choosers in policies.items():
            for choose in choosers:
                rows[name].append(
                    (case["fault"], *diagnose(case, model, choose, passive=passive))
                )
        rows["ALL"].append(
            (
                case["fault"],
                *diagnose(case, model, fixed(PROBES), stop_at=2, passive=passive),
            )
        )
        rows["PASSIVE"].append(
            (case["fault"], *diagnose(case, model, None, probes=(), passive=passive))
        )


def empty_rows():
    return {
        name: [] for name in ("EIG_PER_COST", "CHEAPEST", "RANDOM", "ALL", "PASSIVE")
    }


def summarize(rows):
    confident = [r for r in rows if r[2] >= STOP_AT]
    return dict(
        accuracy=sum(r[0] == r[1] for r in rows) / len(rows),
        cost=sum(r[3] for r in rows) / len(rows),
        probes=sum(r[4] for r in rows) / len(rows),
        confident=len(confident) / len(rows),
        confident_accuracy=(
            sum(r[0] == r[1] for r in confident) / len(confident)
            if confident
            else float("nan")
        ),
    )


def report(rows, faults):
    summary = {name: summarize(r) for name, r in rows.items()}
    print(
        f"{'policy':13s} {'accuracy':>8s} {'cost':>7s} {'probes':>6s} {'>=0.9':>6s} {'acc@>=0.9':>9s}"
    )
    for name, s in summary.items():
        print(
            f"{name:13s} {s['accuracy']:8.2f} {s['cost']:7.0f} {s['probes']:6.2f} {s['confident']:6.2f} {s['confident_accuracy']:9.2f}"
        )
    confusion = Counter((truth, pick) for truth, pick, *_ in rows["ALL"])
    print("\nALL confusion (rows truth, columns diagnosis):")
    print(" " * 11 + " ".join(f"{f[:9]:>9s}" for f in faults))
    for truth in faults:
        print(
            f"{truth:10s} "
            + " ".join(f"{confusion[truth, pick]:9d}" for pick in faults)
        )
    ours, rand, cheap, everything = (
        summary[k] for k in ("EIG_PER_COST", "RANDOM", "CHEAPEST", "ALL")
    )

    def beats(other):
        return (
            ours["cost"] <= 0.8 * other["cost"]
            and ours["accuracy"] >= other["accuracy"] - 0.04
        )

    def verdict(ok):
        return "SUPPORTED" if ok else "NOT SUPPORTED"

    print(
        f"\nH1 diagnosis works (ALL accuracy >= 0.80): {verdict(everything['accuracy'] >= 0.8)}"
    )
    print(
        f"H2 selection saves compute: {('STRONG' if beats(cheap) else 'WEAK') if beats(rand) else 'NOT SUPPORTED'}"
    )
    print(
        f"H3 honest confidence (>= 0.85 right when >= 0.9 sure): {verdict(ours['confident_accuracy'] >= 0.85)}"
    )
    print(
        f"H4 probes add to watching (ALL - PASSIVE >= 0.10): {verdict(everything['accuracy'] - summary['PASSIVE']['accuracy'] >= 0.10)}"
    )


if __name__ == "__main__":
    e7 = json.load(open("results/e7/e7.json"))["cases"]
    rows = empty_rows()
    if sys.argv[1:] == ["posthoc"]:
        print("POST HOC: leave-one-seed-out on E7 itself. Not evidence.\n")
        for seed in sorted({c["seed"] for c in e7}):
            score(
                [c for c in e7 if c["seed"] != seed],
                [c for c in e7 if c["seed"] == seed],
                rows,
            )
    else:
        fresh = json.load(open("results/e7/e7b.json"))["cases"]
        assert not {c["seed"] for c in fresh} & {
            c["seed"] for c in e7
        }, "test seeds overlap the fitting seeds"
        print(f"Fitted on {len(e7)} E7 cases, tested on {len(fresh)} fresh cases.\n")
        score(e7, fresh, rows)
    report(rows, sorted({c["fault"] for c in e7}))
