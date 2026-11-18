"""E7 analysis: leave-one-seed-out diagnosis from probe outcomes. Rules are in run.py's docstring.

Usage: python -m studies.e7_diagnosis.analyze [results/e7/e7.json]   (no argument after `selftest` runs the self-check)
"""

import itertools
import json
import math
import sys
from collections import Counter

PROBES = ("halve_lr", "double_lr", "reeval_4x", "wider")
NEEDS_CONTROL = {"halve_lr", "double_lr", "wider"}
TRAIN_COST = {"halve_lr": 100, "double_lr": 100, "wider": 1600}
CONTROL_COST, STOP_AT, OUTCOMES = 100, 0.9, 3


def probe_readings(case):
    return case["probes"]


def passive_readings(case):
    return {
        "level": case["metric_300"],
        "trend": case["metric_300"] - case["metric_200"],
    }


def finite(value):
    return value if math.isfinite(value) else math.inf


def outcome(value, cuts):
    return sum(finite(value) >= cut for cut in cuts)


def fit(cases, readings):
    """Tercile cut points per reading and p(outcome | fault) with add-one smoothing."""
    faults = sorted({c["fault"] for c in cases})
    cuts, likelihood = {}, {}
    for name in readings(cases[0]):
        values = sorted(finite(readings(c)[name]) for c in cases)
        edges = [len(values) * k // OUTCOMES for k in range(1, OUTCOMES)]
        # midpoint between neighbours, so a held-out value at the edge of a cluster stays with its cluster
        cuts[name] = [(values[i - 1] + values[i]) / 2 for i in edges]
        likelihood[name] = {}
        for fault in faults:
            seen = Counter(
                outcome(readings(c)[name], cuts[name])
                for c in cases
                if c["fault"] == fault
            )
            total = sum(seen.values()) + OUTCOMES
            likelihood[name][fault] = [(seen[y] + 1) / total for y in range(OUTCOMES)]
    return dict(faults=faults, cuts=cuts, likelihood=likelihood)


def entropy(belief):
    return -sum(p * math.log2(p) for p in belief.values() if p > 0)


def update(belief, table, y):
    joint = {fault: p * table[fault][y] for fault, p in belief.items()}
    z = sum(joint.values())
    return {fault: j / z for fault, j in joint.items()}


def info_gain(belief, table):
    """Expected entropy reduction (bits) over faults from observing one reading."""
    expected = 0.0
    for y in range(OUTCOMES):
        p_y = sum(p * table[fault][y] for fault, p in belief.items())
        if p_y > 0:
            expected += p_y * entropy(update(belief, table, y))
    return entropy(belief) - expected


def cost_of(name, case, control_paid):
    if name not in TRAIN_COST:
        return 4 * case["eval_batches"] / 3
    return TRAIN_COST[name] + (0 if control_paid else CONTROL_COST)


def diagnose(case, model, readings, choose, stop_at=STOP_AT):
    """Run probes chosen by `choose` until one fault reaches `stop_at`. Returns (pick, confidence, cost, probes)."""
    belief = {fault: 1 / len(model["faults"]) for fault in model["faults"]}
    remaining, spent, control_paid = list(readings(case)), 0.0, False
    while remaining and max(belief.values()) < stop_at:
        name = choose(belief, remaining, model, case, control_paid)
        remaining.remove(name)
        spent += cost_of(name, case, control_paid)
        control_paid |= name in NEEDS_CONTROL
        y = outcome(readings(case)[name], model["cuts"][name])
        belief = update(belief, model["likelihood"][name], y)
    pick = max(belief, key=belief.get)
    return pick, belief[pick], spent, len(readings(case)) - len(remaining)


def by_gain_per_cost(belief, remaining, model, case, paid):
    return max(
        remaining,
        key=lambda n: info_gain(belief, model["likelihood"][n])
        / cost_of(n, case, paid),
    )


def by_gain(belief, remaining, model, case, paid):
    return max(remaining, key=lambda n: info_gain(belief, model["likelihood"][n]))


def cheapest(belief, remaining, model, case, paid):
    return min(remaining, key=lambda n: cost_of(n, case, paid))


def fixed(order):
    return lambda belief, remaining, model, case, paid: next(
        n for n in order if n in remaining
    )


def summarize(rows):
    """rows: (truth, pick, confidence, cost, probes)."""
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


def evaluate(cases):
    """Leave-one-seed-out. Returns {policy: summary} plus the ALL policy's confusion counts."""
    policies = {
        "EIG_PER_COST": [by_gain_per_cost],
        "EIG": [by_gain],
        "CHEAPEST": [cheapest],
    }
    policies["RANDOM"] = [fixed(order) for order in itertools.permutations(PROBES)]
    rows = {name: [] for name in [*policies, "ALL", "PASSIVE"]}
    for held_out in sorted({c["seed"] for c in cases}):
        train = [c for c in cases if c["seed"] != held_out]
        model, passive = fit(train, probe_readings), fit(train, passive_readings)
        for case in (c for c in cases if c["seed"] == held_out):
            for name, choosers in policies.items():
                for choose in choosers:
                    rows[name].append(
                        (case["fault"], *diagnose(case, model, probe_readings, choose))
                    )
            rows["ALL"].append(
                (
                    case["fault"],
                    *diagnose(case, model, probe_readings, fixed(PROBES), stop_at=2),
                )
            )
            everything = fixed(list(passive_readings(case)))
            rows["PASSIVE"].append(
                (
                    case["fault"],
                    *diagnose(case, passive, passive_readings, everything, stop_at=2),
                )
            )
    confusion = Counter((truth, pick) for truth, pick, *_ in rows["ALL"])
    return {name: summarize(r) for name, r in rows.items()}, confusion


def report(cases):
    summary, confusion = evaluate(cases)
    faults = sorted({c["fault"] for c in cases})
    print(
        f"{len(cases)} cases, {len({c['seed'] for c in cases})} seeds, chance {1 / len(faults):.2f}\n"
    )
    print(
        f"{'policy':13s} {'accuracy':>8s} {'cost':>7s} {'probes':>6s} {'>=0.9':>6s} {'acc@>=0.9':>9s}"
    )
    for name, s in summary.items():
        print(
            f"{name:13s} {s['accuracy']:8.2f} {s['cost']:7.0f} {s['probes']:6.2f} {s['confident']:6.2f} {s['confident_accuracy']:9.2f}"
        )
    print("\nALL confusion (rows truth, columns diagnosis):")
    print(" " * 11 + " ".join(f"{f[:9]:>9s}" for f in faults))
    for truth in faults:
        print(
            f"{truth:10s} "
            + " ".join(f"{confusion[truth, pick]:9d}" for pick in faults)
        )

    ours, rand, cheap = summary["EIG_PER_COST"], summary["RANDOM"], summary["CHEAPEST"]

    def beats(other):
        return (
            ours["cost"] <= 0.8 * other["cost"]
            and ours["accuracy"] >= other["accuracy"] - 0.04
        )

    h2 = ("STRONG" if beats(cheap) else "WEAK") if beats(rand) else "NOT SUPPORTED"
    print(
        f"\nH1 diagnosis works (ALL accuracy >= 0.80): {'SUPPORTED' if summary['ALL']['accuracy'] >= 0.8 else 'NOT SUPPORTED'}"
    )
    print(f"H2 selection saves compute: {h2}")
    print(
        f"H3 honest confidence (>= 0.85 right when >= 0.9 sure): {'SUPPORTED' if ours['confident_accuracy'] >= 0.85 else 'NOT SUPPORTED'}"
    )
    return summary


def selftest():
    """Every probe reads the fault's level, so any two probes settle it and the cheap ones go first."""
    cases = [
        dict(
            fault=fault,
            seed=seed,
            eval_batches=20,
            metric_200=1.0,
            metric_300=1.0,
            probes={probe: level + 0.001 * seed for probe in PROBES},
        )
        for level, fault in enumerate(("low", "mid", "high"))
        for seed in range(12)
    ]
    summary, _ = evaluate(cases)
    assert summary["ALL"]["accuracy"] == 1.0, summary["ALL"]
    assert summary["EIG_PER_COST"]["accuracy"] == 1.0, summary["EIG_PER_COST"]
    assert (
        summary["EIG_PER_COST"]["cost"]
        < summary["RANDOM"]["cost"]
        < summary["ALL"]["cost"]
    )
    assert (
        abs(info_gain({"a": 0.5, "b": 0.5}, {"a": [1, 0, 0], "b": [0, 1, 0]}) - 1.0)
        < 1e-12
    )
    print("selftest ok")


if __name__ == "__main__":
    argument = sys.argv[1] if len(sys.argv) > 1 else "results/e7/e7.json"
    selftest() if argument == "selftest" else report(json.load(open(argument))["cases"])
