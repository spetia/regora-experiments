"""Decision replay: how often does a comparison point the wrong way?

Turns the study's readings into the question a team actually asks: "did this change help?". For a change whose
direction is established on all the pairs of a cell, it counts how often a small comparison gets that direction
wrong, done two ways with the same number of training runs:

- unpaired: n runs of the control and n runs of the change on different data orders (what a tracker's compare
  view shows when the two runs were launched separately);
- paired: n data orders, the control and the change on each.

Every way of drawing the runs from a checkpoint's orders is enumerated, so there is no sampling noise. The
direction counted as true is the sign of the mean paired difference over all pairs, and only rows where that
mean is more than two standard errors from zero are scored. The truth comes from the same readings the draws are
made from, which flatters both methods slightly and neither more than the other. "Established" uses the paired
standard error, which admits small effects that only pairing can see; the summary therefore also splits the rows
by whether the direction is established without pairing as well.

    python replay.py [cell ...]     # default: every finished cell
"""

import itertools
import math
import os
import sys

import numpy as np

import analyze

SKIP_ARMS = (
    "swap32",
)  # a control on another data order: there is no direction to get right


def wrong_share(diffs, truth):
    """Share of comparisons whose sign disagrees with `truth`; an exact tie counts as half."""
    s = np.sign(np.asarray(diffs))
    return float(((s == -truth) + 0.5 * (s == 0)).mean())


def paired_wrong(ctrl, arm, n, truth):
    """ctrl, arm: (K, S). Mean over checkpoints of the wrong share over every choice of n orders."""
    d = arm - ctrl
    S = d.shape[1]
    picks = [list(c) for c in itertools.combinations(range(S), n)]
    return float(
        np.mean([wrong_share([row[p].mean() for p in picks], truth) for row in d])
    )


def unpaired_wrong(ctrl, arm, n, truth):
    """As `paired_wrong`, but the change runs on n orders and the control on n other orders."""
    S = ctrl.shape[1]
    if 2 * n > S:
        return None
    picks = [
        (list(a), list(b))
        for a in itertools.combinations(range(S), n)
        for b in itertools.combinations([i for i in range(S) if i not in a], n)
    ]
    return float(
        np.mean(
            [
                wrong_share(
                    [arm[k, a].mean() - ctrl[k, b].mean() for a, b in picks], truth
                )
                for k in range(ctrl.shape[0])
            ]
        )
    )


def rows(name):
    """One row per (arm, horizon) of a finished cell whose direction is established."""
    d = analyze.load(name)
    if d is None:
        return []
    hs = list(d["config"]["horizons"])
    ctrl = analyze.table(d, "control")
    out = []
    for arm in dict.fromkeys(k.split("|")[2] for k in d["branches"]):
        if arm in ("control", "twin") + SKIP_ARMS:
            continue
        t = analyze.table(d, arm)
        for i, h in enumerate(hs):
            st = analyze.stats(ctrl[:, :, i], t[:, :, i])
            if st["diverged"] or not st["resolved"]:
                continue
            truth = math.copysign(1.0, st["effect"])
            # Stricter: the direction would also be established without pairing, on the same number of runs.
            strict = abs(st["effect"]) > 2 * math.sqrt(
                (st["sd_arm"] ** 2 + st["sd_ctrl"] ** 2) / ctrl[:, :, i].size
            )
            out.append(
                dict(
                    cell=name,
                    arm=arm,
                    h=h,
                    saved=st["saved"],
                    effect=st["effect"],
                    strict=strict,
                    **{
                        f"u{n}": unpaired_wrong(ctrl[:, :, i], t[:, :, i], n, truth)
                        for n in (1, 2, 3)
                    },
                    **{
                        f"p{n}": paired_wrong(ctrl[:, :, i], t[:, :, i], n, truth)
                        for n in (1, 2, 3)
                    },
                )
            )
    return out


def pct(x):
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def summary(R, label):
    if not R:
        return
    line = f"  {label:34} {len(R):4d} rows  "
    for n in (1, 2, 3):
        have = [r for r in R if r[f"u{n}"] is not None]
        if have:
            line += (
                f" n={n}: unpaired {pct(np.mean([r[f'u{n}'] for r in have]))}"
                f" paired {pct(np.mean([r[f'p{n}'] for r in have]))} ({len(have)})"
            )
    print(line)


def main():
    names = sys.argv[1:] or sorted(
        f[:-5] for f in os.listdir(analyze.OUT) if f.endswith(".json") and f[1] == "_"
    )
    R = [r for n in names for r in rows(n)]
    print(
        "wrong-direction rate, by runs per arm (n). 'rows' = (cell, arm, horizon) with an established direction"
    )
    print(
        f"  {'cell':18} {'arm':8} {'h':>5} {'saved':>8}   n=1 unp / pair    n=2 unp / pair    n=3 unp / pair"
    )
    for r in R:
        print(
            f"  {r['cell']:18} {r['arm']:8} {r['h']:5d} {r['saved']:8.1f}   "
            + "   ".join(f"{pct(r[f'u{n}'])} /{pct(r[f'p{n}'])}" for n in (1, 2, 3))
        )
    print("\nmeans over rows (in brackets: rows that have enough orders for that n)")
    summary(R, "all rows")
    summary([r for r in R if r["strict"]], "established without pairing too")
    summary([r for r in R if not r["strict"]], "established only by pairing")
    for h in sorted({r["h"] for r in R}):
        summary([r for r in R if r["h"] == h], f"h={h}")
    summary([r for r in R if r["saved"] < 3], "seeds saved below 3")
    summary([r for r in R if 3 <= r["saved"] < 30], "seeds saved 3 to 30")
    summary([r for r in R if r["saved"] >= 30], "seeds saved 30 or more")
    hard = [r for r in R if r["u1"] is not None and r["u1"] >= 0.1]
    summary(hard, "unpaired n=1 wrong at least 10%")


if __name__ == "__main__":
    main()
