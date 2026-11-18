"""Post hoc held-out check of E14's picks (not pre-registered).

E14 scored a pick against the change that is best over all 16 seeds, the same seeds the replay draws from. Here the
16 seeds are split at random into a selection half A and a held-out half B. An allocator sees only A (its orders
and budget), picks a change, and the pick is scored on B: its mean gain over the control on seeds the allocator
never saw. The allocators are a Python port of `regora-eie` `src/allocate.rs` (uniform, halving, LUCB with 3 warm-up
seeds); `check` replays the full E14 protocol with the port and compares it with `results/e14/e14_replay.txt`.

    uv run --with numpy --with scipy python -m studies.e14_allocation_hard.heldout check
    uv run --with numpy --with scipy python -m studies.e14_allocation_hard.heldout run | tee results/e14/e14_heldout.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TABLES = Path(__file__).resolve().parents[2] / "results" / "e14" / "tables.json"
SPLITS, ORDERS, SEED = 1000, 1000, 20261116


def t95(df):
    return float(stats.t.ppf(0.975, df))


def stats_of(d):
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5
    return mean, sd, t95(n - 1) * sd / n**0.5


def mean(d):
    return sum(d) / len(d)


def argmax(candidates, score):
    best = None
    for i in candidates:
        if best is None or not score(best) >= score(i):
            best = i
    return best


def pick(seen):
    observed = [i for i in range(len(seen)) if seen[i]]
    best = argmax(observed, lambda i: mean(seen[i]))
    return best if best is not None and mean(seen[best]) > 0 else None


def cost(seen):
    return sum(len(s) for s in seen) + max(len(s) for s in seen)


class Uniform:
    def next(self, seen, cap):
        return argmax(
            [i for i in range(len(seen)) if len(seen[i]) < cap], lambda i: -len(seen[i])
        )


class Halving:
    def __init__(self, changes, per_round):
        self.per_round, self.alive, self.target = (
            per_round,
            list(range(changes)),
            per_round,
        )

    def next(self, seen, cap):
        while True:
            target = min(self.target, cap)
            for i in self.alive:
                if len(seen[i]) < target:
                    return i
            if self.target >= cap:
                return None
            if len(self.alive) > 1:
                self.alive.sort(key=lambda i: -mean(seen[i]))
                self.alive = self.alive[: -(-len(self.alive) // 2)]
            self.target += self.per_round


class Lucb:
    def __init__(self, warmup=3):
        self.warmup = warmup

    def next(self, seen, cap):
        warmup = min(max(self.warmup, 2), cap)
        i = argmax(
            [i for i in range(len(seen)) if len(seen[i]) < warmup],
            lambda i: -len(seen[i]),
        )
        if i is not None:
            return i
        bound = lambda i, sign: stats_of(seen[i])[0] + sign * stats_of(seen[i])[2]
        leader = argmax(range(len(seen)), lambda i: mean(seen[i]))
        challenger = argmax(
            [i for i in range(len(seen)) if i != leader], lambda i: bound(i, 1.0)
        )
        if challenger is None:
            return leader if len(seen[leader]) < cap else None
        if bound(leader, -1.0) > bound(challenger, 1.0):
            return None
        first, second = (
            (leader, challenger)
            if len(seen[leader]) < len(seen[challenger])
            else (challenger, leader)
        )
        return next((i for i in (first, second) if len(seen[i]) < cap), None)


def make(name, changes, budget):
    return {
        "uniform": Uniform,
        "halving": lambda: Halving(changes, max(2, budget // 18)),
        "lucb": Lucb,
    }[name]()


def replay(table, order, budget, allocator):
    seen = [[] for _ in table]
    while (i := allocator.next(seen, len(order))) is not None:
        seen[i].append(table[i][order[len(seen[i])]])
        if cost(seen) > budget:
            seen[i].pop()
            break
    return pick(seen), cost(seen), seen


def load():
    return {t["name"]: t for t in json.loads(TABLES.read_text())}


def check(budgets=range(24, 145, 12)):
    """The E14 protocol (truth = best over all 16 seeds) with the port, to compare with e14_replay.txt."""
    rng = np.random.default_rng(SEED)
    for name, t in load().items():
        table = t["diffs"]
        full = [mean(r) for r in table]
        truth = int(np.argmax(full)) if max(full) > 0 else None
        orders = [rng.permutation(len(table[0])).tolist() for _ in range(300)]
        print(f"{name}: best over all seeds = {t['changes'][truth]}")
        for alg in ("uniform", "halving", "lucb"):
            row = [
                np.mean(
                    [
                        replay(table, o, b, make(alg, len(table), b))[0] == truth
                        for o in orders
                    ]
                )
                for b in budgets
            ]
            print(
                f"  {alg:8s}"
                + "".join(f"{x:6.2f}" for x in row)
                + f"   mean {np.mean(row):.3f}"
            )


def heldout(budgets=range(24, 73, 12)):
    rng = np.random.default_rng(SEED + 1)
    for name, t in load().items():
        table = np.array(t["diffs"])
        n_seeds = table.shape[1]
        half = n_seeds // 2
        splits = []
        for _ in range(SPLITS):
            p = rng.permutation(n_seeds)
            splits.append((p[:half].tolist(), p[half:].tolist()))
        print(
            f"\n{name}: {len(table)} changes, {half} selection seeds and {n_seeds - half} held-out seeds per split, {SPLITS} splits"
        )
        print(
            "  budget  allocator   claimed(sel)  delivered(held-out)  pick>0 held-out  pick best held-out  keep control  runs"
        )
        for b in budgets:
            for alg in ("uniform", "halving", "lucb"):
                claimed, delivered, positive, best, control, runs = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for a_seeds, b_seeds in splits:
                    sel = table[:, a_seeds].tolist()
                    k, c, _ = replay(
                        sel, list(range(half)), b, make(alg, len(table), b)
                    )
                    runs.append(c)
                    if k is None:
                        claimed.append(0.0), delivered.append(0.0), positive.append(
                            False
                        ), control.append(True)
                        best.append(max(table[:, b_seeds].mean(axis=1)) <= 0)
                        continue
                    held = table[k, b_seeds].mean()
                    claimed.append(float(np.mean(sel[k]))), delivered.append(
                        float(held)
                    ), positive.append(bool(held > 0))
                    control.append(False)
                    best.append(
                        k == int(np.argmax(table[:, b_seeds].mean(axis=1))) and held > 0
                    )
                print(
                    f"  {b:6d}  {alg:9s}   {np.mean(claimed):+.4f}      {np.mean(delivered):+.4f}            {np.mean(positive):6.2f}"
                    f"            {np.mean(best):6.2f}            {np.mean(control):5.2f}  {np.mean(runs):5.1f}"
                )
        # the status quo (one run of each change and one control, keep the highest) on the same splits
        claimed, delivered, positive, control = [], [], [], []
        for a_seeds, b_seeds in splits:
            for s in a_seeds:
                col = table[:, s]
                k = int(np.argmax(col))
                if col[k] > 0:
                    claimed.append(float(col[k])), delivered.append(
                        float(table[k, b_seeds].mean())
                    )
                    positive.append(bool(table[k, b_seeds].mean() > 0)), control.append(
                        False
                    )
                else:
                    claimed.append(0.0), delivered.append(0.0), positive.append(
                        False
                    ), control.append(True)
        print(
            f"  status quo (one run each, per selection seed): claimed {np.mean(claimed):+.4f}  delivered {np.mean(delivered):+.4f}"
            f"  pick>0 held-out {np.mean(positive):.2f}  keep control {np.mean(control):.2f}"
        )


if __name__ == "__main__":
    {"check": check, "run": heldout}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
