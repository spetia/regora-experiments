"""Analyse E2 per the rule in run.py. Usage: python -m studies.e2_paired_vs_halving.analyze [results.json] [seeds-per-selector]."""

import json, itertools, math, sys, numpy as np
from scipy.stats import t as tdist

path = sys.argv[1] if len(sys.argv) > 1 else "results/e2/e2.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
d = json.load(open(path))
cfg = d["config"]
C = d["curves"]
confs = list(itertools.product(cfg["lrs"], cfg["warms"]))
S = list(cfg["seeds"])
K = len(confs)
T = cfg["steps"]
steps = [s for s, _ in C[f"{confs[0][0]}|{confs[0][1]}|{S[0]}"]]
# L[config, seed, eval index]; non-finite -> large finite so means and diffs stay defined
L = np.array(
    [
        [
            [v if math.isfinite(v) else 1e9 for _, v in C[f"{c[0]}|{c[1]}|{s}"]]
            for s in S
        ]
        for c in confs
    ]
)
truth = L[:, :, -1].mean(1)
best = truth.min()
J = {h: steps.index(h) for h in (250, 500, 750, 1000)}
TCRIT = tdist.ppf(0.95, N - 1)


def sh(sub, rungs):
    alive, used, prev = list(range(K)), 0, 0
    for h, keep in zip(rungs, (4, 2, 1)):
        used += len(alive) * (h - prev)
        prev = h
        alive = sorted(alive, key=lambda i: L[i, sub, J[h]].mean())[:keep]
    return alive[0], (used + len(alive) * (T - prev)) / (K * T)


def f3(sub):
    alive, used, prev = list(range(K)), 0, 0
    for h in (250, 500, 750):
        used += len(alive) * (h - prev)
        prev = h
        x = L[:, sub, J[h]]
        lead = min(alive, key=lambda i: x[i].mean())
        dm = lambda i: x[i] - x[lead]
        alive = [
            i
            for i in alive
            if i == lead or dm(i).mean() - TCRIT * dm(i).std(ddof=1) / math.sqrt(N) <= 0
        ]
    used += len(alive) * (T - prev)
    return min(alive, key=lambda i: L[i, sub, -1].mean()), used / (K * T)


def full(sub):
    return int(np.argmin(L[:, sub, -1].mean(1))), 1.0


subs = [list(s) for s in itertools.combinations(range(len(S)), N)]
sel = {
    f"SH{r}": (lambda sub, r=r: sh(sub, r))
    for r in itertools.combinations((250, 500, 750, 1000), 3)
}
sel["FULL"] = full
sel["F3"] = f3
res = {}
for name, fn in sel.items():
    out = [fn(sub) for sub in subs]
    res[name] = (
        np.mean([o[1] for o in out]),
        np.mean([truth[o[0]] - best for o in out]),
        np.mean([truth[o[0]] == best for o in out]),
    )

print(
    f"{path}: {K} configs, {len(S)} seeds, {len(subs)} subsets of {N}; t-crit {TCRIT:.2f}"
)
print("truth (all-seed mean final):", dict(zip(map(str, confs), np.round(truth, 4))))
print("seed SD of final:", np.round(L[:, :, -1].std(1, ddof=1), 4))
print(f"random-pick regret: {truth.mean() - best:.4f}")
j = J[250]
pa = [
    (L[a, :, j] - L[b, :, j]).std(ddof=1)
    for a, b in itertools.combinations(range(K), 2)
]
un = [
    math.sqrt(L[a, :, j].var(ddof=1) + L[b, :, j].var(ddof=1))
    for a, b in itertools.combinations(range(K), 2)
]
print(
    f"SD of config differences at step 250, median over pairs: paired {np.median(pa):.4f}, unpaired {np.median(un):.4f}"
)
print(f"\n{'selector':<22}{'compute':>9}{'regret':>9}{'P(best)':>9}")
for name, (c, r, p) in sorted(res.items(), key=lambda kv: kv[1][0]):
    print(f"{name:<22}{c:>9.3f}{r:>9.4f}{p:>9.2f}")
fc, fr, _ = res["F3"]
dom = [
    n
    for n, (c, r, _) in res.items()
    if n != "F3" and c <= fc + 1e-12 and r <= fr + 1e-12
]
print(
    f"\nF3 dominated by: {dom or 'none'}  ->  F3 {'NOT supported' if dom else 'SUPPORTED'} by the pre-registered rule"
)
