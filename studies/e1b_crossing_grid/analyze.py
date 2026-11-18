import json, itertools, math, numpy as np
from scipy.stats import kendalltau

d = json.load(open("results/e1b/e1b.json"))
cfg = d["config"]
C = d["curves"]
confs = list(itertools.product(cfg["lrs"], cfg["warms"]))
S = cfg["seeds"]
steps = [s for s, _ in C[f"{confs[0][0]}|{confs[0][1]}|0"]]


def cv(c, s):
    return np.array(
        [v if math.isfinite(v) else np.inf for _, v in C[f"{c[0]}|{c[1]}|{s}"]]
    )


M = np.array([np.mean([cv(c, s) for s in S], 0) for c in confs])
fin = M[:, -1]
print("configs (lr, warmup):", confs)
print("seed-mean final:", np.round(fin, 4))
print(
    "seed SD of final:",
    [round(float(np.std([cv(c, s)[-1] for s in S], ddof=1)), 4) for c in confs],
)
print(
    "non-finite runs:",
    sum(1 for k, v in C.items() if any(not math.isfinite(x) for _, x in v)),
)
cross = 0
for a, b in itertools.combinations(range(len(confs)), 2):
    fo = fin[a] < fin[b]
    if any((M[a, j] < M[b, j]) != fo for j in range(len(steps) - 1)):
        cross += 1
print(f"crossing pairs: {cross} of {len(confs)*(len(confs)-1)//2}")
for h in (250, 350, 500, 750):
    j = steps.index(h)
    fm = np.where(np.isfinite(M[:, j]), M[:, j], 1e9)
    tau = kendalltau(fm, fin).statistic
    pick = int(np.argmin(fm))
    print(
        f"h={h}: tau {tau:+.2f} regret {fin[pick]-fin.min():.4f} picked {confs[pick]}"
    )
print(
    "\nsuccessive halving (8->4@250->2@500->1@750) per seed, ranked by that seed only"
)
regs = []
for s in S:
    alive = list(range(len(confs)))
    for h, keep in ((250, 4), (500, 2), (750, 1)):
        j = steps.index(h)
        alive = sorted(alive, key=lambda i: cv(confs[i], s)[j])[:keep]
    r = fin[alive[0]] - fin.min()
    regs.append(r)
    print(f"  seed {s}: pick {confs[alive[0]]} regret {r:.4f}")
print(
    f"mean regret {np.mean(regs):.4f}; compute fraction {(8*250+4*250+2*250+250)/(8*1000):.3f}"
)
