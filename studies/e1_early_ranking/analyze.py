import json, itertools, numpy as np
from scipy.stats import kendalltau

d = json.load(open("results/e1/e1.json"))
cfg = d["config"]
C = d["curves"]
steps = [s for s, _ in C[next(iter(C))]]


def mean_curve(lr, wd):
    return np.mean([[v for _, v in C[f"{lr}|{wd}|{s}"]] for s in cfg["seeds"]], 0)


def sd_final(lr, wd):
    return np.std([C[f"{lr}|{wd}|{s}"][-1][1] for s in cfg["seeds"]], ddof=1)


def report(name, confs):
    M = np.array([mean_curve(*c) for c in confs])
    fin = M[:, -1]
    print(
        f"\n{name}: configs {confs}\n  final means {np.round(fin, 4)}; seed SD of final {[round(float(sd_final(*c)), 4) for c in confs]}"
    )
    for h in (100, 200, 300, 400):
        j = steps.index(h)
        tau = kendalltau(M[:, j], fin).statistic
        pick = int(np.argmin(M[:, j]))
        print(
            f"  h={h}: tau {tau:+.2f}  regret {fin[pick]-fin.min():.4f}  picked {confs[pick]}"
        )


all6 = list(itertools.product(cfg["lrs"], cfg["wds"]))
report("6-config grid", all6)
report("LR-only (wd=0)", [(lr, 0.0) for lr in cfg["lrs"]])
