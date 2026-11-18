"""Analysis for the scaling study. Reads results/<cell>.json, prints the tables and the verdicts of PLAN.md.

python analyze.py            # everything that has finished
python analyze.py predict    # freeze the stage-B predictions from stage A (writes results/prediction.json)
python analyze.py json       # the same statistics as one JSON document (for charts)
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
EPS = {"lr1.01": 0.01, "lr1.03": 0.03, "lr1.1": 0.1, "lr1.3": 0.3, "lr2.0": 1.0}
SMALL = ("lr1.01", "lr1.03", "lr1.1")
A_CELLS, B_CELLS = ("A_s0", "A_s1", "A_s2"), ("B_s3", "B_s4")
H_MAIN = 200


def load(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d.get("done") else None


def table(d, arm):
    """(K, S, H) array of val losses for an arm."""
    c = d["config"]
    return np.array(
        [
            [d["branches"][f"{k}|{s}|{arm}"] for s in range(c["orders"])]
            for k in range(c["ckpts"])
        ],
        dtype=float,
    )


def stats(ctrl, arm):
    """ctrl, arm: (K, S) losses. Variances are across data orders within a checkpoint, averaged over checkpoints
    (E11's pooling). A diverged branch (non-finite loss) makes the whole row undefined, and says so.
    """
    if not (np.isfinite(ctrl).all() and np.isfinite(arm).all()):
        return dict(diverged=True)
    K, S = ctrl.shape
    d = arm - ctrl
    v_p = d.var(axis=1, ddof=1).mean()
    va, vc = arm.var(axis=1, ddof=1).mean(), ctrl.var(axis=1, ddof=1).mean()
    cov = np.mean([np.cov(arm[k], ctrl[k], ddof=1)[0, 1] for k in range(K)])
    delta = d.mean()
    # Checkpoint-to-checkpoint spread of the effect, beyond what order noise explains (one-way ANOVA, floored at 0).
    v_between = (
        max(0.0, d.mean(axis=1).var(ddof=1) - v_p / S) if K > 1 else float("nan")
    )
    return dict(
        diverged=False,
        effect=float(delta),
        sd_paired=math.sqrt(v_p),
        sd_arm=math.sqrt(va),
        sd_ctrl=math.sqrt(vc),
        sd_between=math.sqrt(v_between),
        saved=float((va + vc) / v_p) if v_p > 0 else math.inf,
        rho=float(cov / math.sqrt(va * vc)) if va * vc > 0 else float("nan"),
        kappa=math.sqrt(v_p) / abs(delta) if delta != 0 else math.inf,
        resolved=bool(abs(delta) > 2 * math.sqrt(v_p / d.size)),
        # Runs per arm for 80% power at two-sided 5%, normal approximation (E0's formula); floor of 2.
        n_paired=max(2.0, 7.85 * v_p / delta**2) if delta != 0 else math.inf,
        n_unpaired=max(2.0, 7.85 * (va + vc) / delta**2) if delta != 0 else math.inf,
        per_ckpt_effect=[float(x) for x in d.mean(axis=1)],
    )


def saved_interval(ctrl, arm, rng, reps=1000):
    """Bootstrap 95% interval for seeds saved: resample checkpoints, then orders within each."""
    if not (np.isfinite(ctrl).all() and np.isfinite(arm).all()):
        return float("nan"), float("nan")
    K, S = ctrl.shape
    vals = []
    for _ in range(reps):
        ks = rng.integers(0, K, K)
        ss = rng.integers(0, S, (K, S))
        c = np.take_along_axis(ctrl[ks], ss, 1)
        a = np.take_along_axis(arm[ks], ss, 1)
        v_p = (a - c).var(axis=1, ddof=1).mean()
        if v_p > 0:
            vals.append(
                (a.var(axis=1, ddof=1).mean() + c.var(axis=1, ddof=1).mean()) / v_p
            )
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"),) * 2
    return float(lo), float(hi)


def cell_rows(name):
    d = load(name)
    if d is None:
        return None
    c = d["config"]
    hs = list(c["horizons"])
    ctrl = table(d, "control")
    arms = [
        a
        for a in dict.fromkeys(k.split("|")[2] for k in d["branches"])
        if a not in ("control", "twin")
    ]
    rng = np.random.default_rng(0)
    rows = {}
    for a in arms:
        t = table(d, a)
        for i, h in enumerate(hs):
            r = stats(ctrl[:, :, i], t[:, :, i])
            r["lo"], r["hi"] = saved_interval(ctrl[:, :, i], t[:, :, i], rng)
            rows[(a, h)] = r
    twins = [
        float(np.abs(np.array(v) - np.array(d["branches"][k[:-4] + "control"])).max())
        for k, v in d["branches"].items()
        if k.endswith("|twin")
    ]
    twin_sd = {}
    if c["twin"] == "all":
        tw = table(d, "twin")
        for i, h in enumerate(hs):
            twin_sd[h] = math.sqrt(((tw[:, :, i] - ctrl[:, :, i]) ** 2).mean() / 2)
    # Does a branch's paired difference keep its sign from each horizon to the last one?
    sign_keep = {}
    for a in arms:
        dd = table(d, a) - ctrl
        if np.isfinite(dd).all():
            sign_keep[a] = [
                float((np.sign(dd[:, :, i]) == np.sign(dd[:, :, -1])).mean())
                for i in range(len(hs))
            ]
    return dict(
        config=c,
        lr=d["lr"],
        tune=d["tune"],
        rows=rows,
        arms=arms,
        hs=hs,
        wall=d.get("wall_s", 0),
        twin_max=max(twins) if twins else float("nan"),
        twin_sd=twin_sd,
        sign_keep=sign_keep,
        ctrl_sd={
            h: math.sqrt(ctrl[:, :, i].var(axis=1, ddof=1).mean())
            for i, h in enumerate(hs)
        },
        ckpt_loss=d.get("ckpt_loss", {}),
    )


def show(name, R):
    c = R["config"]
    edge = R["lr"] in (min(c["lr_grid"]), max(c["lr_grid"]))
    print(
        f"\n=== {name}: {c['size']} {c['params']:,} params, batch {c['batch']}, train bytes {c['train_len']:,}, "
        f"ckpt {c['ckpt_step']}, K={c['ckpts']} S={c['orders']}, exact={c['exact']}, amp={c['amp']}, "
        f"lr {R['lr']:g}{' (AT GRID EDGE)' if edge else ''}, {R['wall']/60:.1f} min"
    )
    print("    tune: " + ", ".join(f"{k}: {v:.4f}" for k, v in R["tune"].items()))
    print(
        f"    identical-twin max |diff|: {R['twin_max']:.3g}   control SD across orders: "
        + ", ".join(f"h{h}: {v:.4f}" for h, v in R["ctrl_sd"].items())
    )
    if R["twin_sd"]:
        print(
            "    non-determinism SD (twin): "
            + ", ".join(f"h{h}: {v:.6f}" for h, v in R["twin_sd"].items())
        )
    print(
        f"    {'arm':8} {'h':>5} {'effect':>10} {'sd_paired':>10} {'sd_betw':>9} {'rho':>6} {'saved':>9} "
        f"{'95% boot':>20} {'kappa':>7} {'n_pair':>7} {'n_unp':>9} res"
    )
    for a in R["arms"]:
        for h in R["hs"]:
            r = R["rows"][(a, h)]
            if r["diverged"]:
                print(f"    {a:8} {h:5d}  DIVERGED")
                continue
            print(
                f"    {a:8} {h:5d} {r['effect']:+10.5f} {r['sd_paired']:10.6f} {r['sd_between']:9.5f} "
                f"{r['rho']:6.3f} {r['saved']:9.1f} [{r['lo']:8.1f},{r['hi']:9.1f}] {r['kappa']:7.2f} "
                f"{r['n_paired']:7.1f} {r['n_unpaired']:9.1f} {'yes' if r['resolved'] else 'no'}"
            )
    print(
        "    share of branches whose paired difference has the sign it ends with, by horizon:"
    )
    for a, v in R["sign_keep"].items():
        print(f"      {a:8} " + "  ".join(f"h{h}: {x:.2f}" for h, x in zip(R["hs"], v)))


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def saved_at(R, arm, h=H_MAIN):
    r = R["rows"].get((arm, h))
    return None if r is None or r["diverged"] else r["saved"]


def fit_predict(R):
    """Per arm: log(saved) = a + b log(params) on the stage-A cells at h=200."""
    if any(n not in R for n in A_CELLS):
        return None
    fits = {}
    for a in R["A_s0"]["arms"]:
        P = [R[n]["config"]["params"] for n in A_CELLS]
        S = [saved_at(R[n], a) for n in A_CELLS]
        if any(s is None or not math.isfinite(s) for s in S):
            fits[a] = None
            continue
        b, a0 = np.polyfit(np.log(P), np.log(S), 1)
        fits[a] = dict(a=float(a0), b=float(b), points=dict(zip(map(str, P), S)))
    return fits


def verdicts(R):
    print(
        "\n\n##### H1: seeds saved falls as (change size)^-2 for small LR changes, h=200"
    )
    for n in A_CELLS:
        if n in R:
            eps = [EPS[a] for a in SMALL]
            sv = [saved_at(R[n], a) for a in SMALL]
            sd = [R[n]["rows"][(a, H_MAIN)]["sd_paired"] for a in SMALL]
            s1, s2 = slope(eps, sv), slope(eps, sd)
            print(
                f"  {n}: slope of log saved {s1:+.2f} (rule -2.4..-1.6: {'PASS' if -2.4 <= s1 <= -1.6 else 'FAIL'}); "
                f"slope of log paired SD {s2:+.2f}; saved = "
                + ", ".join(f"{x:.0f}" for x in sv)
            )

    print("\n##### H1b: the same for weight decay 0.01 -> 0.1, h=200")
    for n in A_CELLS:
        if n in R:
            sv = [saved_at(R[n], a) for a in ("wd0.01", "wd0.1", "wd1.0")]
            s1 = slope([0.01, 0.1], sv[:2])
            print(
                f"  {n}: slope {s1:+.2f} (rule -2.4..-1.6: {'PASS' if -2.4 <= s1 <= -1.6 else 'FAIL'}); "
                f"saved = "
                + ", ".join("diverged" if x is None else f"{x:.0f}" for x in sv)
            )

    print(
        "\n##### H2: seeds saved at h=400 below h=50, every arm except swap32, stage A"
    )
    tot = ok = 0
    for n in A_CELLS:
        if n in R:
            for a in R[n]["arms"]:
                s50, s400 = saved_at(R[n], a, 50), saved_at(R[n], a, 400)
                if a == "swap32" or s50 is None or s400 is None:
                    continue
                tot += 1
                ok += s400 < s50
                print(
                    f"  {n} {a:8} h50 {s50:10.1f}  h400 {s400:10.1f}  {'falls' if s400 < s50 else 'RISES'}"
                )
    if tot:
        print(f"  {ok} of {tot} fall (rule: all)")

    p = os.path.join(OUT, "prediction.json")
    if os.path.exists(p):
        fits = json.load(open(p))
        print(
            "\n##### H3: stage-B seeds saved predicted from stage A (frozen fit), h=200"
        )
        tot = ok = 0
        for n in B_CELLS:
            if n in R:
                P = R[n]["config"]["params"]
                for a in R[n]["arms"]:
                    got = saved_at(R[n], a)
                    if fits.get(a) is None or got is None:
                        print(
                            f"  {n} {a:8} no prediction or diverged; counted as a miss"
                        )
                        tot += 1
                        continue
                    pred = math.exp(fits[a]["a"] + fits[a]["b"] * math.log(P))
                    hit = 0.5 <= got / pred <= 2
                    tot += 1
                    ok += hit
                    print(
                        f"  {n} {a:8} predicted {pred:10.1f} measured {got:10.1f} ratio {got/pred:6.2f} "
                        f"{'within 2x' if hit else 'MISS'}  (slope b {fits[a]['b']:+.2f})"
                    )
        if tot:
            print(
                f"  {ok} of {tot} within a factor of 2 (rule: at least 75%: "
                f"{'PASS' if ok / tot >= 0.75 else 'FAIL'})"
            )

    if "C_default_kernels" in R and "A_s1" in R:
        print(
            "\n##### H4: default kernels cap the saving for the smallest change, h=200"
        )
        D, E = R["C_default_kernels"], R["A_s1"]
        print(
            f"  identical-twin max |diff| under default kernels: {D['twin_max']:.3g}"
            f"{'  -> twins identical, test uninformative' if D['twin_max'] == 0 else ''}"
        )
        for a in ("lr1.01", "lr2.0"):
            e, dd = saved_at(E, a), saved_at(D, a)
            print(f"  {a}: deterministic {e:.1f}  default {dd:.1f}  ratio {dd/e:.2f}")

    print(
        "\n##### H5: swapping training examples erodes pairing; a fully different order saves nothing, h=200"
    )
    for n in A_CELLS:
        if n in R:
            s1, s8, s32 = (saved_at(R[n], a) for a in ("swap1", "swap8", "swap32"))
            ok = s1 > s8 > s32 and 0.5 <= s32 <= 2
            print(
                f"  {n}: swap1 {s1:.1f}  swap8 {s8:.1f}  swap32 {s32:.2f}  (rule: ordered and swap32 in 0.5..2: "
                f"{'PASS' if ok else 'FAIL'})"
            )


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    names = sorted(
        f[:-5] for f in os.listdir(OUT) if f.endswith(".json") and f[0] in "ABC"
    )
    R = {n: cell_rows(n) for n in names}
    R = {n: v for n, v in R.items() if v}
    if cmd == "predict":
        fits = fit_predict(R)
        assert fits, "stage A not finished"
        p = os.path.join(OUT, "prediction.json")
        assert not os.path.exists(p), "prediction already frozen"
        with open(p, "w") as f:
            json.dump(fits, f, indent=1)
        print(json.dumps(fits, indent=1))
        return
    if cmd == "json":
        doc = {
            n: {
                **{
                    k: v[k]
                    for k in (
                        "config",
                        "lr",
                        "tune",
                        "arms",
                        "hs",
                        "wall",
                        "twin_max",
                        "twin_sd",
                        "sign_keep",
                        "ctrl_sd",
                    )
                },
                "rows": {f"{a}|{h}": r for (a, h), r in v["rows"].items()},
            }
            for n, v in R.items()
        }
        print(json.dumps(doc))
        return
    for n, v in R.items():
        show(n, v)
    verdicts(R)


if __name__ == "__main__":
    main()
