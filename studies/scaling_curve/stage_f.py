"""Stage F of the scaling study, the gate test: how much of an automated tuning loop's "progress" is real?

A scripted proposer hill-climbs the 0.88M model's learning rate and weight decay two ways on the same number of
training runs: keep any candidate that scores better on one run (what an agent loop does), or keep only what a
replicated paired comparison calls real (Regora's verdict). Each loop's claimed gain is then set against the
true gain of its final configuration on fresh data orders. Plan and decision rules: PLAN_F.md. `study.py` is
reused unchanged.

    python stage_f.py run        # every replicate (resumable), then commit the results
    python stage_f.py analyze    # the verdict of each rule
    python stage_f.py smoke      # mechanics check on the CPU with a tiny configuration
"""

import json
import math
import os
import random
import subprocess
import sys

SMOKE = sys.argv[1:2] == ["smoke"]
if SMOKE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # leave the GPU to the running study

import numpy as np
from scipy import stats

import study

NAME = "gate_test"
CFG = dict(
    size="s1",
    lr=4e-3,
    ckpt_step=600,
    horizon=200,
    replicates=8,
    gated_rounds=12,
    pairs=4,
    truth_orders=12,
)
BASE = (1.0, 0.0)  # a configuration is (learning-rate multiplier, weight decay)
LR_MOVES = (0.8, 0.9, 1.1, 1.25)
LR_RANGE = (0.25, 4.0)
WD_VALUES = (0.0, 0.01, 0.03, 0.1)
NULL_SWAP = (
    1 / 32
)  # a "null" candidate is the incumbent with 1 example in 32 swapped: no systematic effect


def propose(rng, cfg):
    """The next candidate: 20% a null, 40% a learning-rate move, 40% a weight-decay move."""
    u = rng.random()
    if u < 0.2:
        return "null", cfg
    if u < 0.6:
        cand = (
            min(max(cfg[0] * rng.choice(LR_MOVES), LR_RANGE[0]), LR_RANGE[1]),
            cfg[1],
        )
        return ("lr", cand) if cand != cfg else ("null", cfg)
    return "wd", (cfg[0], rng.choice([v for v in WD_VALUES if v != cfg[1]]))


class Lab:
    """One replicate: a checkpoint trained at the base configuration, and branches from it."""

    def __init__(self, c, rep, train, val):
        study.set_kernels(True)
        self.c = c
        self.r = study.Runner(study.cell(size=c["size"]), train, val)
        self.ck, _ = self.r.train(
            self.r.new_state(1000 + rep), 2000 + rep, c["ckpt_step"], c["lr"]
        )
        self.runs = 0

    def loss(self, cfg, seed, null=False):
        self.runs += 1
        h = self.c["horizon"]
        _, rd = self.r.train(
            self.ck,
            seed,
            h,
            self.c["lr"],
            lr_mult=cfg[0],
            wd=cfg[1],
            swap=NULL_SWAP if null else 0.0,
            eval_at=(h,),
        )
        return rd[h]


def naive_loop(lab, rep, rounds):
    """Keep a candidate when its single run beats the best single score seen so far."""
    rng, cfg, seed0 = random.Random(rep), BASE, 10_000 + rep * 1000
    first = best = lab.loss(cfg, seed0 + 999)
    log = []
    for i in range(rounds):
        kind, cand = propose(rng, cfg)
        score = lab.loss(cand, seed0 + i, null=kind == "null")
        keep = score < best
        log.append(dict(kind=kind, cand=cand, score=score, keep=keep))
        if keep:
            best, cfg = score, cand
    return dict(final=cfg, claimed=first - best, log=log)


def gated_loop(lab, rep, rounds, n):
    """Keep a candidate when the 95% t interval of its paired difference on n fresh orders is below zero."""
    rng, cfg, claimed = random.Random(rep), BASE, 0.0
    t = float(stats.t.ppf(0.975, n - 1))
    log = []
    for j in range(rounds):
        kind, cand = propose(rng, cfg)
        seeds = [20_000 + rep * 1000 + j * 10 + k for k in range(n)]
        d = [lab.loss(cand, s, null=kind == "null") - lab.loss(cfg, s) for s in seeds]
        mean, half = float(np.mean(d)), t * float(np.std(d, ddof=1)) / math.sqrt(n)
        keep = bool(mean + half < 0)
        log.append(dict(kind=kind, cand=cand, mean=mean, half=half, keep=keep))
        if keep:
            claimed, cfg = claimed - mean, cand
    return dict(final=cfg, claimed=claimed, log=log)


def replicate(c, rep, train, val):
    lab = Lab(c, rep, train, val)
    naive = naive_loop(lab, rep, c["gated_rounds"] * 2 * c["pairs"])
    naive["runs"] = lab.runs
    gated = gated_loop(lab, rep, c["gated_rounds"], c["pairs"])
    gated["runs"] = lab.runs - naive["runs"]
    seeds = [30_000 + rep * 100 + k for k in range(c["truth_orders"])]
    base = [lab.loss(BASE, s) for s in seeds]
    for loop in (
        naive,
        gated,
    ):  # true gain: base loss minus the final configuration's, on orders no loop saw
        final = tuple(loop["final"])
        loop["true_diffs"] = (
            [0.0] * len(seeds)
            if final == BASE
            else [b - lab.loss(final, s) for b, s in zip(base, seeds)]
        )
        loop["true"] = float(np.mean(loop["true_diffs"]))
    return dict(naive=naive, gated=gated)


def run(c=CFG):
    path = os.path.join(study.OUT_DIR, NAME + ".json")
    os.makedirs(study.OUT_DIR, exist_ok=True)
    out = (
        json.load(open(path)) if os.path.exists(path) else dict(config=c, replicates={})
    )
    train, val, sha = study.load_corpus()
    out["config"]["corpus_sha256"] = sha
    for rep in range(c["replicates"]):
        if str(rep) in out["replicates"]:
            continue
        out["replicates"][str(rep)] = replicate(c, rep, train, val)
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        r = out["replicates"][str(rep)]
        print(
            f"replicate {rep}: naive claimed {r['naive']['claimed']:+.4f} true {r['naive']['true']:+.4f} | "
            f"gated claimed {r['gated']['claimed']:+.4f} true {r['gated']['true']:+.4f}",
            flush=True,
        )
    out["done"] = True
    with open(path, "w") as f:
        json.dump(out, f)


def interval(x):
    """Mean and 95% t half-width over replicates."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(
        stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    )


def analyze():
    path = os.path.join(study.OUT_DIR, NAME + ".json")
    if not os.path.exists(path):
        print("not run")
        return
    out = json.load(open(path))
    reps = [out["replicates"][k] for k in sorted(out["replicates"], key=int)]
    print(
        f"{len(reps)} replicates{'' if out.get('done') else ' (NOT FINISHED)'}; config {out['config']}"
    )
    for name in ("naive", "gated"):
        for r in reps:
            L = r[name]
            kept = [e for e in L["log"] if e["keep"]]
            print(
                f"  {name:5} runs {L['runs']:3d} rounds {len(L['log']):3d} kept {len(kept):2d} "
                f"(nulls {sum(e['kind'] == 'null' for e in kept)}) final {tuple(L['final'])} "
                f"claimed {L['claimed']:+.4f} true {L['true']:+.4f}"
            )
    if not all(np.isfinite([r[n]["true"] for r in reps for n in ("naive", "gated")])):
        print("a final configuration diverged on a fresh order -> rules not decided")
        return
    over = {
        n: [r[n]["claimed"] - r[n]["true"] for r in reps] for n in ("naive", "gated")
    }
    (mn, hn), (mg, hg) = interval(over["naive"]), interval(over["gated"])
    f1 = mn - hn > 0
    print(
        f"\n##### F1: the keep-if-better loop claims more than it achieved\n  claimed - true: {mn:+.4f} +/- {hn:.4f}"
        f"  (mean claimed {np.mean([r['naive']['claimed'] for r in reps]):+.4f}, mean true "
        f"{np.mean([r['naive']['true'] for r in reps]):+.4f}) -> {'SUPPORTED' if f1 else 'NOT SUPPORTED'}"
    )
    verdict = (
        "UNINFORMATIVE (F1 not supported)"
        if not f1
        else "PASS" if abs(mg) <= mn / 2 else "FAIL"
    )
    print(
        f"\n##### F2: the gated loop's overclaim is at most half of that\n  claimed - true: {mg:+.4f} +/- {hg:.4f}"
        f"  (mean claimed {np.mean([r['gated']['claimed'] for r in reps]):+.4f}, mean true "
        f"{np.mean([r['gated']['true'] for r in reps]):+.4f}) -> {verdict}"
    )
    print("\n##### F3: the gate keeps at most 5% of null candidates")
    for name in ("naive", "gated"):
        nulls = [e for r in reps for e in r[name]["log"] if e["kind"] == "null"]
        kept = sum(e["keep"] for e in nulls)
        rule = (
            ""
            if name == "naive"
            else f" -> {'PASS' if nulls and kept / len(nulls) <= 0.05 else 'FAIL' if nulls else 'no nulls'}"
        )
        print(f"  {name:5} kept {kept} of {len(nulls)} null candidates{rule}")
    md, hd = interval([r["gated"]["true"] - r["naive"]["true"] for r in reps])
    verdict = (
        "gated better"
        if md - hd > 0
        else "naive better" if md + hd < 0 else "UNRESOLVED"
    )
    print(
        f"\n##### F4: which loop ends at the better configuration on the same number of runs\n"
        f"  true gain, gated - naive: {md:+.4f} +/- {hd:.4f} -> {verdict}"
    )


def smoke():
    import shutil
    import tempfile

    assert study.DEV == "cpu"
    study.OUT_DIR = tempfile.mkdtemp(prefix="smoke_f_")
    c = dict(
        size="s0",
        lr=4e-3,
        ckpt_step=20,
        horizon=5,
        replicates=2,
        gated_rounds=3,
        pairs=2,
        truth_orders=2,
    )
    run(c)
    out = json.load(open(os.path.join(study.OUT_DIR, NAME + ".json")))
    for r in out["replicates"].values():
        assert (
            r["naive"]["runs"] == 3 * 2 * 2 + 1 and r["gated"]["runs"] == 3 * 2 * 2
        ), (r["naive"]["runs"], r["gated"]["runs"])
        assert len(r["naive"]["log"]) == 12 and len(r["gated"]["log"]) == 3
        assert [e["kind"] for e in r["naive"]["log"][:1]] == [
            e["kind"] for e in r["gated"]["log"][:1]
        ], "proposal streams differ"
        assert all(math.isfinite(e["score"]) for e in r["naive"]["log"])
    train, val, _ = study.load_corpus()
    lab = Lab(c, 0, train, val)
    assert lab.loss(BASE, 7) == lab.loss(BASE, 7), "replay is not exact"
    assert lab.loss(BASE, 7) != lab.loss(BASE, 7, null=True) and lab.loss(
        BASE, 7
    ) != lab.loss((0.8, 0.0), 7)
    analyze()
    shutil.rmtree(study.OUT_DIR)
    print("\nSMOKE OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
        subprocess.run(["git", "add", "-A", "results"], cwd=study.HERE)
        subprocess.run(
            ["git", "commit", "-q", "-m", "Stage F raw results"], cwd=study.HERE
        )
    elif cmd == "analyze":
        analyze()
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit(__doc__)
