"""Stage H of the scaling study: paired data tests by re-weighting. Does the pairing survive, and is the answer right?

Stage G made data changes by swapping examples into each batch, which breaks the pairing: the two runs no longer
see the same batches. This stage makes the same changes by re-weighting the loss. Both runs see the control's
batches, and the change gives each example the importance weight that turns the control's sampling into the
swapped mixture in expectation. It measures how much cheaper that makes the decision and whether the re-weighted
test estimates the swapped change's effect. Plan and decision rules: PLAN_H.md. Stage G's results are the
comparison; `study.py` and `stage_g.py` are reused unchanged.

    python stage_h.py run        # the cell (resumable per checkpoint), then commit the results
    python stage_h.py analyze    # the replication check, the tables and the verdict of each rule
    python stage_h.py smoke      # mechanics check on the CPU with a tiny configuration; reads no real outcome
"""

import json
import math
import os
import subprocess
import sys
import time

SMOKE = sys.argv[1:2] == ["smoke"]
if SMOKE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # leave the GPU to the running study

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sstats

import analyze
import stage_g
import study
from stage_g import table

NAME = "H_s2_reweight"
CFG = {
    **{
        k: stage_g.CFG[k]
        for k in ("size", "lr", "batch", "ckpt_step", "ckpts", "orders", "anneals")
    },
    "const": 200,
    "const_reads": (100, 200),
}
# arm -> the stage-G arm whose mixture it re-weights to. The control is stage G's, re-run.
ARMS = {"control": None, **{a + "w": a for a in stage_g.DATA_ARMS}}


def make_members(spans, n_train):
    """subset -> (bool mask of training positions inside its files, pi_S: share of the control's start positions
    inside it). `Runner._indices` draws starts from range(n_train - CTX - 1)."""
    n_starts = n_train - study.CTX - 1
    out = {}
    for name, pred in stage_g.SUBSETS.items():
        mask = torch.zeros(n_train, dtype=torch.bool)
        for p, s, e in spans:
            if s < n_train and pred(p):
                mask[s : min(e, n_train)] = True
        out[name] = (mask, float(mask[:n_starts].double().mean()))
    return out


class HRunner(stage_g.GRunner):
    """Stage G's runner, plus re-weighted arms: the control's batch, with per-example loss weights.

    The weights belong to the batch just drawn and are consumed by the next loss, so evaluation is unweighted.
    """

    def __init__(self, c, train, val, subsets, members):
        super().__init__(c, train, val, subsets)
        self.members, self.rw, self._w = members, None, None

    def weights(self, data_seed, step):
        subset, m = stage_g.ARMS[self.rw]
        mask, share = self.members[subset]
        f = m / self.c["batch"]
        return (1 - f) + f * mask[self._indices(data_seed, step)].float() / share

    def batch(self, data_seed, step, swap=0.0):
        if self.rw is not None:
            self._w = self.weights(data_seed, step).to(study.DEV)
        return super().batch(
            data_seed, step, swap
        )  # self.arm stays None: the control's examples

    def loss(self, x, y):
        w, self._w = self._w, None
        if w is None:
            return super().loss(x, y)
        per_token = F.cross_entropy(
            self.model(x).flatten(0, -2).float(), y.flatten(), reduction="none"
        )
        return (per_token.view(y.shape).mean(1) * w).mean()


def run(c=CFG):
    path = os.path.join(study.OUT_DIR, NAME + ".json")
    os.makedirs(study.OUT_DIR, exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    if out.get("done"):
        print(NAME, "already done", flush=True)
        return
    study.set_kernels(True)
    train, val, sha, spans = stage_g.load_corpus_spans()
    subsets = stage_g.make_subsets(spans, len(train))
    members = make_members(spans, len(train))
    r = HRunner(c, train, val, subsets, members)
    out.setdefault(
        "config",
        {
            **c,
            "params": r.params,
            "device": study.DEV,
            "torch": torch.__version__,
            "corpus_sha256": sha,
            "arms": ARMS,
            "pi": {k: v[1] for k, v in members.items()},
        },
    )
    out.setdefault("branches", {})
    t0 = time.time()
    for k in range(c["ckpts"]):
        if all(
            f"{k}|{s}|{a}" in out["branches"] for s in range(c["orders"]) for a in ARMS
        ):
            continue
        r.rw = None
        ck, rd = r.train(
            r.new_state(1000 + k),
            2000 + k,
            c["ckpt_step"],
            c["lr"],
            eval_at=(c["ckpt_step"],),
        )
        out.setdefault("ckpt_loss", {})[str(k)] = rd[c["ckpt_step"]]
        for s in range(c["orders"]):
            seed = 100 * (k + 1) + s
            for a, base in ARMS.items():
                r.rw = base
                anneal = {
                    str(L): r.branch(ck, seed, L, stage_g.anneal_lr(c["lr"], L), (L,))[
                        L
                    ]
                    for L in c["anneals"]
                }
                const = r.branch(
                    ck, seed, c["const"], lambda _: c["lr"], c["const_reads"]
                )
                out["branches"][f"{k}|{s}|{a}"] = dict(
                    anneal=anneal, const={str(h): v for h, v in const.items()}
                )
        r.rw = None
        out["wall_s"] = out.get("wall_s", 0) + time.time() - t0
        t0 = time.time()
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        print(f"{NAME} checkpoint {k} done ({out['wall_s']:.0f}s so far)", flush=True)
    out["done"] = True
    with open(path, "w") as f:
        json.dump(out, f)


# ---------------------------------------------------------------------------------------------------- analysis


def replication(h, g):
    """Largest |difference| between this stage's control and stage G's, over the checkpoint losses and every
    shared reading. Zero means the control was reproduced bit for bit."""
    c = h["config"]
    diffs = [abs(h["ckpt_loss"][k] - g["ckpt_loss"][k]) for k in h["ckpt_loss"]]
    for k in range(c["ckpts"]):
        for s in range(c["orders"]):
            hb, gb = (
                h["branches"][f"{k}|{s}|control"],
                g["branches"][f"{k}|{s}|control"],
            )
            diffs += [abs(hb["anneal"][L] - gb["anneal"][L]) for L in hb["anneal"]]
            diffs += [abs(hb["const"][x] - gb["long"][x]) for x in hb["const"]]
    return max(diffs)


def ratios(h, g, h_reading, g_reading):
    """Per signal arm: (re-weighted stats, swapped stats, runs ratio swapped/re-weighted, effect ratio rw/sw)."""
    out = {}
    for a in stage_g.SIGNAL_ARMS:
        rw = analyze.stats(
            table(h, "control", *h_reading), table(h, a + "w", *h_reading)
        )
        sw = analyze.stats(table(g, "control", *g_reading), table(g, a, *g_reading))
        out[a] = (rw, sw, (sw["kappa"] / rw["kappa"]) ** 2, rw["effect"] / sw["effect"])
    return out


def verdicts(res):
    """H1 and H2 of PLAN_H.md from `ratios` output."""
    runs = stage_g.geomean([v[2] for v in res.values()])
    h1 = "SUPPORTED" if runs >= 4 else "REJECTED" if runs < 2 else "INCONCLUSIVE"
    eff = [v[3] for v in res.values()]
    outside = sum(not 0.5 <= e <= 2 for e in eff)
    h2 = (
        "REJECTED"
        if min(eff) <= 0 or outside >= 3
        else "SUPPORTED" if outside == 0 else "INCONCLUSIVE"
    )
    return runs, h1, h2


def spearman_vs_truth(g, d_proxy, arm_of, proxy, truth, a_orders, b_orders):
    """Spearman over the signal arms between the proxy's mean effect on orders `a_orders` (arm names mapped by
    `arm_of`) and stage G's truth on the disjoint orders `b_orders`."""

    def effects(d, reading, orders, name):
        ctrl = table(d, "control", *reading, orders)
        return [
            float((table(d, name(a), *reading, orders) - ctrl).mean())
            for a in stage_g.SIGNAL_ARMS
        ]

    p = effects(d_proxy, proxy, a_orders, arm_of)
    t = effects(g, truth, b_orders, lambda a: a)
    return float(sstats.spearmanr(p, t).statistic)


def analyze_cell():
    h, g = analyze.load(NAME), analyze.load(stage_g.NAME)
    if h is None or g is None:
        print("not run" if h is None else "stage G not found")
        return
    c = h["config"]
    S = c["orders"]
    L, T = c["anneals"][1], g["config"]["long_reads"][-1]
    print(
        f"{NAME}: {c['params']:,} params, lr {c['lr']:g}, checkpoint {c['ckpt_step']}, K={c['ckpts']} S={S}, "
        f"{h.get('wall_s', 0) / 60:.0f} min; pi_S "
        + ", ".join(f"{k} {v:.4f}" for k, v in c["pi"].items())
    )
    rep = replication(h, g)
    n_reads = S * c["ckpts"] * (len(c["anneals"]) + len(c["const_reads"]))
    print(
        f"replication of stage G's control (checkpoint losses and {n_reads} readings): largest |difference| "
        f"{rep:.3g}"
    )
    if rep != 0:
        print(
            "REPLICATION FAILED: the control differs from stage G's, so no rule is evaluated."
        )
        return

    designs = [("anneal", x) for x in c["anneals"]] + [
        ("const", x) for x in c["const_reads"]
    ]
    rows = {
        a: [analyze.stats(table(h, "control", *dz), table(h, a, *dz)) for dz in designs]
        for a in ARMS
        if a != "control"
    }
    head = f"{'arm':8} " + " ".join(f"{dz[0]:>6}{dz[1]:>5}" for dz in designs)
    for title, key, fmt in (
        ("mean effect, loss", "effect", "{:+11.5f}"),
        ("seeds saved by pairing", "saved", "{:11.1f}"),
        ("paired runs per arm for 80% power", "n_paired", "{:11.0f}"),
        ("unpaired runs per arm for 80% power", "n_unpaired", "{:11.0f}"),
    ):
        print(f"\n{head}   ({title})")
        for a, st in rows.items():
            print(f"{a:8} " + " ".join(fmt.format(s[key]).rjust(11) for s in st))

    for label, h_reading, g_reading, rules in (
        (f"micro-anneal {L}", ("anneal", L), ("anneal", L), True),
        (f"constant-LR {L}", ("const", L), ("long", L), False),
    ):
        res = ratios(h, g, h_reading, g_reading)
        runs, h1, h2 = verdicts(res)
        print(f"\n##### {label}: re-weighted against swapped (stage G), per arm")
        print(
            f"  {'arm':8} {'effect rw':>10} {'effect sw':>10} {'rw/sw':>6} {'saved rw':>9} {'saved sw':>9} "
            f"{'runs sw/rw':>10}"
        )
        for a, (rw, sw, rr, er) in res.items():
            print(
                f"  {a:8} {rw['effect']:+10.5f} {sw['effect']:+10.5f} {er:6.2f} {rw['saved']:9.1f} "
                f"{sw['saved']:9.1f} {rr:10.1f}"
            )
        tag = (lambda s: f" -> {s}") if rules else (lambda s: f" (reported only: {s})")
        print(
            f"  H1, paired runs needed, swapped / re-weighted, geometric mean: {runs:.1f}{tag(h1)}"
        )
        print(
            f"  H2, re-weighted effect / swapped effect: all six in [0.5, 2]?{tag(h2)}"
        )
        if rules and h2 == "REJECTED":
            print("  H1 is not claimed: H2 is rejected (PLAN_H.md).")

    half = S // 2
    A, B = list(range(half)), list(range(half, S))
    print(
        f"\nSpearman with stage G's truth (long-{T}), proxy and truth on disjoint orders, mean of both splits:"
    )
    for label, d_proxy, arm_of, reading in (
        (f"re-weighted anneal-{L}", h, lambda a: a + "w", ("anneal", L)),
        (f"swapped anneal-{L}", g, lambda a: a, ("anneal", L)),
    ):
        r = (
            spearman_vs_truth(g, d_proxy, arm_of, reading, ("long", T), A, B)
            + spearman_vs_truth(g, d_proxy, arm_of, reading, ("long", T), B, A)
        ) / 2
        print(f"  {label:24} {r:+.2f}")


def smoke():
    import shutil
    import tempfile

    assert study.DEV == "cpu"
    torch.set_num_threads(4)
    study.OUT_DIR = analyze.OUT = tempfile.mkdtemp(prefix="smoke_h_")
    cg = dict(
        stage_g.CFG,
        size="s0",
        ckpt_step=20,
        ckpts=2,
        orders=4,
        anneals=(5, 10),
        long_flat=10,
        long_decay=5,
        long_reads=(5, 10, 15),
    )
    ch = dict(
        CFG,
        size="s0",
        ckpt_step=20,
        ckpts=2,
        orders=4,
        anneals=(5, 10),
        const=10,
        const_reads=(5, 10),
    )
    train, val, sha, spans = stage_g.load_corpus_spans()
    subsets = stage_g.make_subsets(spans, len(train))
    members = make_members(spans, len(train))
    # pi_S agrees with stage G's subset sizes, and the weights average to 1 over the control's sampling
    for name, (mask, share) in members.items():
        assert (
            abs(share - subsets[name].total / (len(train) - study.CTX - 1)) < 1e-4
        ), name
    r = HRunner(ch, train, val, subsets, members)
    assert members["all"][1] == 1.0
    for a in ARMS:
        if a == "control":
            continue
        r.rw = ARMS[a]
        w = torch.cat([r.weights(7, step) for step in range(400)])
        assert abs(float(w.mean()) - 1) < 0.1, (a, float(w.mean()))
        assert a != "null8w" or bool((w == 1).all())
    # a re-weighted arm sees exactly the control's examples; the weights are consumed by the next loss
    r.rw = None
    x0, y0 = r.batch(123, 7)
    r.rw = "enc8"
    x1, y1 = r.batch(123, 7)
    assert torch.equal(x0, x1) and torch.equal(y0, y1) and r._w is not None
    with torch.no_grad():
        r.loss(x1, y1)
    assert r._w is None
    r.rw = None
    # the control reproduces stage G's bit for bit; re-weighted arms differ from it
    stage_g.run(cg)
    run(ch)
    d = json.load(open(os.path.join(study.OUT_DIR, NAME + ".json")))
    gd = json.load(open(os.path.join(study.OUT_DIR, stage_g.NAME + ".json")))
    assert d["done"] and len(d["branches"]) == 2 * 4 * len(ARMS)
    assert replication(d, gd) == 0, "control differs from stage G's"
    for k in range(2):
        for a in ARMS:
            if a not in ("control", "null8w"):
                assert (
                    d["branches"][f"{k}|1|{a}"] != d["branches"][f"{k}|1|control"]
                ), f"{a} equals control"
    analyze_cell()
    shutil.rmtree(study.OUT_DIR)
    print("\nSMOKE OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
        subprocess.run(["git", "add", "-A", "results"], cwd=study.HERE)
        subprocess.run(
            ["git", "commit", "-q", "-m", "Stage H raw results"], cwd=study.HERE
        )
    elif cmd == "analyze":
        analyze_cell()
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit(__doc__)
