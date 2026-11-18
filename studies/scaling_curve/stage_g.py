"""Stage G of the scaling study: paired micro-anneals for data decisions. Cheap, and still right?

Labs test a data source by branching a checkpoint and decaying the learning rate to zero over a short
continuation with the source mixed in (a "micro-anneal"). Stage D found that the learning rate during a branch
sets how long two runs stay paired. This stage measures, for changes to the training data, how much sharing the
data order saves in a micro-anneal against a branch at the training learning rate, and whether the micro-anneal's
verdict agrees with a long continuation's. Plan and decision rules: PLAN_G.md. `study.py` is reused unchanged.

    python stage_g.py run        # the cell (resumable per checkpoint), then commit the results
    python stage_g.py analyze    # the tables and the verdict of each rule
    python stage_g.py smoke      # mechanics check on the CPU with a tiny configuration; reads no real outcome
"""

import glob
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import sysconfig
import time

SMOKE = sys.argv[1:2] == ["smoke"]
if SMOKE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # leave the GPU to the running study

import numpy as np
import torch
from scipy import stats as sstats

import analyze
import study

NAME = "G_s2_anneal"
CFG = dict(
    size="s2",
    lr=2e-3,
    batch=32,
    ckpt_step=1500,
    ckpts=3,
    orders=6,
    anneals=(100, 200, 400),
    long_flat=1200,
    long_decay=400,
    long_reads=(100, 200, 400, 1200, 1600),
)
SUBSET_OFFSET = 700_000  # data-seed offset of the stream that the swapped-in examples are drawn from

# Which training files a subset holds, by path relative to the standard library root.
SUBSETS = {
    "all": lambda p: True,
    "encodings": lambda p: p.startswith("encodings/"),
    "idlelib": lambda p: p.startswith("idlelib/"),
    "toplevel": lambda p: "/" not in p,
    "asyncio": lambda p: p.startswith("asyncio/"),
    "test": lambda p: p.startswith("test/")
    or "/test" in p
    or p.rsplit("/", 1)[-1].startswith("test_"),
}
# arm -> (subset, examples of each batch of 32 drawn from it during the branch). "null8" draws from the whole
# training set, so it has no systematic effect: a measured null.
ARMS = {
    "control": None,
    "null8": ("all", 8),
    "enc8": ("encodings", 8),
    "enc2": ("encodings", 2),
    "idle8": ("idlelib", 8),
    "top8": ("toplevel", 8),
    "async8": ("asyncio", 8),
    "test8": ("test", 8),
}
DATA_ARMS = [a for a in ARMS if a != "control"]
SIGNAL_ARMS = [
    a for a in DATA_ARMS if a != "null8"
]  # null8's true effect is zero by construction


def load_corpus_spans():
    """`study.load_corpus`, plus each file's byte range. Asserts the bytes are the ones the study trains on."""
    root = sysconfig.get_paths()["stdlib"]
    buf, spans = bytearray(), []
    for f in sorted(glob.glob(root + "/**/*.py", recursive=True)):
        if "site-packages" in f:
            continue
        try:
            with open(f, "rb") as fh:
                b = fh.read()
        except OSError:
            continue
        spans.append(
            (os.path.relpath(f, root).replace("\\", "/"), len(buf), len(buf) + len(b))
        )
        buf += b
    train, val, sha = study.load_corpus()
    assert (
        hashlib.sha256(bytes(buf)).hexdigest() == sha
    ), "corpus differs from study.load_corpus"
    return train, val, sha, spans


class Subset:
    """Uniform start positions over the training bytes of some files. A window may run past the end of its file."""

    def __init__(self, ranges, limit):
        assert ranges, "empty subset"
        self.starts = torch.tensor([s for s, _ in ranges])
        self.cum = torch.cumsum(torch.tensor([e - s for s, e in ranges]), 0)
        self.total, self.limit = int(self.cum[-1]), limit

    def sample(self, g, m):
        u = torch.randint(0, self.total, (m,), generator=g)
        i = torch.searchsorted(self.cum, u, right=True)
        before = torch.where(i > 0, self.cum[(i - 1).clamp(min=0)], torch.zeros_like(u))
        return (self.starts[i] + u - before).clamp(max=self.limit)


def make_subsets(spans, n_train):
    limit = n_train - study.CTX - 2
    out = {}
    for name, pred in SUBSETS.items():
        ranges = [(s, min(e, n_train)) for p, s, e in spans if s < n_train and pred(p)]
        out[name] = Subset(ranges, limit)
    return out


class GRunner(study.Runner):
    """`study.Runner` with batches whose first examples come from a subset, and a learning-rate schedule."""

    def __init__(self, c, train, val, subsets):
        super().__init__(study.cell(size=c["size"], batch=c["batch"]), train, val)
        self.subsets, self.arm = subsets, None

    def batch(self, data_seed, step, swap=0.0):
        idx = self._indices(data_seed, step)
        if self.arm is not None:
            subset, m = self.arm
            g = torch.Generator().manual_seed(
                (data_seed + SUBSET_OFFSET) * 1_000_003 + step
            )
            idx[:m] = self.subsets[subset].sample(g, m)
        return self._cut(self.train_data, idx)

    def branch(self, state, data_seed, steps, lr_at, eval_at):
        """Continue `state` with learning rate `lr_at(k)` at branch step k. `state` is left untouched."""
        m = self.model
        m.load_state_dict(state["model"])
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=0.0, weight_decay=0.0)
        opt.load_state_dict(study.clone(state["opt"]))
        readings = {}
        for k in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(k)
                group["weight_decay"] = 0.0
            x, y = self.batch(data_seed, state["step"] + k)
            with self.autocast():
                loss = self.loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), study.CLIP)
            opt.step()
            if (k + 1) in eval_at:
                readings[k + 1] = self.evaluate()
        return readings


def anneal_lr(base, length):
    return lambda k: base * (1 - k / length)


def long_lr(base, flat, decay):
    return lambda k: base if k < flat else base * (1 - (k - flat) / decay)


def run(c=CFG):
    path = os.path.join(study.OUT_DIR, NAME + ".json")
    os.makedirs(study.OUT_DIR, exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    if out.get("done"):
        print(NAME, "already done", flush=True)
        return
    study.set_kernels(True)
    train, val, sha, spans = load_corpus_spans()
    subsets = make_subsets(spans, len(train))
    r = GRunner(c, train, val, subsets)
    out.setdefault(
        "config",
        {
            **c,
            "params": r.params,
            "device": study.DEV,
            "torch": torch.__version__,
            "corpus_sha256": sha,
            "arms": {a: v for a, v in ARMS.items()},
            "subset_bytes": {k: v.total for k, v in subsets.items()},
        },
    )
    out.setdefault("branches", {})
    t0 = time.time()
    flat, decay = c["long_flat"], c["long_decay"]
    for k in range(c["ckpts"]):
        if all(
            f"{k}|{s}|{a}" in out["branches"] for s in range(c["orders"]) for a in ARMS
        ):
            continue
        r.arm = None
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
            for a, arm in list(ARMS.items()) + ([("twin", None)] if s == 0 else []):
                r.arm = arm
                anneal = {
                    str(L): r.branch(ck, seed, L, anneal_lr(c["lr"], L), (L,))[L]
                    for L in c["anneals"]
                }
                lng = r.branch(
                    ck,
                    seed,
                    flat + decay,
                    long_lr(c["lr"], flat, decay),
                    c["long_reads"],
                )
                out["branches"][f"{k}|{s}|{a}"] = dict(
                    anneal=anneal, long={str(h): v for h, v in lng.items()}
                )
        r.arm = None
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


def table(d, arm, design, reading, orders=None):
    """(K, S) losses of an arm for one design ("anneal" or "long") and reading (anneal length or step)."""
    c = d["config"]
    ss = range(c["orders"]) if orders is None else orders
    return np.array(
        [
            [d["branches"][f"{k}|{s}|{arm}"][design][str(reading)] for s in ss]
            for k in range(c["ckpts"])
        ],
        dtype=float,
    )


def geomean(xs):
    return math.exp(float(np.mean(np.log(xs))))


def three_way(r, at_least=4.0, same=2.0):
    return (
        "SUPPORTED"
        if r >= at_least
        else "REJECTED" if 1 / same <= r <= same else "INCONCLUSIVE"
    )


def spearman_split(d, proxy, truth, a_orders, b_orders):
    """Spearman correlation over the data arms between the proxy's mean effect on orders `a_orders` and the
    truth's mean effect on the disjoint orders `b_orders`."""

    def effects(design_reading, orders):
        design, reading = design_reading
        ctrl = table(d, "control", design, reading, orders)
        return [
            float((table(d, a, design, reading, orders) - ctrl).mean())
            for a in DATA_ARMS
        ]

    p, t = effects(proxy, a_orders), effects(truth, b_orders)
    return float(sstats.spearmanr(p, t).statistic), p, t


def single_decisions(d, proxy, truth):
    """Wrong-sign rates of one paired and one unpaired proxy comparison (2 runs each), against the truth's sign
    on orders the comparison did not use. Only (arm, comparison) cases whose truth is established count, and the
    null arm is left out: its true effect is zero, so any direction established for it is itself an error.
    """
    c = d["config"]
    K, S = c["ckpts"], c["orders"]
    td = {a: table(d, a, *truth) - table(d, "control", *truth) for a in SIGNAL_ARMS}
    px = {a: table(d, a, *proxy) for a in ARMS}
    out = {}
    for a in SIGNAL_ARMS:
        paired, unpaired = [], []
        for excl in itertools.chain(
            ((s,) for s in range(S)), itertools.permutations(range(S), 2)
        ):
            rest = td[a][:, [s for s in range(S) if s not in excl]].ravel()
            mean, se = rest.mean(), rest.std(ddof=1) / math.sqrt(rest.size)
            if abs(mean) <= 2 * se:
                continue
            sign = math.copysign(1.0, mean)
            for k in range(K):
                if len(excl) == 1:
                    diff = px[a][k, excl[0]] - px["control"][k, excl[0]]
                    paired.append(
                        float(np.sign(diff) == -sign) + 0.5 * float(diff == 0)
                    )
                else:
                    diff = px[a][k, excl[0]] - px["control"][k, excl[1]]
                    unpaired.append(
                        float(np.sign(diff) == -sign) + 0.5 * float(diff == 0)
                    )
        out[a] = (
            np.mean(paired) if paired else None,
            np.mean(unpaired) if unpaired else None,
        )
    return out


def analyze_cell():
    d = analyze.load(NAME)
    if d is None:
        print("not run")
        return
    c = d["config"]
    K, S = c["ckpts"], c["orders"]
    print(
        f"{NAME}: {c['params']:,} params, lr {c['lr']:g}, checkpoint {c['ckpt_step']}, K={K} S={S}, "
        f"{d.get('wall_s', 0) / 60:.0f} min; checkpoint loss "
        + ", ".join(f"{v:.3f}" for v in d["ckpt_loss"].values())
    )
    twins = [
        max(
            abs(x - y)
            for x, y in zip(
                d["branches"][f"{k}|0|twin"]["long"].values(),
                d["branches"][f"{k}|0|control"]["long"].values(),
            )
        )
        for k in range(K)
    ]
    print(
        f"identical-twin max |diff| {max(twins):.3g}; subset bytes {c['subset_bytes']}"
    )
    designs = [("anneal", x) for x in c["anneals"]] + [
        ("long", h) for h in c["long_reads"]
    ]
    L, T = (
        c["anneals"][1],
        c["long_reads"][-1],
    )  # the main micro-anneal length (200) and the truth's reading (1600)
    print(
        f"\n{'arm':8} "
        + " ".join(f"{dz[0][:6]}{dz[1]:>5}" for dz in designs)
        + "   (mean effect, loss)"
    )
    rows = {}
    for a in DATA_ARMS:
        st = [
            analyze.stats(table(d, "control", *dz), table(d, a, *dz)) for dz in designs
        ]
        rows[a] = dict(zip(designs, st))
        print(f"{a:8} " + " ".join(f"{s['effect']:+11.5f}" for s in st))
    print(
        f"\n{'arm':8} "
        + " ".join(f"{dz[0][:6]}{dz[1]:>5}" for dz in designs)
        + "   (seeds saved by pairing)"
    )
    for a in DATA_ARMS:
        print(f"{a:8} " + " ".join(f"{rows[a][dz]['saved']:11.1f}" for dz in designs))
    print(
        f"control SD across orders: "
        + " ".join(
            f"{dz[0][:6]}{dz[1]}: {math.sqrt(table(d, 'control', *dz).var(axis=1, ddof=1).mean()):.4f}"
            for dz in designs
        )
    )

    print(
        f"\n##### G1: a {L}-step micro-anneal makes the paired comparison at least 4x cheaper than a {L}-step "
        "branch at the training learning rate"
    )
    ga = geomean([rows[a][("anneal", L)]["saved"] for a in DATA_ARMS])
    gc = geomean([rows[a][("long", L)]["saved"] for a in DATA_ARMS])
    print(
        f"  geometric mean seeds saved over {len(DATA_ARMS)} data arms: anneal-{L} {ga:.1f}, constant-LR {L} "
        f"{gc:.1f}, ratio {ga / gc:.2f} -> {three_way(ga / gc)}"
    )

    print(
        "\n##### G2: the micro-anneal's ranking of the data changes agrees with the long continuation's"
    )
    half = S // 2
    A, B = list(range(half)), list(range(half, S))
    proxies = [(f"anneal-{L}", ("anneal", L)), (f"constant-LR {L}", ("long", L))]
    proxies += [(f"anneal-{x}", ("anneal", x)) for x in c["anneals"] if x != L]
    for label, proxy in proxies:
        r1, p1, t1 = spearman_split(d, proxy, ("long", T), A, B)
        r2, p2, t2 = spearman_split(d, proxy, ("long", T), B, A)
        verdict = ""
        if proxy == ("anneal", L):
            m = (r1 + r2) / 2
            verdict = f" -> {'SUPPORTED' if m >= 0.75 else 'REJECTED' if m <= 0.4 else 'INCONCLUSIVE'}"
        print(
            f"  {label:16} Spearman (proxy orders {A[0]}-{A[-1]}, truth {B[0]}-{B[-1]}) {r1:+.2f}; swapped {r2:+.2f}; "
            f"mean {(r1 + r2) / 2:+.2f}{verdict}"
        )
    full_truth = {a: rows[a][("long", T)] for a in DATA_ARMS}
    print(
        f"  truth (long-{T}, all pairs): "
        + ", ".join(
            f"{a} {s['effect']:+.5f}{'*' if s['resolved'] else ''}"
            for a, s in full_truth.items()
        )
        + "  (* established)"
    )

    print(
        "\n##### G3: one paired micro-anneal (2 runs) gets the direction right at least twice as often as one "
        "unpaired pair of runs, and is wrong at most 10% of the time"
    )
    for label, proxy in (
        (f"anneal-{L}", ("anneal", L)),
        (f"constant-LR {L}", ("long", L)),
    ):
        res = single_decisions(d, proxy, ("long", T))
        have = {a: v for a, v in res.items() if v[0] is not None and v[1] is not None}
        line = ", ".join(
            f"{a} {100 * p:.0f}%/{100 * u:.0f}%" for a, (p, u) in have.items()
        )
        if not have:
            print(f"  {label:16} no arm has an established truth")
            continue
        p, u = float(np.mean([v[0] for v in have.values()])), float(
            np.mean([v[1] for v in have.values()])
        )
        verdict = ""
        if proxy == ("anneal", L):
            if len(have) < 3 or u < 0.04:
                verdict = " -> UNINFORMATIVE"
            else:
                verdict = f" -> {'PASS' if p <= 0.10 and p <= u / 2 else 'FAIL'}"
        print(
            f"  {label:16} wrong direction, paired {100 * p:.1f}% vs unpaired {100 * u:.1f}% over {len(have)} arms "
            f"with an established truth{verdict}\n      per arm (paired/unpaired): {line}"
        )


def smoke():
    import shutil
    import tempfile

    assert study.DEV == "cpu"
    torch.set_num_threads(4)
    study.OUT_DIR = analyze.OUT = tempfile.mkdtemp(prefix="smoke_g_")
    c = dict(
        CFG,
        size="s0",
        ckpt_step=20,
        ckpts=2,
        orders=4,
        anneals=(5, 10),
        long_flat=10,
        long_decay=5,
        long_reads=(5, 10, 15),
    )
    # schedules
    f = anneal_lr(1.0, 10)
    assert f(0) == 1.0 and abs(f(9) - 0.1) < 1e-12
    g = long_lr(1.0, 10, 5)
    assert g(9) == 1.0 and g(10) == 1.0 and abs(g(14) - 0.2) < 1e-12
    # subsets draw only from their own files (start positions), and the sampler is deterministic
    train, val, sha, spans = load_corpus_spans()
    subsets = make_subsets(spans, len(train))
    for name, pred in SUBSETS.items():
        pos = subsets[name].sample(torch.Generator().manual_seed(1), 500)
        assert torch.equal(
            pos, subsets[name].sample(torch.Generator().manual_seed(1), 500)
        )
        ok = [
            any(s <= int(p) < e and pred(q) for q, s, e in spans)
            or int(p) == subsets[name].limit
            for p in pos
        ]
        assert all(ok), name
    run(c)
    d = json.load(open(os.path.join(study.OUT_DIR, NAME + ".json")))
    b = d["branches"]
    assert d["done"] and len(b) == 2 * (4 * len(ARMS) + 1)
    for k in range(2):
        assert b[f"{k}|0|twin"] == b[f"{k}|0|control"], "identical twin differs"
        for a in DATA_ARMS:
            assert b[f"{k}|1|{a}"] != b[f"{k}|1|control"], f"{a} equals control"
    # pairing: every arm shares the control's examples outside the swapped ones
    r = GRunner(c, train, val, subsets)
    r.arm = None
    x0, _ = r.batch(123, 7)
    r.arm = ARMS["enc2"]
    x1, _ = r.batch(123, 7)
    assert torch.equal(x0[2:], x1[2:]) and not torch.equal(x0[:2], x1[:2])
    analyze_cell()
    shutil.rmtree(study.OUT_DIR)
    print("\nSMOKE OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
        subprocess.run(["git", "add", "-A", "results"], cwd=study.HERE)
        subprocess.run(
            ["git", "commit", "-q", "-m", "Stage G raw results"], cwd=study.HERE
        )
    elif cmd == "analyze":
        analyze_cell()
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit(__doc__)
