"""Scaling study: how the value of a paired comparison changes with model size, change size, horizon and friends.

Self-contained (does not import the repo's packages). Follows the conventions of `experiments/scale` and
`experiments/tinygpt`: byte-level GPT on Python source, AdamW, 100 steps of warmup then constant LR, bfloat16
autocast on CUDA, the batch at a step is a pure function of (data_seed, step), a state is a dict that can be
branched any number of times.

Usage:
    python study.py bench                 # speed only, no outcome data
    python study.py run <cell> [<cell>..] # run cells (resumable), results in results/<cell>.json
    python study.py list
The plan and decision rules are in PLAN.md and were written before any outcome data existed.
"""

import glob
import hashlib
import json
import math
import os
import sys
import sysconfig
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX = 128
WARMUP = 100
CLIP = 1.0
EVAL_SEED = 12345
EVAL_BATCHES, EVAL_BATCH = 20, 64
TUNE_SEEDS = (3, 4)
SWAP_OFFSET = (
    500_000  # data-seed offset of the stream that swapped-in examples are drawn from
)

# name -> (layers, width)
SIZES = {"s0": (2, 64), "s1": (4, 128), "s2": (6, 256), "s3": (8, 384), "s4": (8, 512)}
# The grid each size's healthy LR is tuned over (a factor of 2 apart).
LR_GRIDS = {
    "s0": (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
    "s1": (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
    "s2": (5e-4, 1e-3, 2e-3, 4e-3, 8e-3),
    "s3": (2.5e-4, 5e-4, 1e-3, 2e-3),
    "s4": (2.5e-4, 5e-4, 1e-3, 2e-3),
}

# arm -> (lr multiplier, weight decay, fraction of each batch swapped for other examples). Control: (1, 0, 0).
FULL_ARMS = {
    "control": (1.0, 0.0, 0.0),
    "lr1.01": (1.01, 0.0, 0.0),
    "lr1.03": (1.03, 0.0, 0.0),
    "lr1.1": (1.1, 0.0, 0.0),
    "lr1.3": (1.3, 0.0, 0.0),
    "lr2.0": (2.0, 0.0, 0.0),
    "lr0.5": (0.5, 0.0, 0.0),
    "wd0.01": (1.0, 0.01, 0.0),
    "wd0.1": (1.0, 0.1, 0.0),
    "wd1.0": (1.0, 1.0, 0.0),
    "swap1": (1.0, 0.0, 1 / 32),
    "swap8": (1.0, 0.0, 8 / 32),
    "swap32": (1.0, 0.0, 1.0),
}
LEAN = (
    "control",
    "lr1.01",
    "lr1.1",
    "lr2.0",
    "lr0.5",
    "wd0.01",
    "wd0.1",
    "wd1.0",
    "swap1",
    "swap8",
)
LONG = ("control", "lr1.01", "lr1.1", "lr2.0", "wd0.1", "swap1")
ARM_SETS = {"full": tuple(FULL_ARMS), "lean": LEAN, "long": LONG}

BASE = dict(
    size="s1",
    batch=32,
    train_bytes=0,  # 0 = all of it
    ckpt_step=600,
    horizons=(25, 50, 100, 200, 400),
    ckpts=4,
    orders=5,
    arms="lean",
    exact=True,
    amp=True,
    twin="first",  # "first": an identical second control on order 0 of each checkpoint; "all": on every order
)


def cell(**kw):
    return {**BASE, **kw}


CELLS = {
    # Stage A: the sizes the law is fitted on
    "A_s0": cell(size="s0", arms="full", orders=6),
    "A_s1": cell(size="s1", arms="full", orders=6),
    "A_s2": cell(size="s2", arms="full", orders=6),
    # Stage B: held-out sizes, predicted from stage A before they run
    "B_s3": cell(size="s3"),
    "B_s4": cell(size="s4", ckpts=3),
    # Stage C: one factor at a time around the 4x128 model
    "C_batch8": cell(batch=8),
    "C_batch128": cell(batch=128, ckpts=3),
    "C_data100k": cell(train_bytes=100_000),
    "C_data1m": cell(train_bytes=1_000_000),
    "C_age150": cell(ckpt_step=150),
    "C_age2400": cell(ckpt_step=2400),
    "C_default_kernels": cell(exact=False, twin="all"),
    "C_fp32": cell(amp=False),
    "C_long": cell(horizons=(100, 200, 400, 800, 1600), arms="long", ckpts=3),
}


def load_corpus():
    """Python standard-library source of the interpreter running this (as `experiments/tinygpt` does), in sorted
    file order, site-packages excluded. First 90% trains, the rest is validation. Returns (train, val, sha256).
    """
    root = sysconfig.get_paths()["stdlib"]
    buf = bytearray()
    for f in sorted(glob.glob(root + "/**/*.py", recursive=True)):
        if "site-packages" in f:
            continue
        try:
            with open(f, "rb") as fh:
                buf += fh.read()
        except OSError:
            pass
    d = torch.frombuffer(buf, dtype=torch.uint8)
    n = int(len(d) * 0.9)
    return d[:n], d[n:], hashlib.sha256(bytes(buf)).hexdigest()


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = (
            self.qkv(self.ln1(x))
            .view(b, t, 3, self.heads, w // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, w))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, layers, width):
        super().__init__()
        self.te, self.pe = nn.Embedding(256, width), nn.Embedding(CTX, width)
        self.blocks = nn.ModuleList(
            Block(width, max(2, width // 64)) for _ in range(layers)
        )
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)

    def forward(self, x):
        h = self.te(x) + self.pe(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


def clone(o):
    """Deep copy, so a checkpoint is never aliased by a run that continues it (the bug in E0's first run)."""
    if torch.is_tensor(o):
        return o.detach().clone()
    if isinstance(o, dict):
        return {k: clone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clone(v) for v in o]
    return o


class Runner:
    """One model shape with its data. Holds a single model instance that states are loaded into."""

    def __init__(self, c, train, val):
        self.c = c
        layers, width = SIZES[c["size"]]
        self.model = GPT(layers, width).to(DEV)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.train_data = (train[: c["train_bytes"]] if c["train_bytes"] else train).to(
            DEV
        )
        self.span = torch.arange(CTX + 1, device=DEV)[None]
        g = torch.Generator().manual_seed(EVAL_SEED)
        idx = torch.randint(
            0, len(val) - CTX - 1, (EVAL_BATCHES, EVAL_BATCH), generator=g
        )
        val = val.to(DEV)
        self.eval_set = [self._cut(val, i) for i in idx]

    def _cut(self, data, idx):
        t = data[idx.to(DEV)[:, None] + self.span].long()
        return t[:, :-1], t[:, 1:]

    def _indices(self, data_seed, step):
        g = torch.Generator().manual_seed(data_seed * 1_000_003 + step)
        return torch.randint(
            0, len(self.train_data) - CTX - 1, (self.c["batch"],), generator=g
        )

    def swapped(self, frac):
        """How many examples of a batch a swap arm replaces."""
        return 0 if frac == 0 else max(1, round(frac * self.c["batch"]))

    def batch(self, data_seed, step, swap=0.0):
        """The batch at a step. With `swap`, the first examples are replaced by ones from another stream, which
        models a change to the training data: the rest of the batch is still shared with the control.
        """
        idx = self._indices(data_seed, step)
        m = self.swapped(swap)
        if m:
            idx[:m] = self._indices(data_seed + SWAP_OFFSET, step)[:m]
        return self._cut(self.train_data, idx)

    def autocast(self):
        return torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda" and self.c["amp"])
        )

    def loss(self, x, y):
        return F.cross_entropy(self.model(x).flatten(0, -2).float(), y.flatten())

    def new_state(self, seed):
        torch.manual_seed(seed)
        fresh = GPT(*SIZES[self.c["size"]]).to(DEV)
        return dict(model=clone(fresh.state_dict()), opt=None, step=0)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        with self.autocast():
            total = sum(self.loss(x, y).item() for x, y in self.eval_set)
        self.model.train()
        mean = total / len(self.eval_set)
        return mean if math.isfinite(mean) else math.inf

    def train(
        self, state, data_seed, steps, lr, lr_mult=1.0, wd=0.0, swap=0.0, eval_at=()
    ):
        """Continue `state` for `steps`. Returns (new_state, {steps_done: val_loss}). `state` is left untouched."""
        m = self.model
        m.load_state_dict(state["model"])
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
        if state["opt"] is not None:
            opt.load_state_dict(clone(state["opt"]))
        readings = {}
        for k in range(steps):
            step = state["step"] + k
            for g in opt.param_groups:
                g["lr"] = lr * lr_mult * min(1.0, (step + 1) / WARMUP)
                g["weight_decay"] = wd  # after load, which restores the saved value
            x, y = self.batch(data_seed, step, swap)
            with self.autocast():
                loss = self.loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            if (k + 1) in eval_at:
                readings[k + 1] = self.evaluate()
        new = dict(
            model=clone(m.state_dict()),
            opt=clone(opt.state_dict()),
            step=state["step"] + steps,
        )
        return new, readings


def set_kernels(exact):
    torch.use_deterministic_algorithms(exact)
    torch.backends.cudnn.benchmark = not exact


def run_cell(name, train, val, sha):
    c = CELLS[name]
    path = os.path.join(OUT_DIR, name + ".json")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    if out.get("done"):
        print(name, "already done", flush=True)
        return
    set_kernels(c["exact"])
    r = Runner(c, train, val)
    arms = {a: FULL_ARMS[a] for a in ARM_SETS[c["arms"]]}
    horizons = tuple(c["horizons"])
    H = max(horizons)
    grid = LR_GRIDS[c["size"]]
    t0 = time.time()
    out.setdefault(
        "config",
        {
            **c,
            "params": r.params,
            "device": DEV,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
            "corpus_sha256": sha,
            "train_len": len(r.train_data),
            "lr_grid": grid,
            "swapped": {a: r.swapped(v[2]) for a, v in arms.items() if v[2]},
        },
    )
    out.setdefault("branches", {})

    def save():
        nonlocal t0
        out["wall_s"] = out.get("wall_s", 0) + time.time() - t0
        t0 = time.time()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)

    # 1. healthy LR: lowest mean val loss at ckpt_step + H over the tuning seeds
    if "lr" not in out:
        scores = {}
        for lr in grid:
            vals = []
            for s in TUNE_SEEDS:
                _, rd = r.train(
                    r.new_state(s),
                    s,
                    c["ckpt_step"] + H,
                    lr,
                    eval_at=(c["ckpt_step"] + H,),
                )
                vals.append(rd[c["ckpt_step"] + H])
            scores[str(lr)] = sum(vals) / len(vals)
            print(f"{name} tune lr {lr:g}: {scores[str(lr)]:.4f}", flush=True)
        out["tune"] = scores
        out["lr"] = float(min(scores, key=scores.get))
        save()
    lr = out["lr"]
    print(f"{name} params {r.params:,} lr {lr:g}", flush=True)

    # 2. checkpoints, then branches
    for k in range(c["ckpts"]):
        need = [
            (s, a)
            for s in range(c["orders"])
            for a in list(arms) + ["twin"]
            if f"{k}|{s}|{a}" not in out["branches"]
            and (a != "twin" or c["twin"] == "all" or s == 0)
        ]
        if not need:
            continue
        ck, rd = r.train(
            r.new_state(1000 + k),
            2000 + k,
            c["ckpt_step"],
            lr,
            eval_at=(c["ckpt_step"],),
        )
        out.setdefault("ckpt_loss", {})[str(k)] = rd[c["ckpt_step"]]
        for s, a in need:
            mult, wd, swap = arms["control"] if a == "twin" else arms[a]
            _, rd = r.train(
                ck,
                100 * (k + 1) + s,
                H,
                lr,
                lr_mult=mult,
                wd=wd,
                swap=swap,
                eval_at=horizons,
            )
            out["branches"][f"{k}|{s}|{a}"] = [rd[h] for h in horizons]
        save()
        print(f"{name} checkpoint {k} done ({out['wall_s']:.0f}s so far)", flush=True)
    out["done"] = True
    save()


def bench(train, val):
    for amp in (True, False):
        for size in SIZES:
            c = cell(size=size, amp=amp)
            set_kernels(True)
            r = Runner(c, train, val)
            st = r.new_state(0)
            r.train(st, 0, 20, 1e-3)
            torch.cuda.synchronize() if DEV == "cuda" else None
            t = time.time()
            r.train(st, 0, 100, 1e-3)
            torch.cuda.synchronize() if DEV == "cuda" else None
            dt = time.time() - t
            t = time.time()
            r.evaluate()
            ev = time.time() - t
            print(
                f"{size} {SIZES[size]} params {r.params:,} amp={amp}: {100 / dt:.1f} steps/s, eval {ev:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for n, c in CELLS.items():
            print(n, c)
        sys.exit()
    train, val, sha = load_corpus()
    print(
        f"corpus: {len(train):,} train bytes, {len(val):,} val bytes, sha256 {sha[:16]}",
        flush=True,
    )
    if cmd == "bench":
        bench(train, val)
    elif cmd == "run":
        for n in sys.argv[2:]:
            run_cell(n, train, val, sha)
