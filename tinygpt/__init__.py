"""Tiny char-level GPT for E0/E1. CPU, deterministic, data order = f(data_seed, step) so branches share batches (CRN)."""

import copy, glob, os, sys, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
DEV = os.environ.get("REGORA_DEV", "cpu")  # cpu | mps | cuda
CTX, BATCH = 128, 32
CFG = dict(n_layer=4, n_embd=128, n_head=4)


def load_corpus(max_bytes=3_000_000):
    root = os.path.dirname(os.__file__)
    buf = bytearray()
    for f in sorted(glob.glob(root + "/**/*.py", recursive=True)):
        try:
            buf += open(f, "rb").read()
        except OSError:
            pass
        if len(buf) >= max_bytes:
            break
    d = torch.tensor(list(bytes(buf[:max_bytes])), dtype=torch.long)
    n = int(len(d) * 0.9)
    return d[:n], d[n:]


TRAIN, VAL = load_corpus()
_vg = torch.Generator().manual_seed(12345)
VAL_IDX = torch.randint(
    0, len(VAL) - CTX - 1, (20 * BATCH,), generator=_vg
)  # fixed eval set


class Block(nn.Module):
    def __init__(s, e, h):
        super().__init__()
        s.l1 = nn.LayerNorm(e)
        s.l2 = nn.LayerNorm(e)
        s.at = nn.MultiheadAttention(e, h, batch_first=True)
        s.ff = nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(), nn.Linear(4 * e, e))

    def forward(s, x, m):
        y = s.l1(x)
        x = x + s.at(y, y, y, attn_mask=m, need_weights=False)[0]
        return x + s.ff(s.l2(x))


class GPT(nn.Module):
    def __init__(s, n_layer, n_embd, n_head):
        super().__init__()
        s.te = nn.Embedding(256, n_embd)
        s.pe = nn.Embedding(CTX, n_embd)
        s.bs = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        s.ln = nn.LayerNorm(n_embd)
        s.hd = nn.Linear(n_embd, 256, bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))

    def forward(s, x):
        h = s.te(x) + s.pe(torch.arange(x.shape[1], device=x.device))
        for b in s.bs:
            h = b(h, s.mask)
        return s.hd(s.ln(h))


def batch(data, idx):
    ix = idx[:, None] + torch.arange(CTX + 1)[None]
    t = data[ix].to(DEV)
    return t[:, :-1], t[:, 1:]


def to_cpu(o):
    if torch.is_tensor(o):
        return o.detach().cpu().clone()
    if isinstance(o, dict):
        return {k: to_cpu(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_cpu(v) for v in o]
    return o


def train_batch(data_seed, step):
    g = torch.Generator().manual_seed(data_seed * 1_000_003 + step)
    return batch(TRAIN, torch.randint(0, len(TRAIN) - CTX - 1, (BATCH,), generator=g))


@torch.no_grad()
def val_loss(m):
    m.eval()
    tot = 0.0
    for i in range(0, len(VAL_IDX), BATCH):
        x, y = batch(VAL, VAL_IDX[i : i + BATCH])
        tot += F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)).item()
    m.train()
    return tot / (len(VAL_IDX) // BATCH)


def lr_at(step, base, warm=50, total=None):
    w = min(1.0, (step + 1) / warm)
    if total is None:
        return base * w
    import math

    return (
        base * w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
    )  # cosine to 10%


def new_state(init_seed):
    torch.manual_seed(init_seed)
    m = GPT(**CFG)
    return dict(model=m.state_dict(), opt=None, step=0)


def run(
    state,
    data_seed,
    steps,
    base_lr,
    lr_mult=1.0,
    eval_every=25,
    on_step=None,
    total=None,
    wd=0.0,
    clip=1.0,
    warm=50,
):
    """Continue `state` for `steps` steps. Returns (new_state, [(step, val_loss)])."""
    m = GPT(**CFG).to(DEV)
    m.load_state_dict(state["model"])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=base_lr, weight_decay=0.0)
    if state["opt"] is not None:
        opt.load_state_dict(
            copy.deepcopy(state["opt"])
        )  # deepcopy: load_state_dict aliases tensors, which would mutate the shared checkpoint
    for g in opt.param_groups:
        g["weight_decay"] = wd  # after load: saved groups would otherwise override
    curve = []
    s0 = state["step"]
    for k in range(steps):
        st = s0 + k
        for g in opt.param_groups:
            g["lr"] = lr_at(st, base_lr, warm=warm, total=total) * lr_mult
        x, y = train_batch(data_seed, st)
        loss = F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        if on_step:
            on_step(st, loss.item())
        if (k + 1) % eval_every == 0:
            curve.append((st + 1, val_loss(m)))
    return (
        dict(
            model=to_cpu(m.state_dict()), opt=to_cpu(opt.state_dict()), step=s0 + steps
        ),
        curve,
    )
