"""Mechanics check. No variance or effect is read here; it only asserts that the machinery is right."""

import torch, study as S

train, val, sha = S.load_corpus()
for amp in (True, False):
    c = S.cell(size="s0", amp=amp)
    S.set_kernels(True)
    r = S.Runner(c, train, val)
    ck, _ = r.train(r.new_state(1), 1, 40, 4e-3)
    snap = S.clone(ck)
    _, a = r.train(ck, 7, 30, 4e-3, eval_at=(10, 30))
    _, x = r.train(ck, 7, 30, 4e-3, lr_mult=1.01, eval_at=(10, 30))
    _, w = r.train(ck, 7, 30, 4e-3, wd=0.1, eval_at=(10, 30))
    _, s = r.train(ck, 7, 30, 4e-3, swap=1 / 32, eval_at=(10, 30))
    _, b = r.train(
        ck, 7, 30, 4e-3, eval_at=(10, 30)
    )  # replay after other branches ran in between
    assert a == b, ("replay not exact", a, b)
    assert all(
        torch.equal(ck["model"][k], snap["model"][k]) for k in snap["model"]
    ), "checkpoint mutated"
    assert all(
        torch.equal(p, q)
        for p, q in zip(
            [
                v
                for st in ck["opt"]["state"].values()
                for v in st.values()
                if torch.is_tensor(v)
            ],
            [
                v
                for st in snap["opt"]["state"].values()
                for v in st.values()
                if torch.is_tensor(v)
            ],
        )
    ), "optimizer state mutated"
    assert x != a and w != a and s != a, "an arm equals its control"
    x0, _ = r.batch(7, 5)
    x1, _ = r.batch(7, 5, 1 / 32)
    x8, _ = r.batch(7, 5, 8 / 32)
    x32, _ = r.batch(7, 5, 1.0)
    rows = lambda u: int((u != x0).any(dim=1).sum())
    assert rows(x1) == 1 and rows(x8) == 8 and rows(x32) == 32, (
        rows(x1),
        rows(x8),
        rows(x32),
    )
    assert torch.equal(r.batch(7, 5)[0], x0) and not torch.equal(r.batch(8, 5)[0], x0)
    # a fresh Runner reproduces the same checkpoint from the same seeds
    r2 = S.Runner(c, train, val)
    ck2, _ = r2.train(r2.new_state(1), 1, 40, 4e-3)
    assert all(
        torch.equal(ck["model"][k], ck2["model"][k]) for k in ck["model"]
    ), "checkpoint not reproducible"
    print(
        f"amp={amp}: replay exact, checkpoint untouched and reproducible, arms differ, swaps 1/8/32 rows OK"
    )
# default kernels must at least run
S.set_kernels(False)
r = S.Runner(S.cell(size="s0"), train, val)
r.train(r.new_state(1), 1, 10, 4e-3)
print("default kernels run")
