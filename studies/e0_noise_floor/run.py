"""E0: seed noise floor and branch (CRN) correlation. Pre-registered in docs/TECHNICAL_DECISIONS.md."""

import json, time, sys
import tinygpt as T

LR, BASE_STEPS, HORIZON, EV = 3e-3, 200, 200, 50
out = dict(
    config=dict(
        lr=LR,
        base_steps=BASE_STEPS,
        horizon=HORIZON,
        eval_every=EV,
        ctx=T.CTX,
        batch=T.BATCH,
        model=T.CFG,
        torch=T.torch.__version__,
    )
)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# (a) from-scratch seeds: init seed and data seed both vary
out["scratch"] = []
for i in range(6):
    _, c = T.run(T.new_state(i), 500 + i, BASE_STEPS + HORIZON, LR, eval_every=EV)
    out["scratch"].append(c)
    log("scratch", i, c[-1])

# (b) shared checkpoint, then branches; CRN = same data_seed across multipliers
ckpt, _ = T.run(T.new_state(0), 999, BASE_STEPS, LR, eval_every=EV)
out["branch"] = {}
for ds in range(12):
    for mult in (1.0, 0.5, 2.0):
        _, c = T.run(ckpt, 1000 + ds, HORIZON, LR, lr_mult=mult, eval_every=EV)
        out["branch"][f"{ds}|{mult}"] = c
    log(
        "branch data_seed",
        ds,
        {m: out["branch"][f"{ds}|{m}"][-1][1] for m in (1.0, 0.5, 2.0)},
    )
_, rep = T.run(ckpt, 1000, HORIZON, LR, lr_mult=1.0, eval_every=EV)
out["determinism_repeat_identical"] = rep == out["branch"]["0|1.0"]
json.dump(out, open("results/e0/e0.json", "w"))
log("done")
