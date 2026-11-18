"""E0b: replication of E0 with a standard schedule and several checkpoints.

PRE-REGISTERED before running (committed first):
- Config: base LR 1e-3, warmup 50, cosine to 10% over total=600 steps. 3 checkpoints (init seeds 0,1,2) at step 300.
  From each: 8 data seeds x LR multiplier {1, 0.5, 2}, 200 steps (to step 500), eval every 50.
- Primary endpoint: mean of val loss at h=100,150,200 (decided after seeing E0, so it is a hypothesis here, not a fresh one).
- Per checkpoint and pooled (paired differences, SD of paired diff within checkpoint, effects averaged across checkpoints):
  effect/SD and n80_paired = 7.85*(SD/effect)^2.
- Decision rule: for a multiplier, "power-limited" if pooled effect/SD < 1 (n80_paired > ~8). Report both multipliers; do not average them.
- Also report rank agreement of {0.5,1,2} at h=50,100,150 vs h=200 and the determinism repeat.
"""

import json, time
import tinygpt as T

LR, CK, HORIZON, EV, TOTAL = 1e-3, 300, 200, 50, 600
out = dict(
    config=dict(
        lr=LR,
        ckpt_step=CK,
        horizon=HORIZON,
        eval_every=EV,
        total=TOTAL,
        model=T.CFG,
        torch=T.torch.__version__,
    ),
    branch={},
    det={},
)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
for ck in range(3):
    state, _ = T.run(T.new_state(ck), 900 + ck, CK, LR, eval_every=EV, total=TOTAL)
    for ds in range(8):
        for mult in (1.0, 0.5, 2.0):
            _, c = T.run(
                state, 1000 + ds, HORIZON, LR, lr_mult=mult, eval_every=EV, total=TOTAL
            )
            out["branch"][f"{ck}|{ds}|{mult}"] = c
    _, rep = T.run(state, 1000, HORIZON, LR, lr_mult=1.0, eval_every=EV, total=TOTAL)
    out["det"][ck] = rep == out["branch"][f"{ck}|0|1.0"]
    log("checkpoint", ck, "done; deterministic:", out["det"][ck])
json.dump(out, open("results/e0b/e0b.json", "w"))
log("done")
