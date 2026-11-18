"""E0c: detection limit for small-effect interventions (weight decay, grad clip).

PRE-REGISTERED before running (committed first):
- Same setup as E0b: LR 1e-3, warmup 50, cosine to 10% over 600, 3 checkpoints at step 300, 8 data seeds, 200-step branches, eval every 50.
- Arms (vs control wd=0, clip=1.0): wd=0.1; wd=1.0 (extreme, positive control for 'some effect'); clip=0.25.
- Control curves are reused from E0b (same deterministic computation); one control per checkpoint is recomputed and must match bitwise, else abort.
- Endpoint: mean val loss at h=100,150,200 (as E0b).
- Per arm: pooled effect, pooled paired SD, effect/SD, per-checkpoint effects, and MDE(n=8) = 2.8*sd_pair/sqrt(8) (80% power, alpha 0.05).
- 'Detectable' = pooled |effect/SD| >= 1 AND same sign in >= 2 of 3 checkpoints. Report the rest as 'not detectable at this n'.
"""

import json, time
import tinygpt as T

LR, CK, HORIZON, EV, TOTAL = 1e-3, 300, 200, 50, 600
b = json.load(open("results/e0b/e0b.json"))["branch"]
ARMS = {"wd0.1": dict(wd=0.1), "wd1.0": dict(wd=1.0), "clip0.25": dict(clip=0.25)}
out = dict(
    config=dict(
        lr=LR,
        ckpt_step=CK,
        horizon=HORIZON,
        total=TOTAL,
        arms=ARMS,
        torch=T.torch.__version__,
    ),
    branch={},
)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
for ck in range(3):
    state, _ = T.run(T.new_state(ck), 900 + ck, CK, LR, eval_every=EV, total=TOTAL)
    _, rep = T.run(state, 1000, HORIZON, LR, eval_every=EV, total=TOTAL)
    assert [list(x) for x in rep] == [
        list(x) for x in b[f"{ck}|0|1.0"]
    ], "control does not reproduce E0b"
    for ds in range(8):
        for name, kw in ARMS.items():
            _, c = T.run(
                state, 1000 + ds, HORIZON, LR, eval_every=EV, total=TOTAL, **kw
            )
            out["branch"][f"{ck}|{ds}|{name}"] = c
    log("checkpoint", ck, "done")
json.dump(out, open("results/e0c/e0c.json", "w"))
log("done")
