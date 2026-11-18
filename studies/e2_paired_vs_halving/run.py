"""E2: does a paired sequential test (F3) beat rank-and-cut successive halving (F4) at equal compute?

PRE-REGISTERED before running (motivated by E1b, which showed halving drops the slow-starting winner):
- Data: same grid as E1b (LR {1e-3,2e-3,4e-3,8e-3} x warmup {50,300}, 1000 steps, cosine to 10%, eval every 50, MPS, 2 threads),
  but 8 FRESH seeds 3..10 (E1b's seeds 0..2 are not used, so nothing here was seen when the rule was written).
  Seed s fixes init (s) and data order (3000+s) across all configs, so differences between configs are paired.
- Truth = mean final val loss over all 8 seeds. Regret = truth[pick] - min(truth).
- Selectors are replayed offline on every 3-seed subset (56 subsets); each selector sees only its 3 seeds.
    SH(r1,r2,r3): rank by 3-seed mean at each rung, keep 4, 2, 1. All r1<r2<r3 from {250,500,750,1000}, plus FULL (no cuts).
    F3: at rungs 250, 500, 750 the leader is the lowest 3-seed mean; drop config c if the paired differences
        d_s = loss[c,s] - loss[leader,s] satisfy mean(d) - 2.92*sd(d)/sqrt(3) > 0 (one-sided 95% t, df=2, no multiplicity
        correction). Pick the lowest mean at step 1000 among survivors.
  Compute = steps trained / (8 configs * 1000), averaged over subsets.
- Decision rule: F3 is supported iff NO SH schedule (including FULL) has mean compute <= F3's AND mean regret <= F3's.
  Otherwise F3 is not supported on this grid.
- Also reported, not part of the rule: random-pick regret; SD of paired vs unpaired config differences at step 250.
- Known limits: subsets overlap (not independent); truth includes the selector's own seeds; one model, one grid.
"""

import json, time, itertools
import torch
import tinygpt as T

torch.set_num_threads(2)
LRS, WARMS, SEEDS, STEPS, EV = (
    (1e-3, 2e-3, 4e-3, 8e-3),
    (50, 300),
    tuple(range(3, 11)),
    1000,
    50,
)
out = dict(
    config=dict(
        lrs=LRS,
        warms=WARMS,
        seeds=SEEDS,
        steps=STEPS,
        device=T.DEV,
        torch=T.torch.__version__,
    ),
    curves={},
)
t0 = time.time()
for s, lr, w in itertools.product(
    SEEDS, LRS, WARMS
):  # seed-major so a partial file still holds complete seeds
    _, c = T.run(
        T.new_state(s), 3000 + s, STEPS, lr, eval_every=EV, total=STEPS, warm=w
    )
    out["curves"][f"{lr}|{w}|{s}"] = c
    print(
        f"[{time.time()-t0:5.0f}s] lr {lr} warm {w} seed {s}: final {c[-1][1]:.4f}",
        flush=True,
    )
    json.dump(out, open("results/e2/e2_partial.json", "w"))
json.dump(out, open("results/e2/e2.json", "w"))
