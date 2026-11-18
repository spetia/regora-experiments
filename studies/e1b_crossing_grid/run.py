"""E1b: early-ranking validity on a grid built to contain crossings and instability.

PRE-REGISTERED before running:
- Grid: LR in {1e-3, 2e-3, 4e-3, 8e-3} x warmup in {50, 300} = 8 configs; 3 seeds; 1000 steps, cosine to 10% over 1000, wd 0, clip 1.0,
  eval every 50. Device mps, 2 CPU threads (shared machine). Slow-start arms (warmup 300) and unstable LR (8e-3) are there on purpose.
- Truth = seed-mean final (step 1000) val loss. Diverged/NaN runs count as val loss = +inf for ranking and are reported.
- Metrics, on seed-means: Kendall tau between ranking at h and final, h in {250, 333~350, 500, 750}; regret of picking best at h.
  Crossings: number of config pairs whose seed-mean order at some eval step h < 1000 differs from their final order.
- Successive halving simulation (8 -> 4 at step 250 -> 2 at step 500 -> 1 at step 750), run separately per seed using only that seed's
  val loss to rank. Report, for each seed: final pick, regret vs truth, and compute used as a fraction of running all 8 to step 1000
  (= (8*250 + 4*250 + 2*250 + 1*250)/(8*1000) = 0.469).
- Decision rule (same thresholds as E1): halving is supported if tau >= 0.7 at h=350 on seed-means AND mean regret over the 3 per-seed
  simulations <= 0.01. If crossings == 0 the test is declared uninformative regardless of the pass.
"""

import json, time, itertools
import torch
import tinygpt as T

torch.set_num_threads(2)
LRS, WARMS, SEEDS, STEPS, EV = (1e-3, 2e-3, 4e-3, 8e-3), (50, 300), (0, 1, 2), 1000, 50
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
for lr, w, s in itertools.product(LRS, WARMS, SEEDS):
    _, c = T.run(
        T.new_state(s), 3000 + s, STEPS, lr, eval_every=EV, total=STEPS, warm=w
    )
    out["curves"][f"{lr}|{w}|{s}"] = c
    print(
        f"[{time.time()-t0:5.0f}s] lr {lr} warm {w} seed {s}: final {c[-1][1]:.4f}",
        flush=True,
    )
    json.dump(out, open("results/e1b/e1b_partial.json", "w"))
json.dump(out, open("results/e1b/e1b.json", "w"))
