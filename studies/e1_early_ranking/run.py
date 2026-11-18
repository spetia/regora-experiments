"""E1: does early ranking of configurations predict final ranking?

PRE-REGISTERED before running:
- Grid: LR in {5e-4, 1e-3, 2e-3} x weight decay in {0, 0.1} = 6 configs; 3 seeds each (init seed = data seed offset = s); 600 steps,
  warmup 50, cosine to 10% over 600, clip 1.0, eval every 50. Device mps, 2 CPU threads (machine is shared).
- Per config: mean val loss over the 3 seeds at each eval step.
- Metrics: Kendall tau between config ranking at step h and at step 600, h in {100,200,300,400};
  regret of 'pick the best at step h' = final loss of that pick minus best final loss; both on seed-means.
  Also the same on the 3 LR-only configs (wd=0) where effect sizes are larger.
- Decision rule: successive halving is supported if tau >= 0.7 at h=200 (one third of the budget) on the 6-config grid
  AND regret at h=200 <= 0.01. Otherwise not supported by this evidence.
- Seed-level: report the SD of final val loss across seeds per config to judge ties.
"""

import json, time, itertools
import torch
import tinygpt as T

torch.set_num_threads(2)
LRS, WDS, SEEDS, STEPS, EV = (5e-4, 1e-3, 2e-3), (0.0, 0.1), (0, 1, 2), 600, 50
out = dict(
    config=dict(
        lrs=LRS,
        wds=WDS,
        seeds=SEEDS,
        steps=STEPS,
        device=T.DEV,
        torch=T.torch.__version__,
    ),
    curves={},
)
t0 = time.time()
for lr, wd, s in itertools.product(LRS, WDS, SEEDS):
    _, c = T.run(T.new_state(s), 2000 + s, STEPS, lr, eval_every=EV, total=STEPS, wd=wd)
    out["curves"][f"{lr}|{wd}|{s}"] = c
    print(
        f"[{time.time()-t0:5.0f}s] lr {lr} wd {wd} seed {s}: final {c[-1][1]:.4f}",
        flush=True,
    )
json.dump(out, open("results/e1/e1.json", "w"))
