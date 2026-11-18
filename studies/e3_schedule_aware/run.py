"""E3: does a schedule-aware cut rule beat fixed successive halving?

PRE-REGISTERED before running. Motivated by E2, where a slow-starting config won and fixed early cuts dropped it.

Data
- Grid: learning rate {2e-3, 4e-3, 8e-3} x warmup {50, 200, 400} = 9 configs. 1000 steps, cosine to 10%, eval every 50,
  MPS, 2 threads. 8 fresh seeds 11..18 (never used before). Seed s fixes init (s) and data order (3000 + s).
- This grid differs from E1b/E2 (no LR 1e-3, three warmups instead of two), so it is a new test, not a rerun.

Evaluation
- Every selector is replayed offline on each 3-seed subset (56 subsets) and sees only those seeds.
- Regret is measured on the 5 held-out seeds: heldout_mean[pick] - min(heldout_mean). This fixes E2's flaw where the
  truth included the selector's own seeds.
- Compute = steps trained / (9 * 1000), averaged over subsets.

Selectors
- FIXED(r1, r2, r3): rank by 3-seed mean at each rung, keep 4, 2, 1. All r1 < r2 < r3 from {250, 500, 750, 1000}, and FULL.
- AWARE: rungs at 250, 500, 750 with targets 4, 2, 1. A config may only be cut once step >= its warmup + 200 (margin fixed
  now, not tuned). Protected configs always survive a rung; the remaining slots (at least 1) go to the best unprotected
  configs. The final pick is the lowest 3-seed mean at step 1000 among survivors.

Decision rules
- H1 (confirms E2's post hoc finding): FIXED(250, 750, 1000) has lower regret than FIXED(250, 500, 750) by at least 0.005.
- H2 (the new claim): let C be the fixed schedules with compute <= AWARE's compute + 0.02.
    AWARE is SUPPORTED   if its regret is lower than every schedule in C by at least 0.005;
    AWARE is WORSE       if some schedule in C has regret lower than AWARE's by at least 0.005;
    otherwise EQUIVALENT (no evidence either way).
  The 0.005 margin is about half the standard error of an 8-seed mean in E2 (seed SD 0.03).

Known limits: subsets overlap, so replays are not independent. One model, one dataset. AWARE uses only warmup length;
it says nothing about other causes of crossing curves.
"""

import itertools
import json
import time

import torch

import tinygpt as T

torch.set_num_threads(2)
LEARNING_RATES = (2e-3, 4e-3, 8e-3)
WARMUPS = (50, 200, 400)
SEEDS = tuple(range(11, 19))
STEPS = 1000
EVAL_EVERY = 50

out = dict(
    config=dict(
        lrs=LEARNING_RATES,
        warms=WARMUPS,
        seeds=SEEDS,
        steps=STEPS,
        device=T.DEV,
        torch=torch.__version__,
    ),
    curves={},
)
start = time.time()
# seed-major order, so a partial file always holds complete seeds
for seed, lr, warm in itertools.product(SEEDS, LEARNING_RATES, WARMUPS):
    _, curve = T.run(
        T.new_state(seed),
        3000 + seed,
        STEPS,
        lr,
        eval_every=EVAL_EVERY,
        total=STEPS,
        warm=warm,
    )
    out["curves"][f"{lr}|{warm}|{seed}"] = curve
    print(
        f"[{time.time() - start:5.0f}s] lr {lr} warm {warm} seed {seed}: final {curve[-1][1]:.4f}",
        flush=True,
    )
    json.dump(out, open("results/e3/e3_partial.json", "w"))
json.dump(out, open("results/e3/e3.json", "w"))
