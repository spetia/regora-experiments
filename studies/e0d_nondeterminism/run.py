"""E0d: how much pairing survives device non-determinism?

PRE-REGISTERED before running:
- Device from REGORA_DEV (mps now; cuda later on a GPU box). Checkpoint 0 of E0b is rebuilt ON CPU (identical to E0b), then branches run on the device.
- Same control config, same data seed, repeated R=3 times, for data seeds 0..3 (12 runs, 200 steps, eval every 50).
- sigma_nd = pooled SD of the endpoint (mean val loss at h=100,150,200) across repeats within a data seed. On CPU it is exactly 0.
- An intervention's pairing 'survives' if |effect| > 2.8*sigma_nd*sqrt(2)/sqrt(8) (MDE at n=8, both arms noisy). Compare against CPU effects from E0b/E0c, checkpoint 0 only:
  wd0.1 0.0011, wd1.0 0.0133, clip0.25 0.0093, 2xLR -0.0382, 0.5xLR +0.0246.
- Also reports max abs difference of repeat curves at each eval, and whether any two repeats are bitwise identical.
"""

import json, os, time
import numpy as np
import tinygpt as T

LR, CK, HORIZON, EV, TOTAL = 1e-3, 300, 200, 50, 600
DEV = T.DEV
T.DEV = "cpu"  # checkpoint is built on CPU so it is identical to E0b
state, _ = T.run(T.new_state(0), 900, CK, LR, eval_every=EV, total=TOTAL)
T.DEV = DEV
out = dict(device=T.DEV, torch=T.torch.__version__, reps={})
t0 = time.time()
for ds in range(4):
    for r in range(3):
        _, c = T.run(state, 1000 + ds, HORIZON, LR, eval_every=EV, total=TOTAL)
        out["reps"][f"{ds}|{r}"] = c
    print(f"[{time.time()-t0:5.0f}s] ds {ds}", flush=True)
ep = lambda c: np.mean([v for _, v in c][1:])
sds = [
    np.std([ep(out["reps"][f"{ds}|{r}"]) for r in range(3)], ddof=1) for ds in range(4)
]
sig = float(np.sqrt(np.mean(np.square(sds))))
out["sigma_nd"] = sig
ident = any(
    out["reps"][f"{ds}|0"] == out["reps"][f"{ds}|{r}"]
    for ds in range(4)
    for r in (1, 2)
)
mde = 2.8 * sig * np.sqrt(2) / np.sqrt(8)
print(
    f"device {T.DEV}: sigma_nd {sig:.5f}; any bitwise-identical repeat pair: {ident}; MDE(n=8, both arms noisy) {mde:.5f}"
)
for k, e in dict(
    wd01=0.0011, wd10=0.0133, clip025=0.0093, lr2x=-0.0382, lr05=0.0246
).items():
    print(f"  {k}: |effect| {abs(e):.4f} survives: {abs(e) > mde}")
json.dump(out, open(f"results/e0d/e0d_{T.DEV}.json", "w"))
