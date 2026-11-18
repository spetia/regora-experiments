"""E15 data collection: runs with two faults at once, and with a fault the diagnosis rule has no name for.

The cases are built exactly as in E11 (same models, same branches, same fields), so E9's frozen rule reads them
unchanged. Only the faults differ. The healthy LR is the one E11 tuned. The plan is in PLAN.md.

Usage (with REGORA_DEV=cuda): python -m studies.e15_mixed_causes.run pilot|run lm|vision
"""

import json
import os
import sys

from studies.e11_scale import run as e11

# `train` options reach every training step of the case, so the fault is still there in the probe branches.
HEAVY_DECAY = dict(lm=0.5, vision=1e-2)  # the tasks' own values are 0 and 5e-4
CONDITIONS = lambda name: {
    "healthy": dict(),
    "low_and_noisy": dict(lr_mult=1 / 12, noisy_eval=True),
    "high_and_noisy": dict(lr_mult=3.0, noisy_eval=True),
    "heavy_decay": dict(train=dict(weight_decay=HEAVY_DECAY[name])),
    "low_and_decay": dict(lr_mult=1 / 12, train=dict(weight_decay=HEAVY_DECAY[name])),
}
PILOT_SEEDS, TEST_SEEDS = (12,), range(201, 211)
OUT = "results/e15"

if __name__ == "__main__":
    stage, name = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    tuned = json.load(open(f"results/e11/tune_{name}.json"))
    # E11's collector reads its conditions and output directory from its module
    e11.CONDITIONS, e11.OUT = CONDITIONS(name), OUT
    seeds = dict(pilot=PILOT_SEEDS, run=TEST_SEEDS)[stage]
    follow = dict(pilot=e11.BRANCH_CHUNKS, run=e11.BRANCH_CHUNKS)[
        stage
    ]  # no 1500-step follow-up: nothing here uses it
    e11.collect(
        name,
        tuned,
        seeds,
        f"{dict(pilot='pilot', run='e15')[stage]}_{name}",
        follow,
        e11.CONDITIONS,
    )
