"""E16 data collection: every configuration on a small grid, fine-tuned on every seed, so that any tuning loop
over the grid can be replayed afterwards without training again. The plan is in PLAN.md.

The fine-tuning and scoring are E12's, unchanged. A configuration is (LR multiple, label smoothing, steps); the
start of every loop is E12's control, (1.0, 0.0, 250).

Usage (with REGORA_DEV=cuda REGORA_EXACT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8):
       python -m studies.e16_gate.run pilot TASK
       python -m studies.e16_gate.run run TASK     (a killed run resumes from its partial file)
"""

import itertools
import os
import sys

LR_MULTS, SMOOTHINGS, STEPS = (0.5, 0.71, 1.0, 1.41), (0.0, 0.1), (200, 250, 300)
GRID = list(itertools.product(LR_MULTS, SMOOTHINGS, STEPS))
START = (1.0, 0.0, 250)
name_of = lambda c: f"lr{c[0]}_ls{c[1]}_st{c[2]}"
ARMS = {name_of(c): dict(lr_mult=c[0], smoothing=c[1], steps=c[2]) for c in GRID}
CONTROL_LR = {"topic": 3.2e-4, "nli": 6.4e-4}  # from E12's tuning
# the pilot runs the start and the two opposite corners only
PILOT_ARMS = [name_of(c) for c in (START, GRID[0], GRID[-1])]
PILOT_SEEDS, SEARCH_SEEDS, TRUTH_SEEDS = (13,), range(301, 309), range(309, 313)
OUT = "results/e16"

if __name__ == "__main__":
    import scale
    from studies.e12_paired_eval import run as e12

    stage, task = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    assert scale.EXACT or scale.DEV != "cuda", "run with REGORA_EXACT=1"
    assert START in GRID and len(ARMS) == 24
    arms = ARMS if stage == "run" else {a: ARMS[a] for a in PILOT_ARMS}
    # E12's collector reads its arms and output directory from its module
    e12.ARMS, e12.OUT = arms, OUT
    seeds = PILOT_SEEDS if stage == "pilot" else [*SEARCH_SEEDS, *TRUTH_SEEDS]
    e12.collect(
        task,
        dict(lr=CONTROL_LR[task]),
        seeds,
        f"{dict(pilot='pilot', run='e16')[stage]}_{task}",
    )
