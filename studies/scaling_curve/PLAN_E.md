# Stage E plan: does the saving last, and does pairing decide better, on the largest model?

PRE-REGISTERED, as an addition to `PLAN.md` and `PLAN_D.md`. Written after stages A, B and C were seen, while
stage D was running and before any stage-D result was read, and before any stage-E run. Ran before this file was
committed: `stage_e.py smoke`, a mechanics check on the CPU with a tiny cell, and `replay.py` on the finished
stage A to C cells (the post hoc replay described below).

## Why

Two things came out of stages A to C that nobody had claimed in advance, and both bear on what Regora can
honestly say to a customer:

- **The saving does not fall to 1.** In `C_long` (0.88M parameters) seeds saved levels off between 3 and 7 for
  small changes at 800 and 1600 steps. That is one cell on a small model, seen after the fact.
- **In decision terms the gap is large.** Replaying the finished cells (`replay.py`, `replay_ABC.txt`): over 439
  (cell, arm, horizon) rows whose direction is established, one run of the control and one of the change on
  different data orders point the wrong way 20.9% of the time, and the same two runs as a pair 4.5%. On the 282
  rows whose direction is established without pairing too, 7.5% against 1.2%. Post hoc, mostly short horizons.

Stage E tests both in advance, where they are least certain: the largest model the laptop can train, a later
checkpoint, the longest horizon, and changes of the size a team actually tries.

## Setup

As `PLAN.md` (same corpus, model family, optimiser, evaluation, seeds and statistics), `study.py` unchanged. One
cell, `E_s4_long`, defined in `stage_e.py`:

- Model 8x512 (25.5M parameters, E11's shape), learning rate fixed at 0.0005 (what `B_s4` tuned to), bfloat16,
  deterministic kernels, batch 32.
- Checkpoint at step 2000 (stages A and B: 600), then 1600 steps; read at 200, 400, 800 and 1600.
- K=3 checkpoints, S=6 data orders, identical twin on order 0 of each checkpoint.
- Arms: control; LR x0.9, x1.1, x1.3; weight decay 0.01 and 0.1; swap of 1 example in 32.

Estimated GPU time: about 4 hours.

## Decision rules

`stage_e.py analyze`, committed with this plan, computes them.

- **E1 (the saving lasts).** At h=1600 the geometric mean of seeds saved over LR x0.9, LR x1.1, weight decay
  0.01 and weight decay 0.1 is at least 3. Supported at 3 or more, rejected below 2, inconclusive between.
- **E2 (a pair decides better on the same compute).** Over the rows of this cell at h=800 and h=1600 whose
  direction is established (the mean paired difference over all 18 pairs is more than two standard errors from
  zero), the mean wrong-direction rate of one pair is at most half that of one run of each on different data
  orders. Every draw is enumerated, as in `replay.py`. Uninformative if fewer than 3 rows qualify or if the
  unpaired rate is below 4%.

Reported only: the full table; runs needed for 80% power, paired and unpaired; the rows of E2 split by whether
the direction is established without pairing too; how often a branch's paired difference at h=200 has the sign
it has at h=1600; the swap arm.

## Known limits, stated in advance

- One cell: one model, one learning rate, one checkpoint age, constant learning rate, 18 pairs. E1 pools four
  arms because one ratio of seeds saved is uncertain by about a factor of 2.5.
- "Established" uses the paired standard error, so it admits small effects that only pairing can see, and on
  those the unpaired comparison is close to a coin toss by construction. The split is reported for that reason.
- The truth in E2 comes from the same 18 pairs the draws are made from.
- The comparison is between runs branched from one checkpoint. A team that trains the control and the change
  from scratch on different seeds also has initialisation noise, which this does not measure (E0: larger).
- 1600 steps from step 2000 is still early training on a small model. Nothing here measures a full training
  run, a 100M+ model, or a fine-tune, where E12 found sharing a seed worth about 1x.
- A direction that is established is not a difference that matters. Whether the change is worth making is the
  customer's call (E0c).
