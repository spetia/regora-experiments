# Stage D plan: what sets how long a paired comparison lasts?

PRE-REGISTERED, as an addition to `PLAN.md`. Written after the stage-A and stage-B results were seen and before
any stage-D run. Stage C was running while this was written; nothing of it had been read except the learning
rate its first cell tuned to. Ran before this file was committed: `smoke_d.py`, a mechanics check on the CPU with
tiny cells, which asserts the wiring and reads no effect or variance from a real cell.

## Why

Stages A and B rejected the plan's two main claims: the inverse-square law at h=200 (H1, H1b) and the
extrapolation by model size (H3). What they showed instead, post hoc:

- At h=25 the saving is of the same order at every size and close to inverse-square in the change size.
- It decays with horizon in every cell: the paired noise of a 1% learning-rate change grows until it is of the
  order of the noise between two data orders.
- How fast it decays is not ordered by model size (25.5M holds its saving longer than 4.9M and 14.4M). The tuned
  learning rate halves with each size step, so size and learning rate are confounded in stages A and B.

Stage D separates them, and asks whether the decay is a property of training or of bfloat16 rounding.

## The idea being tested

Two runs that start from the same checkpoint and differ by a small change drift apart at a rate set by the
training dynamics, not by the size of the change: a smaller change starts closer and takes longer to drift, but
ends in the same place. The rate rises with the learning rate and, at a fixed learning rate, with model size.
If so, the saving from pairing has a shelf life, and a change of one part in a million loses it too.

## Setup

As `PLAN.md` (same corpus, model family, optimiser, evaluation, seeds and statistics), with `study.py` unchanged
and three differences, all in `stage_d.py`:

- The learning rate of a cell is fixed, not tuned. It is passed as a one-value tuning grid, so the result file
  still records the loss at that learning rate on the two tuning seeds.
- Arms ("probe"): control, LR x(1 + 1e-6), LR x(1 + 1e-4), LR x1.01, weight decay 0.1, swap of 1 example in 32.
- Horizons 25, 50, 100, 200, 400, 800. K=4 checkpoints, S=6 data orders, checkpoint at step 600, batch 32,
  deterministic kernels, identical twin on order 0 of each checkpoint.

| Cell | Model | Learning rate | Precision | Purpose |
|---|---|---|---|---|
| D_s2_lr2e-3 | 6x256 (4.9M) | 0.002 | bfloat16 | repeats A_s2 with the new arms and h=800 |
| D_s2_lr2e-3_fp32 | 6x256 | 0.002 | fp32 | precision |
| D_s2_lr5e-4 | 6x256 | 0.0005 | bfloat16 | learning rate |
| D_s2_ckpt2e-3_branch5e-4 | 6x256 | checkpoint 0.002, every branch 0.0005 | bfloat16 | learning rate now, or the state reached |
| D_s2_lr1e-3 | 6x256 | 0.001 | bfloat16 | middle point for D1 and D2 |
| D_s1_lr1e-3 | 4x128 (0.88M) | 0.001 | bfloat16 | size at a fixed learning rate |

Run in that order. A cell dropped for time or memory is reported as not run. Estimated GPU time: about 6 hours.

## Decision rules

The statistic is seeds saved for LR x1.01 at h=200 (H1's and H3's). Each comparison is a ratio of two cells,
read three ways: **supported** at 4 or more, **rejected** between 0.5 and 2, **inconclusive** otherwise.
`analyze_d.py`, committed with this plan, computes them.

- **D1 (learning rate).** D_s2_lr5e-4 over D_s2_lr2e-3. Supported means a lower learning rate keeps the saving
  longer at a fixed size.
- **D1b (learning rate now, not history).** D_s2_ckpt2e-3_branch5e-4 over D_s2_lr2e-3: the same checkpoints,
  branched at a quarter of the learning rate. Supported means the rate during the branch is what matters.
- **D2 (size at a fixed learning rate).** D_s1_lr1e-3 over B_s3 (14.4M at 0.001, measured: 13.8). Supported means
  a smaller model keeps the saving longer at the same learning rate.
- **D3 (precision).** D_s2_lr2e-3_fp32 over D_s2_lr2e-3. Supported means bfloat16 rounding is what amplifies
  small changes; rejected means the decay is there in full precision too.
- **D4 (no change is small enough).** In D_s2_lr2e-3, seeds saved for LR x(1 + 1e-6) at h=800 is below 10. If
  that arm's readings equal the control's exactly, the test is uninformative and is reported as such. The fp32
  cell's value is reported.
- **D5 (reproducibility).** D_s2_lr2e-3 has A_s2's seeds, learning rate and kernels, so every reading the two
  share (control, LR x1.01, weight decay 0.1, swap-1, twin; h=25 to 400) should be bitwise equal. A Windows
  update and three restarts lie between the two runs. Reported as equal or not, with the largest difference.

Reported only: every table; the shelf life of each arm (the first horizon at which seeds saved is below 10);
how the paired noise of the three learning-rate arms grows with horizon, and the lag between them; the loss at
each cell's learning rate; D_s2_lr1e-3 as the middle point of D1 and D2.

## Known limits, stated in advance

- Two of the eight numbers the rules use were known when this was written: A_s2 (which D_s2_lr2e-3 should
  reproduce: 6.9) and B_s3 (13.8). D1 and D2 therefore each rest on one new number against a known one.
- In D1 the checkpoint is trained at the cell's learning rate, so a lower rate also means a less-trained
  checkpoint at step 600. D1b is there to separate the two; it changes the learning rate at the branch, which
  no stage-A cell does.
- B_s3 has 5 orders and the stage-D cells 6. A single ratio of seeds saved is uncertain by about a factor of
  2.5 either way, which is why the rules need a factor of 4 and call 2 to 4 inconclusive.
- One model family, one dataset, one GPU, two sizes, three learning rates, early training, constant learning
  rate. Nothing here says what the shelf life is for a 100M+ model, a fine-tune, or a schedule with decay.
- LR x(1 + 1e-6) changes each update by less than a 32-bit float resolves in most weights, so it acts as a
  scattering of last-bit flips, not a smooth change. That is the intended probe, and the reason D4 has an
  uninformative clause.
- A shelf life read off six horizons is coarse: it is only known to within a factor of 2.
