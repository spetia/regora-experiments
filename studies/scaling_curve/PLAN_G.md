# Stage G plan: paired micro-anneals for data decisions. Cheap, and still right?

PRE-REGISTERED, as an addition to `PLAN.md`. Written after stages A to C and four of stage D's six cells were
read (D1 and D1b supported, D3 rejected, D4 passed, D5 bitwise equal), and before any stage-E, F or G run.
Ran before this file was committed: `stage_g.py smoke`, a mechanics check on the CPU with a 0.14M model and a
20-step checkpoint. It printed effects and ratios for that toy configuration, which were seen; they are not
evidence for anything below.

## Why

- Stage D found that the learning rate during a branch sets how long two runs stay paired. At a quarter of the
  training learning rate, small changes stayed above 1,000x through 800 steps; at the training rate they were
  below 10x by 200 steps. It also showed the catch: at the low rate the same changes had different effects
  (LR x1.01 helped instead of hurting; weight decay 0.1 hurt instead of helping). A cheap comparison can answer
  a different question.
- Labs already test data while the learning rate is low. Llama 3 graded data sets by annealing a half-trained
  model to zero on them; OLMo 2 and Olmo 3 use short "micro-anneals". DataDecide (2025) and Signal and Noise
  (2025) measure how often small experiments pick the right data, but their abstracts mention neither
  shared-order designs nor the learning rate during the test.
- This tests both halves on a small model: does a micro-anneal make the paired comparison of two data mixtures
  cheap, and is its verdict the one a long continuation gives?

## Setup

As `PLAN.md` (corpus, model family, optimiser, evaluation, kernels, seeds); `study.py` unchanged. `stage_g.py`
adds batches that draw some examples from a subset of the training files, and learning-rate schedules.

- Model 6x256 (4.9M parameters), learning rate 0.002 (what `A_s2` tuned to), batch 32, bfloat16,
  deterministic kernels. K=3 checkpoints at step 1500 at constant learning rate 0.002 (init seed 1000+k, data
  seed 2000+k); S=6 data orders per checkpoint (seed 100(k+1)+s).
- Data changes (arms): during the branch, the first m of each batch's 32 examples come from a subset of the
  training files, drawn from their own stream; the other examples are the control's. Subsets by path, with
  their share of the training bytes: `encodings/` (13.2%, codec tables), `idlelib/` (11.6%), top-level modules
  (39.8%), `asyncio/` (4.8%), test files (6.9%), and all files.
  - null8: 8 of 32 from all files, so no systematic effect;
  - enc8, enc2 (8 and 2 of 32 from `encodings/`), idle8, top8, async8, test8 (8 of 32 each).
- Two designs, every arm on every order:
  - **micro-anneal**: the learning rate decays linearly from 0.002 to 0 over L = 100, 200 or 400 steps, read
    at the end;
  - **long continuation** (the truth): 1200 steps at 0.002, then a linear decay to 0 over 400 steps; read at
    100, 200 and 400 (the constant-learning-rate proxy), 1200, and 1600 (the truth).
- An identical second control on order 0 of each checkpoint checks exact replay.

Estimated GPU time: about 2 hours. Runs after stage F.

## Decision rules

`stage_g.py analyze`, committed with this plan, computes them. Seeds saved is defined as in `PLAN.md`.

- **G1 (a micro-anneal makes the paired comparison cheap).** The geometric mean of seeds saved over the 7 data
  arms at the end of the 200-step micro-anneal, divided by the same at step 200 of the long continuation
  (learning rate still constant). Supported at 4 or more, rejected between 0.5 and 2, inconclusive otherwise.
- **G2 (the micro-anneal ranks the data changes as the long continuation does).** The Spearman correlation over
  the 7 data arms between the mean effects of the 200-step micro-anneal on orders 0 to 2 and those of the long
  continuation at step 1600 on orders 3 to 5, averaged with the swapped split. Supported at 0.75 or more,
  rejected at 0.4 or less, inconclusive between. The constant-learning-rate proxy at step 200 and the 100- and
  400-step micro-anneals are reported the same way.
- **G3 (one paired micro-anneal decides).** Over the 6 arms other than null8, for every comparison whose truth
  (the long continuation at step 1600, on all the orders the comparison did not use) is more than two standard
  errors from zero: the wrong-direction rate of one paired 200-step micro-anneal (control and change on the same
  order, 2 runs) is at most 10%, and at most half the rate of one unpaired micro-anneal comparison (control and
  change on different orders, 2 runs). Uninformative if fewer than 3 arms have an established truth or the
  unpaired rate is below 4%. The constant-learning-rate proxy is reported the same way.

Reported only: every arm's effect and seeds saved in every design and reading; the control's spread across
orders; the identical-twin check; the checkpoint loss.

## Known limits, stated in advance

- A 4.9M model on Python source, with a validation set that is the alphabetical tail of the library. The data
  changes are re-weightings of one corpus, not new sources, and their effects may be large next to real data
  decisions, which makes G2 easier than in practice.
- The truth is a 1600-step continuation, not a full training run. Nothing here says whether either predicts the
  effect at 100M+ parameters or after full training.
- Orders 0 to 2 and 3 to 5 share checkpoints, so the split removes shared data-order noise but not
  checkpoint-specific effects.
- With 7 arms a Spearman correlation is coarse: swapping one adjacent pair moves it by about 0.04.
- One learning rate, one checkpoint age, one decay shape.
- Stage D's numbers suggested G1's direction; G1's threshold was chosen after seeing them.
