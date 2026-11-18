# Stage H plan: paired data tests by re-weighting. Does the pairing survive, and is the answer still right?

PRE-REGISTERED, as an addition to `PLAN.md`. Written after every result of stages A to D and G was read, while
stage E was running, and before any stage-H run. Stage G's numbers motivated this stage and are its comparison.
They had been seen when the thresholds below were chosen. Ran before this file was committed:
`stage_h.py smoke`, a mechanics check on the CPU with a 0.14M model and a 20-step checkpoint. Its toy numbers are
not evidence for anything below.

## Why

- Stage G made data changes by swapping examples into each batch: the first m of 32 came from one part of the
  training files. Pairing then saved only 1.1 to 6.8x, the null swap (no effect) included. The swapped examples
  differ between the two runs, and stage D's large savings need identical batches.
- The same change can be made without changing the batches. Both runs see the control's batches. The change gives
  each example the weight that turns the control's sampling into the swapped mixture in expectation (importance
  weighting). The two runs then differ only in how much each shared example counts, the kind of change that stage
  D found stays paired.
- Re-weighting and re-sampling are not the same in training. An et al. (2021) report different SGD dynamics, and
  LARE (2026) finds that loss re-weighting at fixed exposure acts as a control that mixture ratios do not capture.
  So the re-weighted test may be cheap but answer a different question. This stage measures both.
- Searches of the 17,000-record OpenAlex harvest (`LITERATURE.md`) and of the web found loss re-weighting used to
  train (DoReMi, LARE, ADAPT) and tiny-learning-rate proxies for data decisions (Wang et al., ICLR 2026). They
  found no use of shared batches with re-weighting to compare two data mixtures.

## Setup

As stage G (`PLAN_G.md`): the same model (4.9M parameters), learning rate 0.002, batch 32, bfloat16,
deterministic kernels, the same three checkpoints rebuilt from their seeds, the same six data orders.
`study.py` and `stage_g.py` are reused unchanged.

- Arms: the control, re-run, and for each of stage G's seven data changes a re-weighted twin (`enc8w` for `enc8`,
  and so on). A re-weighted arm trains on exactly the control's batches. Example i's loss is weighted by
  w_i = (1 - f) + f * [start_i in S] / pi_S. Here f = m/32 is the swapped share, S the subset's files, and pi_S the
  share of the control's start positions that lie in S. The batch loss is the mean of w_i times the example's mean
  token loss. It is unnormalised, so its expected gradient is the swapped design's. `null8w` has w = 1 for every
  example: a pure check of the weighted code path.
- Designs: micro-anneals of 100, 200 and 400 steps from 0.002 to 0 (as G), and 200 steps at constant 0.002, read at
  100 and 200 (G's long continuation at those steps).
- The truth and the swapped comparison are stage G's results. The control's 3 x 6 x 5 readings and the checkpoint
  losses must equal stage G's bit for bit. That is the replication check; if it fails, no rule is evaluated.

Estimated GPU time: about 70 minutes at stage G's speed. Runs after stage F.

## Decision rules

`stage_h.py analyze`, committed with this plan, computes them. All at the end of the 200-step micro-anneal, over
stage G's six arms with an effect (null8 has none), on all 18 pairs.

- **H1 (re-weighting makes the paired data test cheaper).** For each arm, the paired runs needed for a given
  confidence scale as (paired SD / effect)^2. The ratio of that number for the swapped design (stage G) to the
  re-weighted design, geometric mean over the six arms: supported at 4 or more, rejected below 2 (below 1,
  re-weighting needs more runs), inconclusive between. H1 is claimed only if H2 is not rejected: a cheap test of
  a different effect is not a saving.
- **H2 (the re-weighted test estimates the swapped change's effect).** Each arm's ratio of re-weighted to
  swapped mean effect. Supported if all six lie in [0.5, 2]. Rejected if any is 0 or below (wrong direction)
  or three or more lie outside [0.5, 2]. Inconclusive otherwise.

Reported only:

- the replication check;
- every arm's effect, seeds saved and runs for 80% power, in every design;
- null8w;
- the Spearman correlation of the re-weighted 200-step micro-anneal with stage G's truth, the long continuation at
  step 1600, on disjoint orders as in G2, beside the swapped design's over the same six arms;
- the H1 and H2 numbers for the constant-learning-rate design.

## Known limits, stated in advance

- All of stage G's limits: a 4.9M model on Python source, re-weightings of one corpus, a 1600-step truth.
- The weights match the swapped mixture only in expectation. The swapped design puts exactly m subset examples in
  every batch. The re-weighted one has a varying number, on average fewer and weighted more. That raises the
  gradient noise from the subset and changes how often clipping (at 1.0) acts. That is part of what H2 tests.
- pi_S counts the control's start positions; G's sampler moves starts in the last 129 training positions to the
  last allowed one. The two definitions of the mixture differ by about 1e-5.
- Six arms, four of which have large effects within a factor of 1.3 of each other. Each variance in H1 has 15
  degrees of freedom, so a per-arm ratio is uncertain by a factor of about 1.7, and the geometric mean of six by
  about 1.25.
- One learning rate, one checkpoint age, one decay shape; no truth for the re-weighted design itself.
