# E11 plan: do the supported parts hold on larger models and a second task?

PRE-REGISTERED before the test run. Written after E9 and E10. Everything measured so far used one byte-level corpus
and transformers under 5M parameters, so nothing yet speaks to scale or to another kind of model.

## Claims

1. Pairing: branches from one checkpoint on shared data need at least 2x fewer seeds than unpaired runs (E0, E0b).
2. GPU non-determinism under mixed precision does not swamp that pairing (E0d tested a tiny model for 200 steps).
3. E9's two reliable diagnoses, "LR too low" and "evaluation too noisy", stay right, without false alarms.
4. The short-lived advantage of a lower LR on a healthy run (E9, E10) is also there at this scale.

## What is fixed in advance, and what the pilot was for

- The diagnosis rule is E9's `diagnose`, imported and not changed: same thresholds, same 300-step probes, same three
  data orders. No threshold was fitted to these models.
- Checkpoint at step 1000, probes of 300 steps and a follow-through of 1500 steps were chosen before any run here.
  Runs are longer than E9's (300, 300, 900) because the models are larger; nothing else was tried.
- I looked for a rule that predicts from the first 300 steps whether a lead will last, on E9's follow-through
  branches, and found none (a growing lead lasted in 84 of 105 cases, a shrinking one in 36 of 49). So no such rule
  is tested here; how long leads last is measured and reported.
- Pilot: one seed (11) per task, all five conditions, 300-step branches only, to check that the code runs, nothing
  diverges unexpectedly and the tuned LR is not on the edge of its grid. Its outcome is recorded at the end of this
  file. Seed 11 is not used below.

## Data

- Language: byte-level GPT, 8 layers x 512 wide, 25.6M parameters, on enwik8 (90M bytes train, 5M validation).
  AdamW, batch 64 x 256 bytes, gradient clipping at 1.0. Metric: loss on 40 fixed validation batches.
- Vision: ResNet-18 (CIFAR form), 11.2M parameters, on CIFAR-100 (45,000 train, 5,000 validation). SGD with momentum
  0.9, batch 128, weight decay 5e-4, random crop and flip. Metric: loss on the 5,000 validation images.
- Both: 100 steps of LR warmup, then constant. bfloat16 autocast, default (non-deterministic) kernels, one L40S.
- Healthy LR per task: lowest mean validation loss at step 1300 over a 2x grid, seeds 3 and 4. Language: 1.25e-4 to
  2e-3, picked 1e-3. Vision: 1.5625e-3 to 5e-2, picked 1.25e-2.
- Conditions: healthy; lr_low (1/12x); mild_low (1/4x); mild_high (3x, reported only); eval_noise (metric read on one
  fresh batch).
- 10 fresh seeds 101..110 per task and condition: 100 cases.
- Per case: train 1000 steps, reading the metric every 100; read it 4 more times; control, half-LR and double-LR
  branches on 3 data orders, read every 100 steps to 300 (the run's own data order runs on to 1500).
- Healthy cases also get a second control branch (identical settings, to measure non-determinism) and a
  weight-decay branch (language 0 to 0.1, vision 5e-4 to 1e-3: a small intervention).

## Decision rules (`analyze.py`)

- H1 pairing. For each task and each of half LR, double LR and weight decay, on the 10 healthy checkpoints at 300
  steps: 2 x (variance of the control across data orders) / (variance of arm minus control across data orders) is at
  least 2. Variances are pooled within checkpoints. Runs with a non-finite reading are left out and counted.
- H2 non-determinism. For each task: the RMS gap between the two identical control branches at 300 steps is at most
  0.1 of the control's SD across data orders.
- H3 diagnosis. Of the 60 lr_low, mild_low and eval_noise cases, at least 85% are right, and at least 75% on each
  task.
- H4 false alarms. At most 10% of the 20 healthy cases are told "LR too low" or "evaluation too noisy".
- H5 advice. When the rule says "LR too low", the doubled-LR branch is below the control 900 steps after the
  checkpoint in at least 80% of cases.
- H6 transient. On the 20 healthy cases the half-LR branch is below the control at 300 steps in at least 50%, and of
  those at most 50% are still below it at 1500 steps.
- H7 exact replay. The healthy cases are run again with deterministic kernels (`REGORA_EXACT=1`, same seeds,
  300-step branches). Both the H1 and the H2 limits hold on every task.
- Reported only: E9's suggested repair (the advised branch itself must beat the control on all 3 data orders, no
  threshold); how often healthy runs are told "LR too high"; what a 3x LR reads as; how often a branch's sign at
  each 100 steps matches its sign at 1500; non-determinism at each 100 steps up to 1500; the weight-decay effect and
  its interval; diagnosis cost.

## Known limits

Two tasks, one model each, one GPU type. Runs of 2500 steps, which is early training for both models. Constant LR
after warmup, no schedule. Single injected causes. Healthy LRs come from a 2x grid on two seeds. H4 and H6 rest on 20
cases, so each is decided by a handful of runs. Each task's validation set is used for tuning and for scoring; no
threshold is fitted, so this affects the healthy LR only.

## Pilot outcome (`results/e11/pilot_*.json`, seed 11, one case per condition and task)

Changes made after the pilot, before this plan was committed:

- Vision LR grid. The first grid (1.25e-2 to 2e-1) put the best LR on its lower edge, so it was extended downward by
  three points. The pick did not move (1.25e-2). The first grid's files are in `results/e11/first_grid/`.
- H7 was added. On the language model two identical branches differed at 300 steps by 0.52 of the SD across data
  orders (vision: 0). A short check showed deterministic kernels replay exactly on both models at about the same
  speed, so the remedy is tested rather than assumed.
- The reported-only repair rule was added to the analysis.

No threshold, horizon or step count was changed. What the pilot leads me to expect:

- H2 fails on the language model. H1 is close to its limit there for half LR (2.0x on one checkpoint).
- H3 fails. On both tasks the rule was right on 1 of 3 in-scope cases (the noisy evaluation). LR 1/4x read as "LR too
  high" and LR 1/12x as healthy: at 300 steps the half-LR branch is ahead of the double-LR branch even when the LR
  is too low. The healthy case also read as "LR too high" on both tasks. This is the transient from E9 and E10, and
  it is larger here.
- H5 will rest on few cases, since the rule rarely says "LR too low".
- H4 and H6 cannot be judged from one case with 300-step branches.

The run goes ahead unchanged: the point is to measure these at scale on fresh seeds, and the 1500-step
follow-through is what a better rule would have to be designed on.
