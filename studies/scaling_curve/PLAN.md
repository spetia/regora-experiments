# Scaling study plan: what sets the value of a paired comparison?

PRE-REGISTERED. Written before any outcome data existed. Three things ran before this file was committed: the
install, a speed benchmark (`study.py bench`, steps per second only), and a mechanics check (`smoke.py`, which
asserts exact replay, an untouched checkpoint and correct swap arms on 30-step runs and reads no effect or
variance). Local study, not part of the Regora repository; it follows that repository's rule that the plan and
decision rules are fixed before the run and reported against even when inconvenient.

## Question

Regora's measured benefit is "seeds saved by pairing": how many fewer runs a paired comparison needs than an
unpaired one, at equal interval width. E11 found 4x to 12x for weight decay and 2x or less for learning-rate
changes, on two models. Nobody has measured how that number moves with model size, the size of the change, the
kind of change, the horizon, batch size, data size or checkpoint age. A curve to show a customer needs those,
and it needs a test of whether small models predict larger ones.

## The idea being tested

For a small change of relative size eps, a branch's loss differs from its control's by about eps x g(order),
where g is how sensitive this checkpoint is on this data order. Then the paired noise is proportional to eps,
the unpaired noise does not depend on eps, and

    seeds saved  ~  2 x (seed SD)^2 / (eps^2 x SD(g)^2 + 2 x (non-determinism SD)^2)

Consequences, each a claim below: the saving grows as 1/eps^2 as the change gets smaller, whatever kind of
change it is (H1, H1b); it is capped by whatever non-determinism the kernels add (H4); a change to the training
data removes the shared batches the pairing rests on (H5); and whether any of it holds across model sizes is an
empirical question (H3). E0b and E0c are consistent with the first on one tiny model (paired SD 0.0002 to 0.008
across effects of 0.0004 to 0.040), which is why it is being tested, and is not evidence for it: those numbers
were known.

## Setup

Byte-level GPT on the Python standard-library source of the local interpreter (12.0 MB, sorted file order,
site-packages excluded, first 90% train, last 10% validation; the kind of corpus E0 to E10 use; enwik8, E11's
corpus, could not be downloaded). Context 128, AdamW, 100 steps of warmup then constant LR, gradient clip 1.0,
weight decay 0 in the control, bfloat16 autocast, deterministic kernels, torch 2.14.1+cu126, one RTX 4060
laptop GPU. The batch at a step is a function of (data seed, step). Validation loss on a fixed set of 20 batches
of 64.

Per cell: the healthy LR is the grid value (factor-2 grid per model size, in `study.py`) with the lowest mean
validation loss at the end of the branch horizon, on tuning seeds 3 and 4, tuned per cell. Then K checkpoints
(init seed 1000+k, data seed 2000+k) at the checkpoint step; from each, S data orders (seed 100(k+1)+s); on each
order a control and every arm, read at every horizon. An identical second control on order 0 of each checkpoint
checks exact replay.

Arms (13 in "full"):

- Learning rate x1.01, x1.03, x1.1, x1.3, x2.0, x0.5.
- Weight decay 0.01, 0.1, 1.0.
- Data swap: 1, 8 or all 32 of a batch's 32 examples replaced by examples from another stream (scaled by batch
  size in the batch cells). This stands for a change to the training data. Swapping all 32 is a control on a
  different data order, so it is the unpaired case measured the paired way: its saving should be about 1.

"Lean" cells drop x1.03, x1.3 and swap-32. The long cell runs x1.01, x1.1, x2.0, wd 0.1 and swap-1.

| Cell | Model (layers x width, parameters) | Design |
|---|---|---|
| A_s0, A_s1, A_s2 | 2x64 (0.14M), 4x128 (0.88M), 6x256 (4.9M) | full arms, K=4, S=6, batch 32, checkpoint 600, horizons 25, 50, 100, 200, 400 |
| B_s3 | 8x384 (14.4M) | held-out size; lean arms, K=4, S=5 |
| B_s4 | 8x512 (25.5M, E11's shape) | held-out size; lean arms, K=3, S=5 |
| C_batch8, C_batch128 | 4x128 | batch size (K=3 for 128) |
| C_data100k, C_data1m | 4x128 | training bytes (base: 10.8M) |
| C_age150, C_age2400 | 4x128 | checkpoint step |
| C_default_kernels | 4x128 | default kernels, identical twin on every order |
| C_fp32 | 4x128 | no autocast |
| C_long | 4x128 | horizons 100, 200, 400, 800, 1600; K=3 |

Stage C cells use lean arms, K=4, S=5 unless stated. Estimated GPU time from the benchmark: A about 1.5 hours,
B about 2.2 hours, C about 4 hours.

Order of running: stage A, then `analyze.py predict` freezes the stage-B predictions to a file which is committed
by the driver (`run_all.py`) with no human in between, then stage B, then stage C. A cell dropped for time or
memory is reported as not run.

## Measures

For an arm at a horizon, with variances taken across data orders within a checkpoint and averaged over
checkpoints (E11's pooling): seeds saved = (variance of arm + variance of control) / variance of the paired
difference. Also reported: the mean effect, the paired SD, the pooled correlation, paired SD / |effect|, runs
needed for 80% power paired and unpaired, a bootstrap interval for seeds saved, and the spread of the effect
between checkpoints beyond what order noise explains. A branch whose loss is not finite is reported as diverged
and never imputed.

## Decision rules

All at h=200 unless stated. `analyze.py`, committed with this plan, computes them.

- **H1 (inverse-square in change size).** In each of A_s0, A_s1, A_s2: the slope of log(seeds saved) against
  log(eps) over LR x1.01, x1.03, x1.1 lies in [-2.4, -1.6].
- **H1b (not specific to the learning rate).** In each stage-A cell the slope from weight decay 0.01 to 0.1 lies
  in [-2.4, -1.6].
- **H2 (horizon).** In stage A, seeds saved at h=400 is below seeds saved at h=50 for every arm except swap-32
  in every cell.
- **H3 (small models predict larger ones).** Per arm, fit log(seeds saved) = a + b log(parameters) on the three
  stage-A cells, freeze it, and predict B_s3 and B_s4. Supported if at least 75% of the predictions (9 arms x 2
  cells) are within a factor of 2 of the measured value. A diverged arm counts as a miss.
- **H4 (non-determinism ceiling).** C_default_kernels against A_s1: seeds saved for LR x1.01 under default
  kernels is below a quarter of its value under deterministic kernels, while for LR x2.0 the two are within a
  factor of 1.5. If the identical twins under default kernels are bitwise equal the test is uninformative and is
  reported as such.
- **H5 (data changes erode pairing).** In each stage-A cell: seeds saved for swap-1 > swap-8 > swap-32, and
  swap-32 lies between 0.5 and 2.

Reported only: every stage-C cell (batch size, data size, checkpoint age, precision, long horizon); paired SD /
|effect| and the runs needed; the between-checkpoint spread of each effect; how often a branch's paired
difference keeps its sign from an early horizon to the last; whether the identical twin is exact; a comparison
of B_s4 with E11's published language numbers (different dataset, hardware, kernels, checkpoint step and LR, so
a loose check); wall time.

## Known limits, stated in advance

- One architecture family, one dataset, one GPU, models up to about 25M parameters, early training (checkpoint
  at step 600 in the base), constant LR. Nothing here measures 100M+ models or fine-tuning; E12 found seed
  sharing worth about 1x for whole fine-tunes.
- Seeds saved is a variance ratio. A large ratio for a tiny change says the change is cheap to measure, not that
  it matters (E0c), and a short-horizon effect need not match the effect after full training (E11: sign at 300
  steps matched 1500 in 54 of 80).
- Three sizes is a thin basis for a fit. H3 is the test of whether extrapolation is safe; a pass on two held-out
  sizes within one order of magnitude does not license extrapolation by several orders.
- With S=5 or 6 orders and K=3 or 4 checkpoints a variance ratio has 12 to 20 degrees of freedom; single ratios
  are uncertain by roughly a factor of 2, which is why H3's and H5's tolerances are a factor of 2.
- At the tuned LR the first-order effect of an LR change may be near zero, so paired SD / |effect| can be large
  or undefined for LR arms. H1 is about the variance ratio, which does not depend on the mean effect.
- bfloat16 rounding could add a floor of its own under the smallest change. C_fp32 is there to show it.
- The LR is tuned on two seeds per cell. A tuned LR at the edge of its grid is flagged in the output.
- The stage-C cells change one factor at a time around one model; interactions are not measured.
