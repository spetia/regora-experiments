# scaling_curve (proposal, on a branch for review)

How does "seeds saved by pairing" move with model size, the size and kind of the change, horizon, batch size,
data size, checkpoint age and kernel determinism, and do small models predict larger ones? The plan, the six
decision rules and the limits are in [`PLAN.md`](PLAN.md).

Status: **stages A to H done (D in [`PLAN_D.md`](PLAN_D.md), E in [`PLAN_E.md`](PLAN_E.md), F in
[`PLAN_F.md`](PLAN_F.md), the gate test, G in [`PLAN_G.md`](PLAN_G.md), paired micro-anneals for data decisions,
H in [`PLAN_H.md`](PLAN_H.md), the same data changes made by re-weighting the loss on shared batches); verdicts
below; nothing is queued.** Each queue ended with `publish.sh`, which pushes the raw results and the analysis scripts' output (`results/analysis_*.txt`)
without anyone having read them; this file covers only what is written in it. [`LITERATURE.md`](LITERATURE.md) is
a reading note on what is already known and where the opening may be. `PLAN.md` has the same blob hash as the
commit made in a local repository before any outcome data existed; `study.py` and `analyze.py` did too until
a399b0b formatted every Python file with black after the last run, changing no syntax tree beyond docstring
whitespace (every analysis output reproduces). `results/` holds every finished cell and `prediction.json`, the stage-B predictions frozen by
`run_all.py` before stage B started. `run_all.py` changed once after that: a Windows Update restart killed stage
C two minutes in, and the driver printed ALL DONE although every cell had exited non-zero, so it now lists
failed cells and exits 1. No stage-C result existed at the time; stage C was then run from the start.

Against the committed rules (h=200):

| Rule | Verdict | Numbers |
|---|---|---|
| H1, LR slope in [-2.4, -1.6] | **fails** in 2 of 3 cells | -1.65, -1.42, -0.39 (0.14M, 0.88M, 4.9M) |
| H1b, weight-decay slope in [-2.4, -1.6] | **fails** in all 3 | -1.57, -0.92, -0.39 |
| H2, saving falls from h=50 to h=400 for every arm | **fails** as written | 30 of 33 fall; the 3 that rise are LR x0.5 or x2.0, all between 1.5 and 3.5 |
| H3, stage A predicts 14.4M and 25.5M within 2x | **fails** | 9 of 18 within a factor of 2 (rule: at least 75%) |
| H5, swaps erode pairing, swap-32 near 1 | passes in all 3 | swap-1 6.1, 5.4, 3.4; swap-32 0.89, 0.80, 0.97 |
| H4, default kernels cap the saving | **uninformative**, by its own clause | identical twins under default kernels are bitwise equal on this GPU, so they add no run-to-run noise here |

Identical twins are bitwise equal in all fourteen cells.

H3's misses are one-sided: every miss measured more saving than predicted, by up to 175x (25.5M, LR x1.01:
predicted 0.7, measured 129; weight decay 0.01: predicted 5.2, measured 227). The arms the fit gave a flat
slope (LR x2.0, x0.5, swap-8) are the ones it predicted.

Post hoc, not pre-registered claims. Seeds saved for LR x1.01 at h = 25, 50, 100, 200, 400, with each cell's
tuned LR:

| Cell | LR | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| 0.14M | 0.008 | 2429 | 1374 | 1078 | 1009 | 282 |
| 0.88M | 0.004 | 1484 | 926 | 1027 | 96 | 103 |
| 4.9M | 0.002 | 992 | 980 | 22 | 7 | 4 |
| 14.4M | 0.001 | 2383 | 576 | 67 | 14 | 3 |
| 25.5M | 0.0005 | 2922 | 2683 | 1440 | 129 | 8 |

- At h=25 the saving is of the same order at every size (LR x1.01: 990 to 2900; LR x1.1: 18 to 37; swap-1: 13
  to 20; swap-8: about 2), and the LR x1.01 to x1.1 ratio is near the inverse square.
- It decays with horizon in every cell. By h=400 the three largest models are at 3 to 12 for every small change.
- The decay is not ordered by model size: 25.5M holds its saving longer than 4.9M and 14.4M. A note written
  after stage A ("decays faster the larger the model") was wrong. The tuned LR halves with each size step, so
  size and LR are confounded here; nothing in this design separates them.
- Whether bfloat16 rounding amplifies the small changes is what `C_fp32` is for.

Stage C, reported only (0.88M model; seeds saved for LR x1.01 by horizon; one factor changed per row):

| Cell | Tuned LR | 25 | 50 | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|---|---|---|
| A_s1, the base (checkpoint 600, batch 32, 10.8M bytes) | 0.004 | 1484 | 926 | 1027 | 96 | 103 | | |
| C_long, the base again with K=3 | 0.004 | | | 1243 | 189 | 127 | 6.5 | 6.6 |
| C_default_kernels | 0.004 | 2420 | 1394 | 543 | 175 | 30 | | |
| C_fp32 | 0.004 | 2257 | 1876 | 600 | 219 | 40 | | |
| C_age150, checkpoint at step 150 | 0.004 | 293 | 116 | 13 | 7.6 | 4.6 | | |
| C_age2400, checkpoint at step 2400 | 0.004 | 592 | 1236 | 585 | 189 | 299 | | |
| C_batch8 | 0.002 | 3583 | 1199 | 495 | 68 | 51 | | |
| C_batch128 | 0.008 | 837 | 239 | 47 | 55 | 11 | | |
| C_data100k, 100k training bytes | 0.001 | 894 | 1492 | 394 | 945 | 577 | | |
| C_data1m, 1M training bytes | 0.004 | 1915 | 1029 | 820 | 307 | 15 | | |

- Four cells share the base configuration up to kernels, precision and the number of orders (A_s1, C_long,
  C_default_kernels, C_fp32). At h=200 they give 96, 189, 175 and 219, and at h=400 103, 127, 30 and 40. That
  spread is the uncertainty of one cell's estimate, so differences below about 3x between rows mean little.
- Full precision does not keep the saving longer than bfloat16 on this model.
- The decay is fastest from an early checkpoint (step 150) and slowest from a late one (step 2400).
- In C_long the saving does not fall to 1: it levels off near 6.5 for LR x1.01 at 800 and 1600 steps (LR x1.1:
  3.8 and 4.1; weight decay 0.1: 5.2 and 4.9; swap-1: 2.9 and 3.3; LR x2.0: 2.4 and 1.4).
- Default kernels give different numbers from deterministic ones (up to 0.028 in loss on the same branch) but
  the same numbers on every repeat.

Stage D (4.9M model unless stated; seeds saved for LR x1.01 at h=200 unless stated):

| Rule | Verdict | Numbers |
|---|---|---|
| D1, a lower learning rate keeps the saving longer | **supported** | LR 0.0005: 1567; LR 0.002: 6.9 (ratio 226) |
| D1b, it is the learning rate during the branch, not the checkpoint's history | **supported** | the same checkpoints branched at 0.0005: 3391, against 6.9 (ratio 488) |
| D2, a smaller model keeps it longer at a fixed learning rate | **supported** | at 0.001: 0.88M 1495, 14.4M 13.8 (ratio 108); 4.9M 454 (reported) |
| D3, bfloat16 rounding is what amplifies small changes | **rejected** | fp32 6.1, bfloat16 6.9 (ratio 0.89) |
| D4, a one-in-a-million change loses its pairing too | **passes** | LR x(1+1e-6) at h=800: 2.6 (fp32: 5.4); largest gap from the control 0.0073 |
| D5, the repeat of `A_s2` is bitwise equal | **equal** | 500 shared readings, largest difference 0 |

Post hoc, not pre-registered claims:

- Branched at 0.0005, every small change (LR x(1+1e-6), x(1+1e-4), x1.01, weight decay 0.1) stays above 1,000x
  through h=800, except weight decay 0.1 in `D_s2_lr5e-4` (79 at h=800). Swap-1 stays at 7 to 29; at 0.002 it is
  2 to 18.
- Seeds saved at h=200 for LR x1.01 by learning rate as a fraction of each model's tuned rate: at a quarter, the
  two cells that have it agree (0.88M at 0.001: 1495; 4.9M at 0.0005: 1567); at a half, 454 (4.9M at 0.001); at
  the tuned rate, 6.9 to 1009 across the five sizes (stages A and B). The learning rate relative to the tuned
  rate may be the variable that sets how long pairing lasts. Not tested; the tuning grid is a factor of 2 apart.
- In fp32 the 1e-6 change is indistinguishable from the control at first (seeds saved about 1e11 at h=25) and
  falls to 5.4 by h=800, the shape of exponential divergence. In bfloat16 it starts at about 15,000.
- At the low learning rate the same changes have different effects. LR x1.01 lowers the loss (-0.00019 at h=200)
  where at 0.002 it raises it (+0.00155); weight decay 0.1 raises it (+0.00155) where at 0.002 it lowers it
  (-0.00144). A cheap comparison at a low learning rate can answer a different question. Stage G tests whether
  that matters for data decisions made in a micro-anneal, where the learning rate falls to zero by design.

Stage G (4.9M model, learning rate 0.002, K=3, S=6). A data change replaces the first 8 (or 2) of every batch's 32
examples with examples from one part of the training files. The truth is a 1600-step continuation (1200 steps at
0.002, then a linear decay to 0):

| Rule | Verdict | Numbers |
|---|---|---|
| G1, a 200-step micro-anneal makes the paired comparison at least 4x cheaper than 200 steps at the training learning rate | **rejected** | geometric mean seeds saved over the 7 data arms: 3.8 against 2.4 (ratio 1.57) |
| G2, the micro-anneal ranks the data changes as the long continuation does | **supported** | Spearman 0.89 in both splits (proxy and truth on disjoint orders) |
| G3, one paired micro-anneal decides | **uninformative**, by its own clause | one unpaired comparison was already wrong only 2.8% of the time (paired: 1.9%) |

Identical twins are bitwise equal. The truth, change in validation loss at step 1600: top-level modules (top8)
-0.0040; null8 +0.0003 (not established, as it should be); encodings 2 of 32 +0.0045; test files +0.0168,
encodings +0.0192, idlelib +0.0208, asyncio +0.0224.

Post hoc, not pre-registered claims:

- Pairing saves 1.1 to 6.8x for every 8-of-32 change in every design. That includes null8, whose effect is zero,
  so it is the changed examples, not the effect, that break the pairing: the two runs no longer see the same
  batches, which stage D's 1,000x needs (stage D's swap-1 stayed at 7 to 29 even at the low learning rate). The
  2-of-32 change keeps more: 30x at the end of a 100-step micro-anneal, 3.8x at step 100 at the training rate.
- What the micro-anneal does is cut the noise of each run. The control's spread across orders is 0.0016 at the
  end of the 200-step micro-anneal, and 0.0039 at step 200 at the training rate.
- Runs per arm for 80% power at 200 steps, unpaired / paired (`posthoc_g.py`, output in `results/posthoc_G.txt`):
  - The four large changes need the minimum of 2 in every design, except test files at the training rate (3 / 2).
  - The two small ones need more. Encodings 2 of 32: 29 / 5 at the training rate, 5 / 2 in the micro-anneal.
    Top-level modules: 16 / 8 and 5 / 3.
  - Against an unpaired micro-anneal (what the labs in `PLAN_G.md` already do), pairing saves about 2x on these two.
- No short reading (micro-anneals of 100, 200 and 400 steps; steps 100, 200 and 400 at the training rate) points
  against the truth for any of the six arms with an effect. Unlike stage D, the low learning rate flipped no
  sign here.
- The effects grow with training: at step 1600 they are 1.6 to 2.2x their size at the end of the 200-step
  micro-anneal. The three largest end within 0.0032 of each other and come out in a different order (encodings
  worst at 200 steps, asyncio at 1600).
- A short run at the training rate ranks the arms as well as the micro-anneal (Spearman 0.88), and a 400-step
  micro-anneal ranks them better (0.96).

Decision replay of stages A to C (`replay.py`, output in `replay_ABC.txt`). Post hoc. For every (cell, arm,
horizon) whose direction is established on all pairs, how often a small comparison points the wrong way, with
every draw enumerated:

| Rows | Count | One run each, different orders | One pair | Two runs each | Two pairs |
|---|---|---|---|---|---|
| All with an established direction | 439 | 20.9% | 4.5% | 18.3% | 2.7% |
| Established without pairing too | 282 | 7.5% | 1.2% | 4.6% | 0.5% |
| Established only by pairing | 157 | 45.0% | 10.5% | 42.9% | 6.5% |
| At h=400 | 78 | 17.1% | 5.9% | 13.8% | 3.4% |

The truth comes from the same readings the draws are made from, "established" uses the paired standard error,
and most rows are short horizons on small models. Stage E tests the same comparison in advance on the 25.5M
model at 800 and 1600 steps, together with whether the saving lasts.

Stage E (25.5M model; learning rate 0.0005, fixed in advance as `B_s4`'s tuned rate, so the "at grid edge" flag in
`results/analysis_E.txt` reflects a one-point grid; checkpoint at step 2000, then 1600 steps at a constant rate;
K=3, S=6):

| Rule | Verdict | Numbers |
|---|---|---|
| E1, the saving lasts: at h=1600 the geometric mean seeds saved over LR x0.9, x1.1 and weight decay 0.01, 0.1 is at least 3 | **supported** | 5.3, 16.1, 7.5 and 8.5; geometric mean 8.6 |
| E2, on the same compute one pair points the wrong way at most half as often as one run of each on different orders | **supported** | 7 rows at h=800 and 1600: one run each 22.7%, one pair 7.9% |

Identical twins are bitwise equal.

Reported only, or post hoc; not pre-registered claims:

- E2's rows split as the plan asked. The 5 rows whose direction is established without pairing too: 16.4% (one
  run each) against 3.3% (one pair). The 2 established only by pairing (LR x1.1 at h=1600, weight decay 0.1 at
  h=800): 38.3% against 19.4%. The truth comes from the same 18 pairs the draws are made from.
- Seeds saved at h = 200, 400, 800, 1600: LR x0.9 11.1, 11.2, 9.0, 5.3; x1.1 12.9, 7.4, 4.8, 16.1; x1.3 2.4, 4.0,
  7.7, 13.3; weight decay 0.01 95.8, 43.3, 8.9, 7.5; 0.1 166.5, 27.4, 10.3, 8.5; swap-1 3.6, 2.5, 6.1, 4.2. Weight
  decay starts far higher and falls; at h=1600 every arm is between 4.2 and 16.1, and LR x1.3 rises with horizon.
  This is the level `C_long` showed after the fact on the 0.88M model (about 6.5), here on the largest model. One
  ratio is uncertain by about a factor of 2.5 (bootstrap intervals in `results/analysis_E.txt`).
- Runs per arm for 80% power at h=1600, unpaired / paired: LR x0.9 33.9 / 6.4; x1.1 41.3 / 2.6; x1.3 4.9 / 2.0
  (the minimum); weight decay 0.1 312.7 / 36.8. At h=1600 the 18 pairs establish the direction of the three LR
  changes only; weight decay 0.1 is established at h=200 to 800.
- A pair's sign at h=200 matches its own sign at h=1600 in 89% to 100% of branches for the LR changes, and in 61%
  to 72% for weight decay and swap-1, whose effects at h=1600 are not established.
- One cell: one model, learning rate and checkpoint age, a constant rate, and branches from a shared checkpoint, so
  no initialisation noise. It says nothing about full training runs or fine-tunes. There, E12 found sharing a seed
  worth about 1x for larger changes (half the learning rate, label smoothing 0.1).

Stage F, the gate test (0.88M model, learning rate 0.004, checkpoint at step 600; a configuration is scored by its
validation loss 200 steps after the checkpoint; 8 replicates). A scripted proposer offers learning-rate and
weight-decay changes, one in five a null (the incumbent with 1 example in 32 swapped). The keep-if-better loop runs
each candidate once and keeps it if it beats the best single score so far (96 candidates). The gated loop runs
candidate and incumbent on 4 shared orders and keeps it if the 95% t interval of the paired difference is below
zero (12 candidates). Both use 96 training runs. The truth is each loop's final configuration against the starting
one on 12 orders neither loop saw. Intervals are 95% over the 8 replicates:

| Rule | Verdict | Numbers |
|---|---|---|
| F1, the keep-if-better loop claims more than it achieved | **supported** | claimed minus true +0.0109 ± 0.0040 (claimed 0.0433, true 0.0324) |
| F2, the gated loop's overclaim is at most half of that | **passes** | +0.0020 ± 0.0018 (claimed 0.0222, true 0.0202) |
| F3, the gate keeps at most 5% of the null candidates | **fails** | 2 of 18 (11%); the keep-if-better loop kept 13 of 148 (9%) |
| F4, which loop ends better on the same runs (no direction predicted) | **keep-if-better ends better** | true gain, gated minus keep-if-better: -0.0123 ± 0.0084 |

Post hoc, not pre-registered claims:

- A quarter of the keep-if-better loop's claimed gain was not real (0.0109 of 0.0433), against a tenth of the
  gated loop's (0.0020 of 0.0222), which is small but above zero. The gated loop's real gain was 38% smaller.
- Lowering the learning rate helps at 200 steps here (stage A). The keep-if-better loop ends at 0.36x to 0.57x the
  starting rate in every replicate; the gated loop at 0.46x to 0.9x, after keeping 1 to 6 changes in its 12 rounds.
  The gate kept 22 of the 34 learning-rate moves it was shown and 1 of the 44 weight-decay moves. With eight times
  as many candidates, the cheap loop went further in the direction that helps.
- F3's two nulls: the 18 null t statistics fit the t distribution the gate assumes (Kolmogorov-Smirnov p = 0.87),
  and a correctly calibrated gate keeps 2 or more of 18 nulls 7% of the time, so 18 nulls cannot separate bad luck
  from a lax gate. One passed on a large mean (-0.0042); the other on four nearly equal differences (interval
  half-width 0.0009, against 0.0028 to 0.0089 for 15 of the 18): a 4-pair interval rests on a spread estimated from
  4 numbers.
- Not tested: searching with the cheap loop and checking only its final configuration on fresh paired orders. That
  is what the truth measurement does (24 runs on top of the 96), and it reports the cheap loop's gain without the
  overclaim.
- Limits stated in advance: a scripted proposer over two settings is not an agent, a 0.88M model at 200 steps is
  not a real task, and nothing here touches fabrication, leakage or reward hacking. This landscape has a strong
  real direction; where most candidates do nothing, the keep-if-better loop's claimed gain would be mostly
  overclaim (expected, not measured).

Stage H (stage G's model, learning rate, checkpoints and orders). Each of stage G's data changes is made by
re-weighting instead of swapping: both runs train on the control's batches, and each example's loss is weighted so
that the expected gradient is the swapped design's. The control's checkpoint losses and 90 readings equal stage G's
bit for bit, so the rules are evaluated. At the end of the 200-step micro-anneal, over the six changes with an
effect:

| Rule | Verdict | Numbers |
|---|---|---|
| H1, re-weighting makes the paired data test at least 4x cheaper | **inconclusive, and not claimed** because H2 is rejected | paired runs needed, swapped / re-weighted, geometric mean 3.8 |
| H2, the re-weighted test estimates the swapped change's effect (every ratio in [0.5, 2]) | **rejected** | re-weighted / swapped: encodings 1.10, encodings 2 of 32 0.64, idlelib 2.17, asyncio 3.51, test files 4.57, top-level modules -0.56 |

`null8w`, which weights every example 1, equals the control bit for bit in every design, so the differences come
from the weights, not from the weighted code path.

Post hoc, not pre-registered claims:

- Top-level modules is the one change that lowers the loss, in the truth (-0.0040 at step 1600) and in the swapped
  micro-anneal (-0.0025, lower in 16 of 18 pairs). Re-weighted, it raises it (+0.0014, higher in 18 of 18). Each
  of the other five changes has one sign in all 18 pairs in both designs.
- The three overstated changes (idlelib, asyncio, test files) give their examples the largest weights, 2.9 to 5.9,
  because their files are rare (11.6%, 4.8% and 6.9% of start positions: 1.5 to 3.7 examples a batch). The other
  three give 1.4 to 2.7. Six arms, so this is a pattern, not a test.
- The ranking survives: Spearman with the truth on disjoint orders is 0.83 for both the re-weighted and the swapped
  200-step micro-anneal over these six arms.
- Seeds saved by pairing at the end of the 200-step micro-anneal, re-weighted / swapped: encodings 6.6 / 4.2,
  encodings 2 of 32 50.3 / 12.8, idlelib 7.0 / 3.0, top-level modules 36.7 / 1.7, asyncio 2.0 / 2.8, test files
  2.7 / 5.7.
- At the constant learning rate (reported only) it is further off: H1 3.1, H2 rejected; ratios 1.20, 0.32, 3.90,
  6.72, 8.43, and -2.22 for top-level modules.
- So shared batches with re-weighted losses are not a stand-in for swapping data here: they test a different
  change, as An et al. (2021) and LARE (2026), cited in the plan, warned they might. For data decisions the study's
  answer stays stage G's: pairing saves 1.1 to 6.8x, and a micro-anneal ranks the changes well.
- Limits stated in advance: stage G's (one 4.9M model on Python source, one corpus, a 1600-step truth), one
  learning rate, checkpoint age and decay shape, and no truth for the re-weighted design itself. The thresholds
  were set after stage G's numbers had been read.

Where it departs from this repository's conventions, so you can decide whether to adopt, rework or drop it:

- Self-contained. It does not import `tinygpt` or `scale`; `study.py` carries its own model and loop, modelled on
  `scale` (bfloat16 autocast, deterministic kernels, batch = f(data seed, step), deep-copied states).
- One folder with its own `results/`, not `studies/<name>/run.py` plus `results/<name>/`.
- Run outside the uv workspace: Windows, RTX 4060 Laptop, CPython 3.13, torch 2.14.1+cu126 (the workspace pins
  2.15.0). Formatted with black 26.10.1, the version the hook pins, in one pass after the last run.
- Corpus is the Python standard-library source of the local interpreter (12.0 MB; SHA-256 recorded in every
  result file), because enwik8 could not be downloaded. So B_s4 has E11's model shape but not E11's data.
- No write-up in `research/experiment-results` and no submodule bump. Nothing outside this folder is touched.

```sh
python study.py bench          # speed only
python smoke.py                # mechanics: exact replay, untouched checkpoint, swap arms
python run_all.py A            # then B, then C
python analyze.py              # tables and the verdict of each rule
python stage_d.py all          # then analyze_d.py; likewise stage_e/f/g/h.py: run, then analyze
```
