# E11 — do the supported parts hold on larger models and a second task?

Date: 2026-11-13. Plan pre-registered in `experiments` (`studies/e11_scale/PLAN.md`, commit "Pre-register E11").
Code: `experiments/scale/`, `experiments/studies/e11_scale/`. Raw: `experiments/results/e11/` (`e11_lm.json`,
`e11_vision.json`, `exact_lm.json`, `exact_vision.json`, logs, tuning, pilots, `e11_analysis.txt`).

## Setup

Language: byte-level GPT, 8 layers x 512, 25.6M parameters, enwik8, AdamW. Vision: ResNet-18 (CIFAR form), 11.2M
parameters, CIFAR-100, SGD with momentum. Both: 100 steps of warmup then constant LR, bfloat16 autocast, one L40S.
Healthy LR from a 2x grid on seeds 3 and 4 (language 1e-3, vision 1.25e-2).

Per task, seeds 101..110 on five conditions (healthy, LR 1/12x, LR 1/4x, LR 3x, noisy evaluation): 100 cases.
Each trains 1000 steps, then control, half-LR and double-LR branches on 3 data orders for 300 steps; the run's own
data order continues to 1500. Healthy cases add an identical second control and a weight-decay branch. The
diagnosis rule is E9's, imported unchanged. The healthy cases were then repeated with deterministic kernels.

## Result

Seeds saved by pairing at 300 steps (unpaired variance / paired variance), 10 healthy checkpoints per task:

| Task | Kernels | Half LR | Double LR | Weight decay | Non-determinism / data-order SD |
|---|---|---|---|---|---|
| Language | default | 1.32x | 1.25x | 3.70x | 0.57 |
| Language | deterministic | 1.93x | 1.95x | 3.93x | 0 |
| Vision | default | 2.59x | 1.99x | 12.04x | 0 |
| Vision | deterministic | 2.59x | 1.99x | 12.04x | 0 |

Correlation behind those ratios (control against arm, across data orders, pooled within the 10 checkpoints;
default kernels). Computed after the plan by `analyze.py`, reported only:

| Task | Half LR | Double LR | Weight decay |
|---|---|---|---|
| Language correlation | 0.03 | 0.46 | 0.80 |
| Vision correlation | 0.48 | 0.66 | 0.91 |

Pairing is not free for every intervention. For language half LR the pairs are almost uncorrelated. Per-checkpoint
savings run from 0.2x to 129.6x across the six rows, and the minimum is below 1x in all but vision weight decay.
Each checkpoint has 3 data orders, so single-checkpoint savings are noisy; the pooled correlation is the stable number.

Diagnosis, truth (rows) against the rule's reading, 10 cases per row:

| Task | Truth | Noisy eval | Healthy | LR too high | LR too low |
|---|---|---|---|---|---|
| Language | noisy evaluation | 10 | | | |
| Language | healthy | | | 10 | |
| Language | LR 1/12x | | 4 | | 6 |
| Language | LR 1/4x | | | 10 | |
| Language | LR 3x | | | 10 | |
| Vision | noisy evaluation | 9 | | 1 | |
| Vision | healthy | | | 10 | |
| Vision | LR 1/12x | | 10 | | |
| Vision | LR 1/4x | | | 10 | |
| Vision | LR 3x | | | 10 | |

| Rule | Result |
|---|---|
| H1: pairing saves at least 2x for every intervention on every task | **Not supported**: language 1.3x for both LR changes; vision double LR 1.99x |
| H2: non-determinism at most 0.1 of data-order SD on every task | **Not supported**: language 0.57 |
| H3: LR-too-low and noisy-evaluation cases right (85% pooled, 75% per task) | **Not supported**: 25 of 60 (42%); language 53%, vision 30% |
| H4: at most 10% of healthy runs told "LR too low" or "noisy evaluation" | Supported: 0 of 20 |
| H5: doubling the LR, when advised, is ahead 900 steps later in 80% | Supported: 6 of 6 |
| H6: half LR ahead at 300 steps on healthy runs, and mostly not at 1500 | **Not supported**: ahead 20 of 20, still ahead 20 of 20 |
| H7: with deterministic kernels both the H1 and H2 limits hold | **Not supported**: non-determinism is 0, but pairing is 1.93x, 1.95x (language) and 1.99x (vision) |

Reported only:

- Every healthy run (20 of 20) and every LR 3x run (20 of 20) read as "LR too high".
- E9's suggested repair was right on 11 of 40 low-LR cases and 19 of 20 noisy-evaluation cases, and still read all
  20 healthy runs as "LR too high".
- A branch's sign at 300 steps matched its sign at 1500 in 54 of 80 followed half-LR branches and 58 of 80
  double-LR branches. Half LR reaches 68 of 80 at 900 steps.
- Under default kernels the language model's two identical branches differ by 0.22 of the data-order SD after 100
  steps and by 0.6 to 1.3 of it from 500 steps on.
- Weight-decay effect at 300 steps: language +0.00096 ± 0.00032; vision -0.0051 ± 0.0179.
- Diagnosis cost about 1485 steps per case. No run produced a non-finite reading.

## What it shows

- **Noisy-evaluation detection carries over**: 19 of 20 across two tasks and two optimisers.
- **The learning-rate diagnosis does not.** It caught LR 1/12x on 6 of 10 language runs and no vision runs, and
  read LR 1/4x and healthy runs alike as "LR too high". Where it did say "LR too low" (6 cases, all language LR
  1/12x) doubling helped, and it raised no false "LR too low" on healthy runs. So "LR too low" is trustworthy
  when given and usually not given.
- **On healthy runs a halved LR was still ahead at 1500 steps on all 20.** At this scale the lead is not the
  short-lived effect seen in E9 and E10.
- **Pairing is worth 4x to 12x for a small intervention (weight decay) and about 2x or less for LR changes.** On the
  language model it resolved a weight-decay effect of 0.001 in loss from 10 checkpoints.
- **Default GPU kernels remove much of the pairing on the language model.** Deterministic kernels restore exact
  replay on both models and raise the language LR pairing from 1.3x to 1.9x. The pilot measured them at about the
  same speed; the full run did not time them separately.

## Conclusion

Four of seven registered claims failed, three of them (H2, H3 and H1 on the language model) as the plan predicted
from the pilot. H7 failed narrowly, on pairing just under 2x, not on determinism. What holds at 11M to 25M
parameters: noisy-evaluation detection, paired comparison of a small change, and exact replay with deterministic
kernels. What does not: diagnosing the learning rate from 300-step half and double branches.

Post hoc, not tested: the healthy LR was the best constant LR at step 1300, and halving a constant LR at step 1000
acts as LR decay, which helps for the rest of the run. If so, the half-against-double comparison measures "would
decay help now", which is nearly always yes, and not "is the LR wrong". The LR pairing ratio is low for a related
reason: an LR change moves the loss by different amounts on different data orders, so the paired difference keeps
much of the variance. Both readings need their own test.

Limits: two tasks, one model each, one GPU type, 2500 steps (early training), constant LR, single injected causes,
healthy LR from two seeds. H4, H5 and H6 rest on 20, 6 and 20 cases.
