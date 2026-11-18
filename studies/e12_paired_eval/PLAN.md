# E12 plan: is a difference between two fine-tunes real?

PRE-REGISTERED before the test run. Written after E11. Regora's direction is a verdict on every model change. Until
now every comparison was a validation loss on runs trained from scratch. This asks the same question where most
teams meet it: two fine-tunes of a pretrained model, scored on a set of evaluation items.

## Claims

1. Scoring both models on the same items needs at least 2x fewer items than scoring them on separate samples.
2. The interval an evaluation tool prints from those items is too narrow, because it treats each model as fixed
   and ignores that retraining with another seed moves the score.
3. Fine-tuning both arms on the same seed (same data order) needs at least 2x fewer seeds than separate seeds.
   E11 found 3.7x to 12x for a small change over 300 steps from scratch; a whole fine-tune is untested.

## What is fixed in advance, and what the pilot was for

- Model: SmolLM2-360M (causal LM, all weights trained). The answer is read as the next token among the label words.
- Tasks: topic (AG News, 4 labels) and NLI (MNLI matched, 3 labels). 4000 training items, 2000 test items and
  1000 dev items per task, drawn once with fixed seeds; dev and test do not overlap.
- Fine-tune: 250 steps of batch 32, AdamW, 25 warmup steps then linear decay to zero, gradient clipping at 1.0,
  bfloat16 autocast, deterministic kernels (E11: default kernels cost pairing on a transformer).
- The seed sets the order of the training data and nothing else. The model has no dropout and no new weights.
- Control LR per task: highest mean dev accuracy over a 2x grid (5e-6 to 2.56e-3), seeds 3 and 4. Topic: 3.2e-4.
  NLI: 6.4e-4. Test items are not read during tuning.
- Arms: control; half LR; a reworded question in the prompt, used for training and scoring; label smoothing 0.1
  (control 0); 300 steps instead of 250. No weight decay anywhere.
- Pilot: seed 11, every arm, both tasks, to check the code runs, a seed replays exactly and the tuned LR is not on
  the edge of its grid. Its outcome is recorded at the end of this file. Seed 11 is not used below.

## Data

20 fresh seeds 101..120 per task and arm: 200 fine-tunes. Each records, for all 2000 test items, whether the
answer was right and the loss.

## Decision rules (`analyze.py`)

- H1 item pairing. For each task and each non-control arm, over the 20 same-seed pairs (arm, control):
  (variance of arm's item scores + variance of control's) / (variance of the per-item difference), pooled over
  pairs, is at least 2.
- H2 overconfident interval. For each task, over all 190 pairs of control runs on different seeds: the paired 95%
  interval over items (normal, 1.96 standard errors) excludes zero in at least 10% of pairs. 5% is what the
  interval promises if training noise did not matter. The 190 pairs share runs, so the rate is descriptive and
  has no interval of its own.
- H3 seed pairing. For each task and each non-control arm, on accuracy over the 20 seeds: (variance of arm +
  variance of control) / (variance of arm minus control on the same seed) is at least 2.
- Reported only: SD of control accuracy across seeds next to the standard error from 2000 items; item pairing
  between control runs on different seeds; each arm's mean accuracy difference with its 95% t interval over 20
  seeds; how many single-seed comparisons the item interval calls significant, and how many match the sign of the
  20-seed mean.

## Known limits

One model of 360M parameters, two classification tasks, accuracy only. Fine-tunes are short (2 epochs of 4000
items). The seed moves data order only; a setup with dropout or a new head has more training noise than this one.
Control LR comes from two seeds. H2's pairs are not independent. All arms are small changes; nothing here says how
the three numbers behave for a large one.

## Pilot outcome (`results/e12/pilot_*.json`, seed 11, one run per arm and task)

Changes made after the pilot, before this plan was committed:

- LR grid. The first grid (5e-6 to 8e-5) put both tasks on its upper edge, and the second (to 6.4e-4) left NLI
  there, so it was extended twice. Earlier scores were kept, since a run replays exactly. The files of the two
  earlier rounds are in `results/e12/first_grid/` and `results/e12/second_grid/`.
- The weight-decay arm (0.1) was dropped. In the first pilot it gave the same answer and the same loss as the
  control on all 2000 items of both tasks, so it tested nothing. The reworded-question arm replaces it.

No threshold or rule was changed. What the pilot shows:

- A seed replays exactly on both tasks, and another seed gives different losses. A fine-tune takes about 80 s
  (topic) and 100 s (NLI) alone on an A10G.
- Accuracy at the control LR: topic 0.9135, NLI 0.7845. Arms differ from the control on 36 to 60 of 2000 items on
  topic and 260 to 286 on NLI.
- NLI's control LR is one step below the LR at which dev accuracy collapses (0.8155 at 6.4e-4, 0.5725 at
  1.28e-3), and on seed 11 half LR scored higher than the control (0.8115 against 0.7845). Some NLI seeds may
  train badly; they are kept and reported, not removed.
- One seed cannot say anything about H1 to H3.
