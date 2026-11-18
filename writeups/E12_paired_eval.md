# E12 — is a difference between two fine-tunes real?

Date: 2026-11-14. Plan pre-registered in `experiments` (`studies/e12_paired_eval/PLAN.md`, commit "Pre-register
E12"). Code: `experiments/studies/e12_paired_eval/`. Raw: `experiments/results/e12/` (`e12_topic.json`,
`e12_nli.json`, logs, tuning, pilots, `e12_analysis.txt`).

## Setup

SmolLM2-360M, all weights fine-tuned, on topic (AG News, 4 labels) and NLI (MNLI matched, 3 labels). The answer
is read as the next token among the label words. 4000 training items, 2000 test items per task. 250 steps of
batch 32, AdamW, warmup then linear decay, bfloat16, deterministic kernels, one A10G. The seed sets the order of
the training data and nothing else. Control LR from a 2x grid on seeds 3 and 4 (topic 3.2e-4, NLI 6.4e-4).

Per task, seeds 101..120 on five arms (control, half LR, reworded question, label smoothing 0.1, 300 steps): 200
fine-tunes. Each records right or wrong and the loss for all 2000 test items.

## Result

| Task | Control accuracy | SD across seeds | SE from 2000 items | Seed-only pairs the item interval calls different |
|---|---|---|---|---|
| Topic | 0.9183 | 0.0035 | 0.0061 | 20 of 190 (11%) |
| NLI | 0.7833 | 0.0083 | 0.0092 | 36 of 190 (19%) |

"Items saved" and "Seeds saved" are variance ratios: (variance of arm + variance of control) / variance of the
paired difference, over items (H1) or over per-seed accuracy (H3). At equal interval width, that is the factor by
which the sample shrinks. It is not a power calculation for a chosen effect size.

Per arm against control, 20 same-seed pairs:

| Task | Arm | Items saved | Seeds saved | Mean diff | 95% interval (20 seeds) | Single-seed item interval significant | Sign matches mean |
|---|---|---|---|---|---|---|---|
| Topic | half LR | 5.3x | 1.2x | -0.0047 | [-0.0064, -0.0029] | 5 of 20 | 18 |
| Topic | reworded question | 6.8x | 1.1x | -0.0018 | [-0.0041, +0.0006] | 4 of 20 | 15 |
| Topic | label smoothing | 6.3x | 0.9x | +0.0027 | [+0.0007, +0.0047] | 4 of 20 | 15 |
| Topic | 300 steps | 8.6x | 1.6x | +0.0004 | [-0.0014, +0.0023] | 4 of 20 | 11 |
| NLI | half LR | 2.5x | 0.8x | +0.0219 | [+0.0164, +0.0273] | 13 of 20 | 20 |
| NLI | reworded question | 2.4x | 0.9x | -0.0027 | [-0.0109, +0.0055] | 4 of 20 | 10 |
| NLI | label smoothing | 2.5x | 1.4x | +0.0017 | [-0.0043, +0.0078] | 5 of 20 | 10 |
| NLI | 300 steps | 2.6x | 0.9x | -0.0083 | [-0.0336, +0.0171] | 5 of 20 | 8 |

- H1, item pairing saves at least 2x items for every arm on every task: SUPPORTED (2.4x to 8.6x).
- H2, the item interval calls two seed-only twins different in at least 10% of pairs on every task: SUPPORTED
  as a descriptive threshold (11% and 19%; the interval promises 5%). The 190 pairs are all pairs of 20 runs, so
  each run is in 19 pairs, the pairs are not independent, and the rates have no interval of their own.
- H3, sharing the training seed saves at least 2x seeds for every arm on every task: NOT SUPPORTED (0.8x to
  1.6x; no arm reached 2x).

Reported only:

- Item pairing between control runs on different seeds saves 4.7x (topic) and 2.3x (NLI).
- One NLI run trained badly: 300 steps on seed 120 scored 0.5575 (arm mean 0.7750, SD 0.0517). It is kept, as the
  plan says. It dominates that arm's interval and its seed-saving ratio. No topic run and no other NLI run fell
  more than 0.05 below its arm's mean.
- On NLI, half LR beat the control on all 20 seeds by 2.2 points. The control LR, chosen on two seeds, sits one
  grid step below collapse and is too high.

## What it shows

1. Same-item scoring is worth having. It cuts the items needed by 2.4x to 8.6x, more on the easier task where
   two models agree on most items.
2. The interval computed from one evaluation run is too narrow for the question "does this recipe differ?". It is
   a valid statement about the narrower question "do these two checkpoints differ on this test set", which is
   what an evaluation tool answers. The null here is control against control: same configuration, only the training
   data order differs. The interval is the paired per-item difference with 1.96 standard errors over items, and
   ignores seed variance. Seed noise in accuracy (SD 0.0035 and 0.0083) is
   the same order as the item standard error, and pairing items removes item noise but not seed noise. For the
   four arms whose 20-seed interval includes zero or sits near it, a single-seed comparison is still called
   significant in 4 or 5 of 20 seeds, and its sign matches the mean in as few as 8 of 20.
3. Sharing the seed does not help for a whole fine-tune. E11 found 3.7x to 12x for 300-step branches from a
   shared checkpoint; here both arms start from the same pretrained weights and share the data order, and the
   ratio is about 1. Reading, not tested: two arms that differ from step 0 for 250 steps diverge enough that the
   shared order no longer correlates their outcomes.

## Conclusion

For fine-tunes, a verdict on a change needs both models scored on the same items and more than one training seed
per arm. One run per arm with an item-level interval is wrong more often than it claims (11% to 19% false alarms
against a promised 5%). Seed sharing, which was the saving in E11, gives nothing here, so the cost of a verdict is
the number of fine-tunes, and the "more seeds" estimate in the harness is the part that matters.

Seed-by-treatment interaction is not modelled. Its visible trace is the low seed ratios and the single-seed signs
that disagree with the 20-seed mean (8 to 10 of 20 on three NLI arms).

Limits: one 360M model, two classification tasks, accuracy only, short fine-tunes, seed moves data order only (no
dropout, no new head, so training noise here is a lower bound for setups that have them). Control LR from two
seeds. H2's pairs are not independent. All arms are small changes.
