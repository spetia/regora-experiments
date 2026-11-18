# E1: does early ranking of configurations predict final ranking?

**Label: MEASURED, small. Plan and decision rule committed before the run (`experiments/studies/e1_early_ranking/run.py` docstring). Device: Apple MPS.**

Setup: LR in {5e-4, 1e-3, 2e-3} x weight decay in {0, 0.1}, 3 seeds each (init and data seed vary), 600 steps, cosine to 10%, eval every 50. Ranking on seed-mean val loss. Raw: `experiments/results/e1/e1.json`, `e1_analysis.txt`.

| h (steps) | Kendall tau vs final, 6 configs | regret of picking best at h |
|---|---|---|
| 100 | +0.87 | 0.0060 |
| 200 | +1.00 | 0.0000 |
| 300 | +1.00 | 0.0000 |
| 400 | +1.00 | 0.0000 |

LR-only subset (wd=0): tau +1.00 and regret 0 at every h >= 100. Final seed-mean val loss: LR 5e-4: 2.864, 1e-3: 2.747, 2e-3: 2.567; weight decay 0.1 changes it by +0.003 to +0.006 (within seed SD 0.008-0.032).

## Pre-registered rule
"Successive halving supported if tau >= 0.7 at h=200 and regret <= 0.01." **Met** (tau 1.00, regret 0.000).

## Why this is a weak test, so read the pass cautiously
- The grid has no curve crossing. In this undertrained 600-step regime a larger LR is better at every step, so rankings are almost trivially stable. The test only shows rankings do not reorder when nothing should reorder.
- The weight-decay pairs are ties within seed noise, and they still ordered consistently with the final ranking only because of the LR ordering around them.
- Regimes where early ranking fails (a high LR that wins early then destabilises or plateaus; schedules that decay late) are absent. That is the case successive halving must survive.
- Tiny model, one corpus, one schedule shape, 3 seeds.

## Next
A harder E1b: extend LR up to unstable values (e.g. 4e-3, 8e-3), vary schedule length so crossing can occur, and add slow-start configurations (longer warmup). Only a failure to reorder there would be informative.
