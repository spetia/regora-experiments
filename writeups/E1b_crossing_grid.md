# E1b — early ranking on a grid with crossings

Date: 2026-11-11. Plan pre-registered in `experiments` (commit "pre-register E1b") before the run.
Code: `experiments/studies/e1b_crossing_grid/run.py`, `experiments/studies/e1b_crossing_grid/analyze.py`. Raw: `experiments/results/e1b/e1b.json`, `e1b.log`, `e1b_analysis.txt`.

## Setup

TinyGPT (875k params), MPS, 1000 steps, cosine to 10%, eval every 50 steps.
Grid: LR {1e-3, 2e-3, 4e-3, 8e-3} × warmup {50, 300}, 3 seeds each (24 runs, ~65 s each). No non-finite runs.

## Pre-registered rule

Successive halving (F4) is supported if **both**:
1. Kendall tau ≥ 0.7 between seed-mean ranking at h=350 and the final ranking.
2. Mean regret of the per-seed halving simulation (8→4@250→2@500→1@750, 46.9% of full compute) ≤ 0.01.

If no pair of configs crosses, the test is uninformative.

## Result

| Config (lr, warmup) | Final val loss (seed mean) | Seed SD |
|---|---|---|
| 1e-3, 50 | 2.5528 | 0.027 |
| 1e-3, 300 | 2.6300 | 0.026 |
| 2e-3, 50 | 2.3669 | 0.032 |
| 2e-3, 300 | 2.4310 | 0.018 |
| 4e-3, 50 | 2.2771 | 0.010 |
| 4e-3, 300 | 2.2963 | 0.030 |
| 8e-3, 50 | 2.2676 | 0.008 |
| 8e-3, 300 | **2.2563** | 0.024 |

Crossing pairs: 9 of 28, so the test is informative (E1 had none).

| Horizon | Tau vs final | Regret of early pick | Early pick |
|---|---|---|---|
| 250 | 0.71 | 0.0113 | 8e-3, 50 |
| 350 | 0.71 | 0.0208 | 4e-3, 50 |
| 500 | 0.86 | 0.0208 | 4e-3, 50 |
| 750 | 1.00 | 0.0000 | 8e-3, 300 |

Per-seed halving: regret 0.0000, 0.0113, 0.0208. Mean **0.0107**.

## Verdict

**Rule not met.** Condition 1 passes (0.71 ≥ 0.7, barely). Condition 2 fails (0.0107 > 0.01, barely). By the pre-registered rule, halving is not supported on this grid. I am not moving the threshold after the fact.

## What it does and does not show

- Halving never picked a bad config. It always landed in the top three, and the top three differ by 0.021, which is about one seed SD (0.008–0.030). With 3 seeds, the "true best" itself is not resolved.
- The failure mode is specific: the eventual winner (8e-3 with long warmup) starts slow and is under-ranked until after step 500. Halving at 250 and 500 dropped it in 2 of 3 seeds. This is the slow-starter problem the literature review flagged for F4, and E1b reproduced it.
- Regret here is measured against a seed-mean best that is itself noisy. The 0.01 threshold was set tighter than the noise in the reference. That is a flaw in my pre-registration, not a reason to reinterpret the result.
- One model, one dataset, 3 seeds, 8 configs. No scale conclusion.

## Consequence for the design

- F4 as a plain rank-and-cut is cheap (47% compute) and safe against bad picks here, but it is biased against long-warmup configs. It should not be the only selector.
- Two fixes are worth testing next (E2): compare configs at matched schedule phase rather than matched step, and replace hard cuts with a paired sequential test (F3) that only drops a config when the difference exceeds the seed noise.
- Any future threshold must be stated in units of the measured seed SD, with enough seeds to resolve the reference best.
