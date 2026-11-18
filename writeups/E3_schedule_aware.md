# E3 — schedule-aware cut rule vs fixed halving

Date: 2026-11-11. Plan and rules committed in `experiments/studies/e3_schedule_aware/run.py` before the run.
Raw: `experiments/results/e3/e3.json`, `e3_analysis.txt`. All values measured.

## Setup

New grid: learning rate {2e-3, 4e-3, 8e-3} × warmup {50, 200, 400}, 9 configs, 1000 steps, 8 fresh seeds (72 runs), TinyGPT on MPS.
Selectors are replayed on all 56 three-seed subsets and scored on the 5 held-out seeds.

## Result

| Selector | Compute | Held-out regret | P(picks best) |
|---|---|---|---|
| Fixed, cuts at 250 / 500 / 750 | 0.444 | 0.0028 | 0.95 |
| Fixed, cuts at 250 / 500 / 1000 | 0.472 | 0.0028 | 0.95 |
| Fixed, cuts at 250 / 750 / 1000 | 0.528 | 0.0000 | 1.00 |
| **Schedule-aware (guard 200)** | 0.583 | **0.0062** | 0.88 |
| Fixed, cuts at 500 / 750 / 1000 | 0.667 | 0.0014 | 0.96 |
| Full training | 1.000 | 0.0034 | 0.91 |

Random pick: regret 0.0870. Seed SD of the final loss: 0.017–0.039. 15 of 36 config pairs cross.
Best config: LR 8e-3 with warmup 200 (2.2350), then LR 8e-3 with warmup 400 (2.2569).

## Verdict

- **H1 (later cuts beat early cuts): not confirmed.** The difference was 0.0028, below the 0.005 margin. E2's post hoc
  finding did not replicate as a meaningful effect.
- **H2 (schedule-aware beats fixed halving): rejected. The rule is WORSE.** It used more compute than three fixed
  schedules and had higher regret than the best of them by more than the margin.

## What it shows

1. **Plain successive halving was enough here.** With three paired seeds it found the best config 95% of the time at 44% of
   full compute, even though 15 pairs of curves cross.
2. **E2's slow-starter problem was specific to that grid.** There the winner had the longest warmup. Here the winner has a
   middle warmup and is already competitive at step 250.
3. **Protecting slow starters has a cost.** The schedule-aware rule kept long-warmup configs alive, spent 14 points more
   compute, and still picked worse, partly because it commits at step 750.
4. **Three seeds of full training are not better than halving.** Full training had regret 0.0034, about the same as halving.
   At this noise level the limit is the number of seeds, not the number of steps.

## Limits

Subsets overlap. One model, one dataset. Regret differences below about 0.005 are within the uncertainty of the held-out mean.

## Consequence

- Drop the schedule-aware rule as a default. It stays in the code as a selector for comparison, not as a recommendation.
- Two experiments (E2, E3) now show no selection rule beating successive halving on this model. The evidence does not
  support building the product around a smarter selector.
