# E6 — Regora selectors vs Optuna at equal compute

Date: 2026-11-11. Plan and rule committed in `experiments/studies/e6_optuna_baseline/run.py` before the E3 data existed.
Raw: `experiments/results/e6/e6_analysis.txt`. Replay on the E3 table; no new training runs. Optuna 5.0.0.

## Result

| Method | Compute | Held-out regret | P(picks best) |
|---|---|---|---|
| Regora schedule-aware | 0.583 | 0.0062 | 0.88 |
| Regora fixed halving | 0.444 | 0.0028 | 0.95 |
| Optuna TPE + Hyperband | 0.578 | 0.0169 | 0.56 |
| Optuna random + median pruning | 0.578 | 0.0117 | 0.68 |
| Optuna grid + successive halving | 0.527 | 0.0074 | 0.83 |

Optuna's budget was set to the schedule-aware selector's compute (5,247 steps), 3 sampler seeds per subset.

## Verdict

**No advantage by the pre-registered rule.** The rule was about the schedule-aware selector, and Optuna's grid variant is
within 0.005 of it.

## What it shows

- Regora's fixed halving had lower regret than all three Optuna variants while using less compute. Against TPE and random
  search the gap (0.009–0.014) is larger than the margin. Against Optuna's own successive halving it is 0.0046, inside the margin.
- That comparison was reported but was not the pre-registered claim, so it is an observation, not a confirmed result.
- The likely reason is structural, not clever: Regora's ladder compares all configs at the same step on the same paired
  seeds, while sequential Optuna prunes each trial against whatever happened to run before it, and TPE and random search
  re-sample configs on a 9-point grid.

## Limits

A 9-point grid run sequentially is close to Optuna's worst case; it is built for large or continuous spaces and parallel
workers. Fixed halving is an established method, not a Regora invention. Subsets overlap. One model, one dataset.

## Consequence

The spec's success criterion (a measurable advantage over a meaningful baseline) is **not met** for the selector that was
supposed to carry it. A fair follow-up would pre-register fixed halving with paired seeds against Optuna on a larger search space.
