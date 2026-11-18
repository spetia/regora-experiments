# E8 — diagnosis without labelled examples

Date: 2026-11-12. Plan pre-registered in `experiments` (`studies/e8_label_free/PLAN.md`, commit "Pre-register E8").
Code: `experiments/studies/e8_label_free/`. Raw: `experiments/results/e8/` (`e8.json`, `e8.log`, `tune.json`,
`pilot.json`, `e8_analysis.txt`). Background: `E7_diagnosis.md`, `E7c_larger_model.md`.

## Setup

A fixed rule with no fitting. It reads each probe's effect relative to the run's own progress over its last 100
steps and stops at the first stage that fires: noisy evaluation (spread of 4 extra readings), learning rate (half
and double LR on 3 data orders, must be statistically resolved), capacity (2x wider model), otherwise healthy.

Three models (layers x width): tiny 2x96, mid 6x256, deep 8x192. Each model's healthy LR was tuned first over
{1e-3, 2e-3, 4e-3, 8e-3}; it came out at 8e-3, 2e-3 and 1e-3. Faults are relative to it: LR 8x, LR 1/12x, metric on
one fresh batch, width / 4. 10 fresh seeds each, 150 decisive cases, plus 60 mild LR faults (3x, 1/4x). A10G, CUDA.

## Result

| Model | Label-free rule | Likelihoods fitted on E7 (probes only) | Mean cost (steps) |
|---|---|---|---|
| tiny | 44 of 50 | 42 of 50 | 1414 |
| mid | 45 of 50 | 35 of 50 | 1478 |
| deep | 34 of 50 | 32 of 50 | 1222 |
| pooled | **123 of 150 (82%)** | 109 of 150 (73%) | |

| Rule | Result |
|---|---|
| H1: at least 80% pooled and at least 70% on every model | **Not supported**: 82% pooled, but deep is 68% |
| H2: beats the E7-fitted likelihoods by at least 10 points | **Not supported**: 9.3 points |

Both misses are narrow, and both count as misses.

Per condition, pooled over the three models (30 cases each): noisy evaluation 30, LR too low 28, LR too high 26,
capacity 20, healthy 19.

Mild faults (reported only): 26 of 60. A 3x LR reads as healthy in 19 of 30 cases; a 1/4x LR reads as capacity in
15 of 30.

## What it shows

- Noisy evaluation and LR direction are diagnosed on unseen models with no fitting: 84 of 90. These rest on paired,
  replicated comparisons, which is the part of Regora with the strongest evidence behind it.
- On the mid model, where fitted likelihoods failed in E7c, the rule scored 45 of 50.
- The healthy-versus-capacity split is what fails: 39 of 60. On the deep model 9 of 10 healthy runs read as
  capacity, and 6 of 10 capacity runs read as LR too low.
- The cost is not low: about 1200 to 1500 steps per diagnosis, four to five times the 300-step run being diagnosed.

## Why capacity fails (interpretation, not tested)

- "A wider model does better after 400 steps" is true of the deep model's healthy runs (median effect -0.13). The
  probe measures whether more width helps, which is not the same as whether the run is capacity-starved. The fixed
  threshold of 0.5 x progress does not separate the two across models.
- The deep model's tuned LR sits at the bottom of the grid, so its true optimum may be lower still.
- A narrow model run at the full-width model's LR can really have an LR that is too low for it. The plan named this
  as a limit.

## Consequence

- Supported enough to build on: noise and LR-direction diagnosis from paired, replicated probes, with no fitting.
- Not supported: a capacity diagnosis from one wider-model probe, and detection of mild (3x, 1/4x) LR faults at 100
  probe steps.
- Any new capacity rule or threshold drawn from this data is post hoc and needs fresh seeds and a fresh model.
