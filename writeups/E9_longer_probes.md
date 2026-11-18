# E9 — longer LR probes, capacity dropped

Date: 2026-11-13. Plan pre-registered in `experiments` (`studies/e9_longer_probes/PLAN.md`, commit "Pre-register
E9"). Code: `experiments/studies/e9_longer_probes/`. Raw: `experiments/results/e9/` (`e9_squat.json`,
`e9_broad.json`, `e9_tall.json`, logs, `tune.json`, pilots, `e9_analysis.txt`). Background: `E8_label_free.md`.

## Setup

The label-free rule from E8 with two changes: the LR probes are read after 300 steps instead of 100 (all three
replicates must agree in direction, and the mean effect must exceed 0.4 x the run's recent progress), and the
capacity stage is removed. The rule now answers: noisy evaluation, LR too high, LR too low, or no cause found.

Three models not seen before (layers x width): squat 3x160, broad 4x320, tall 10x128. Healthy LR tuned per model
over {1e-3 .. 1.6e-2}: 4e-3, 2e-3, 2e-3. Seeds 81..90. Seven conditions per seed: healthy, LR 8x, LR 1/12x, noisy
evaluation, LR 3x, LR 1/4x, and width / 4 (out of scope, reported only). 210 cases, A10G, CUDA.

## Result

| Rule | Result |
|---|---|
| H1: mild LR faults at least 70% right, and at most 10% of healthy runs get an LR diagnosis | **Not supported**: 49 of 60 right (82%), but 12 of 30 healthy runs flagged (40%) |
| H2: four conditions at least 85% pooled and 75% per model | **Supported**: 108 of 120 (90%), worst model 78% |
| H3: the advised LR change lowers loss 900 steps later in at least 80% of advised cases | **Supported**: 108 of 128 (84%) |

Per condition (30 cases each): noisy evaluation 30, LR 8x 30, LR 1/12x 30, healthy 18. LR 1/4x 30, LR 3x 19.

Per model on the four conditions: squat 38 of 40, tall 39 of 40, broad 31 of 40. Mean cost 1370 steps.

## What it shows

- **Every real fault of 4x or more was caught, on three unseen models.** 120 of 120 for noisy evaluation, LR 8x,
  LR 1/12x and LR 1/4x.
- **The rule cannot be trusted when it says "LR too high".** All 12 errors on healthy runs are that diagnosis, 9 of
  them on one model (broad, 9 of 10 healthy runs). They are false: across the 12, halving the LR beat the control
  900 steps later in 5; across all 30 healthy runs it did so in 10, and the mean effect of halving was a loss.
- **Longer probes traded one error for another.** Read at 100, 200 and 300 steps, LR 1/4x is caught in 15, 25 and
  30 of 30 cases; LR 3x in 30, 21 and 19. A lower LR looks better for a few hundred steps on a healthy run and
  that advantage fades, so a mildly high LR and a healthy one are hard to tell apart at any single probe length.
- **H3 passes on the pooled count because most advice was given to runs with real faults.** On healthy runs the
  advice was wrong more often than right.
- Capacity (reported only): 23 of 30 width-starved runs read as "no cause found", which is the intended answer
  now that capacity is out of scope. The early wider-model reading still does not separate them from healthy runs
  (median lead -0.35 against -0.04, with overlapping ranges).

## Conclusion

"LR too low" and "evaluation too noisy" are diagnoses the rule gets right without labels on models it has not
seen (60 of 60 and 30 of 30, no false alarms of either kind). "LR too high" is not usable: 40% of healthy runs
receive it. H1 failed as registered and the rule is not changed here.

Post hoc, not tested: the rule compares half LR against double LR, so "doubling hurts" counts the same as
"halving helps". Requiring the advised branch itself to beat the control is the obvious repair. It needs its own
pre-registered test on new models, and the transient described above suggests it should be read later than 300
steps.

## Limits

One dataset; models under 5M parameters; single injected faults; 10 seeds per cell. Nine of the twelve false
alarms come from one model, so the false-alarm rate on other shapes is poorly pinned down (1 to 9 of 10).
