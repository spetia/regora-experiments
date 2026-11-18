# E0: seed noise floor and branch (CRN) correlation

**Label: MEASURED, small and exploratory. One checkpoint, 12 data seeds, 6 from-scratch seeds, one tiny model.**

## Setup
- Char-level GPT, 875,264 params (4 layers, 128 dim, 4 heads, ctx 128, batch 32), trained on 2.7 MB of Python stdlib source, 0.3 MB held-out. CPU, torch 2.15.0, AdamW, constant LR 3e-3 after 50 warmup steps, grad clip 1.0.
- Val loss on a fixed set of 20 batches (same for all runs, so eval-set noise cancels in comparisons; it is **not** an estimate of dataset-sampling variance).
- From scratch: 6 runs, init seed and data seed both vary, 400 steps.
- Branches: one shared checkpoint (init seed 0, 200 steps), 12 data seeds x LR multiplier in {1, 0.5, 2}, 200 steps each. Batches are a pure function of (data_seed, step), so arms with the same data seed share batches (CRN).
- Code: `experiments/tinygpt/`, `experiments/studies/e0_noise_floor/run.py`, `experiments/studies/e0_noise_floor/analyze.py`. Raw: `experiments/results/e0/e0.json`, `e0_analysis.txt`.
- Determinism check (repeat of one branch): **bitwise identical**.

## Invalid first run (kept, not used)
First run reported `determinism_repeat_identical: False`. Cause: `load_state_dict` aliased the checkpoint's Adam state, so branches mutated the shared checkpoint. Fixed with `copy.deepcopy`; the whole experiment was rerun. Files: `experiments/results/e0/e0_run1_INVALID.*`.

## Results (rerun)
From-scratch SD of val loss: 0.07 (step 50), 0.06, 0.04, 0.02, 0.05, 0.02, 0.06, 0.04 (step 400). Non-monotone in step: with n=6 the SD estimates are themselves noisy (roughly +/-30%).

Branch control SD across data seeds: 0.035, 0.036, 0.057, 0.028 at h = 50, 100, 150, 200.

Paired (CRN) vs unpaired replicates for 80% power (normal approx, `n = 7.85 (SD/mean)^2`, per arm), final-horizon endpoint:

| h | 0.5x LR: mean effect | SD paired | est. rho | n paired | n unpaired |
|---|---|---|---|---|---|
| 50 | +0.010 | 0.025 | 0.65 | 47 | 135 |
| 100 | +0.011 | 0.023 | 0.72 | 34 | 123 |
| 150 | +0.041 | 0.032 | 0.76 | 5 | 20 |
| 200 | +0.026 | 0.024 | 0.61 | 6 | 17 |

For 2x LR the mean effect at h=200 is -0.011 (sign flips across horizons: +0.014, +0.018, +0.023, -0.011); n paired 23-61.

**Post-hoc (not pre-registered) smoothed endpoint** = mean of val loss at h = 100, 150, 200:
- 0.5x LR: effect +0.026, SD paired 0.018, effect/SD 1.43, n paired about 4, unpaired about 16.
- 2x LR: effect +0.010, SD paired 0.023, effect/SD 0.43, n paired about 42, unpaired about 171.

**Rank agreement** of the ordering {0.5x, 1x, 2x} at horizon h vs at h=200, fraction of the 12 data seeds: 0.08 (h=50), 0.33 (h=100), 0.08 (h=150). Chance for a random 3-way ordering is 0.17.

## What this supports and what it does not
- Supported (this setup only): pairing on shared batches reduces the replicates needed by about 2-5x (estimated rho about 0.5-0.8). The effect-to-noise ratio of a 2x LR change is about 0.4-1.4 depending on the arm and endpoint, so affordable detection needs roughly 4-60 paired replicates per comparison. Per-seed ordering of arms at short horizons barely predicts the final ordering.
- Not supported: anything about larger models, other interventions, other data, or LR schedules. LR 3e-3 constant is likely at the edge of stability for this model, and the non-monotone SD(h) suggests spiky val loss, which inflates noise. One checkpoint only, so checkpoint-to-checkpoint variation is unmeasured. With n=12 the effect/SD ratios have wide uncertainty (not computed here).
- Pre-registered decision rule (docs/TECHNICAL_DECISIONS.md E0): "if the effect-to-noise ratio of typical interventions is below ~1 at affordable n, every method here is power-limited." This data is on both sides of that line, so **the rule does not trigger or clear**. Next step is a replication at a more standard LR schedule and with more checkpoints before drawing the conclusion.
