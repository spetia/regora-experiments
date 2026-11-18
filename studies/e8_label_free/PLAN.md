# E8 plan: diagnosis without labelled examples

PRE-REGISTERED before the test run. Written after E7c, where likelihoods fitted on one model failed on another (32 of
50, and confidently wrong).

## Claim

A fixed rule that reads probe effects relative to the run's own recent progress diagnoses an injected cause on models
it has never seen, with no fitting at all.

## How the rule was built (disclosure)

- Built on the E7, E7b and E7c data, so those cannot test it. I tried a 3 x 3 x 3 grid of the three thresholds there
  (best about 86%) and saw that single probe readings were too noisy to tell "capacity" from "LR too low".
- That led to one protocol change that the old data cannot check: each LR probe is replicated on 3 data orders and
  must be statistically resolved. The noise rule was set a priori.
- Pilot: the E7 model, seeds 5..8, five conditions, 20 cases. The rule scored 20 of 20 and nothing was changed after.
- E7c's "healthy" label was wrong for its model: the untuned LR really was too high there. So here each model's
  healthy LR is tuned first, and faults are defined relative to it.

## Data

- Models (layers, width): tiny (2, 96), mid (6, 256), deep (8, 192). Tiny and deep are new. Mid has E7c's shape.
- Healthy LR per model: lowest mean validation loss at step 400 over {1e-3, 2e-3, 4e-3, 8e-3}, seeds 3 and 4.
- Conditions: healthy; lr_high (8x); lr_low (1/12x); eval_noise (metric read on one fresh batch); capacity (width / 4).
  Reported but not decisive: mild_high (3x), mild_low (1/4x).
- 10 fresh seeds 71..80 per model and condition: 150 decisive cases, 60 mild. CUDA (A10G).
- Per case: train 300 steps; read the metric 4 more times; then control, half-LR and double-LR branches of 100 steps
  on 3 data orders; then a 2x wider model trained 400 steps from scratch.

## The rule (`analyze.py`, `diagnose`)

`scale` = max(|metric change over steps 200..300|, 1% of the metric). Stages run in order and stop at the first hit.

1. eval_noise if 2 x SD of the 4 extra readings > 0.25 x scale.
2. `lean` = (half-LR effect - double-LR effect), mean over 3 replicates. It must be resolved (|mean| > 2.92 standard
   errors). lr_low if mean / scale > 0.1; lr_high if mean / scale < -1.0.
3. capacity if the wider model's effect / scale < -0.5.
4. Otherwise healthy.

Cost in baseline steps: stage 1 about 27; stage 2 adds 900; stage 3 adds 1600.

## Decision rules

- H1: the rule is right on at least 80% of the 150 decisive cases, and at least 70% on each model.
- H2: it beats the E7-fitted likelihood model (probes only, first replicate) on the same cases by at least 10 points.
- Mild faults are reported only. A miss there is expected to read as "healthy".

## Known limits

Single injected causes, one architecture family, one dataset, at most a few million parameters. The capacity
condition runs at the full-width model's LR, which may not be the narrow model's best. Thresholds were chosen on
this model family; a different family could need different ones, and that would count against the claim.
