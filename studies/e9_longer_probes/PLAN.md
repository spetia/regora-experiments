# E9 plan: longer LR probes, and capacity dropped from the diagnosis

PRE-REGISTERED before the test run. Written after E8, where the label-free rule missed mild LR faults (26 of 60) and
could not tell a healthy run from a width-starved one (39 of 60).

## Claims

1. Reading the LR probes after 300 steps instead of 100 detects 3x and 1/4x LR faults without raising false alarms.
2. With the capacity stage removed, the rule is reliable on the four conditions it still covers.
3. Acting on its LR advice lowers the loss later in the run.

## How the rule was built (disclosure)

- E8's data was re-read first. The wider-model probe there tracked the LR, not capacity: on E8's mid model a healthy
  run got worse when widened (+0.27), and the "capacity" cases ran at the full model's LR, which was too low for them.
- Pilot 1 (`results/e9/pilot.json`): tiny and base models, seeds 11..13, six conditions, on the old LR grid.
  Pilot 2 (`pilot2_*.json`): E8's mid and deep models, same seeds and conditions, LR grid extended to 1.6e-2.
  All four models and these seeds were seen before; none is used below.
- LR probes. At 100 steps healthy runs lean towards "halving helps" (a transient), which is why E8 needed a lopsided
  threshold. At 300 steps most of the lean is gone, so the threshold is now symmetric. At 0.3 two healthy pilot runs
  still read as "LR too high"; at 0.4 none did, so it is 0.4.
- Agreement test. E8's t-test on 3 replicates rejected one 8x fault whose effect was large but uneven. It is replaced
  by: all three replicates agree in direction. On the 60 pilot cases without the capacity condition this scored 53
  against the t-test's 49 at the same threshold (`results/e9/pilot_analysis.txt`). I compared three agreement tests
  at four thresholds there, so the pilot number is optimistic.
- Capacity. Three probes were tried in the pilots: a 2x wider model for 400 steps, the same at half the LR, and the
  wider model's lead at step 100. The last separated healthy from capacity on the two small models and overlapped on
  mid and deep. None is used. Capacity is out of scope for the rule; the step-100 lead is still collected and reported.
- The narrow model in the capacity condition now runs at its own tuned LR.
- The follow-through branches (900 steps) were added in pilot 2 because a 3x LR barely changed the base model's loss,
  so "mild fault" is not always a fault. Advice is scored by what it does to the loss.

## Data

- Models (layers, width), all new: squat (3, 160), broad (4, 320), tall (10, 128).
- Healthy LR per model and per width / 4 version: lowest mean validation loss at step 400 over
  {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}, seeds 3 and 4.
- Conditions: healthy; lr_high (8x); lr_low (1/12x); eval_noise (metric read on one fresh batch); mild_high (3x);
  mild_low (1/4x); capacity (width / 4 at its own LR; reported only).
- 10 fresh seeds 81..90 per model and condition: 210 cases. CUDA (A10G), one process per model.
- Per case: train 300 steps; read the metric 4 more times; control, half-LR and double-LR branches on 3 data orders,
  read every 100 steps to 300 (replicate 0 runs on to 900); a 2x wider model trained 100 steps from scratch.

## The rule (`analyze.py`, `diagnose`)

`scale` = max(|metric change over steps 200..300|, 1% of the metric). Stages run in order and stop at the first hit.

1. eval_noise if 2 x SD of the 4 extra readings > 0.25 x scale. (Unchanged from E8.)
2. `lean` = half-LR metric minus double-LR metric at 300 steps, per replicate. If all 3 agree in sign and
   |mean| / scale > 0.4: lr_low when positive, lr_high when negative.
3. Otherwise healthy, meaning no cause found.

Cost in baseline steps: stage 1 about 27; stage 2 adds 1800. The control branch is not needed for the diagnosis.

## Decision rules

- H1 mild faults: at least 70% of the 60 mild cases are right, and at most 10% of the 30 healthy cases get an LR
  diagnosis.
- H2 four conditions (healthy, lr_high, lr_low, eval_noise; 120 cases): at least 85% right pooled and 75% on each
  model.
- H3 advice: over all 210 cases, when the rule says lr_low or lr_high, the advised branch (double or halve, replicate
  0) has lower loss than control 900 steps after the checkpoint in at least 80% of them.
- Reported only: accuracy with 100- and 200-step probes; the t-test variant; what the rule says on capacity cases;
  the wider model's step-100 lead for healthy and capacity cases.

## Known limits

Single injected causes, one architecture family, one dataset, at most a few million parameters, runs of 300 steps.
The diagnosis costs six times the run so far. The 900-step follow-through uses one replicate, so H3 is noisy per
case. Healthy LRs come from a 2x grid, so "healthy" can itself be up to about 1.4x off.
