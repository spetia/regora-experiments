# E10 plan: a run that corrects its own learning rate, against tuned schedules

PRE-REGISTERED before the test run. Written while E9 was still running; its interim results showed that paired
LR branches read the direction of an LR fault reliably. E10 asks whether acting on those branches all the way
through a run can replace an LR search.

## Claims

1. A run that branches on its LR every 300 steps and keeps the best branch ends no worse than a schedule picked by
   successive halving at the same total compute.
2. It ends better than that schedule.
3. Its result does not depend on the LR it starts from.

## How the controller was built (disclosure)

- Pilot (`results/e10/pilot_small.json`, `pilot_deep.json`, summary in `pilot_analysis.txt`): tiny and base models
  on seeds 11 and 12, deep on seed 11. All were seen in E7 to E9; none is used below.
- The pilot ran two versions: change the LR whenever a branch wins (margin 0.0), or only when it wins by more than
  0.2 x the segment's progress. Margin 0.0 had the lower test loss in 11 of 15 paired runs (mean -0.045), so the
  margin is 0.0. Nothing else was tuned.
- What the pilot predicts. From the default start, margin 0.0 beat both halving baselines in 2 of 5 cases and lost
  to both in 3. On the deep model it lost from every start, by 0.03 to 0.30. In every pilot run the controller
  halved the LR on most segments, down to 1/8 to 1/32 of its peak: halving nearly always wins the next 300 steps,
  whether or not it is good for the run. This short-horizon bias is the known trap of greedy LR control and the
  design does not correct it. I expect claim 2 to fail and claim 3 to fail on the deepest model; the test is run
  to measure this on unseen models, not because the pilot was encouraging.

## Data

- Models (layers, width), none seen before: slim (5, 96), stout (3, 256), lanky (8, 128). Seeds 91..100.
  Seed s fixes init (s) and data order (3000 + s). 2400 steps, 50-step warmup, A10G.
- Baselines, per model and seed: 6 constant-LR runs and 6 cosine runs (to 10% of peak) over the grid
  {5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}. Successive halving is replayed on them: rank by validation loss at step
  600 and keep 3, at 1200 keep 2, at 1800 keep 1. That costs 7200 steps.
- Controller, per model and seed, from each start in {5e-4, 2e-3 (default), 1.6e-2}: on every 300-step segment
  train three branches from the same state on the same data at 0.5x, 1x and 2x the current LR, and continue from
  the branch with the lowest validation loss. If all three diverge, retry the segment at LR / 4; the wasted steps
  count. Budget 3 x 2400 = 7200 steps, the same as halving.
- Selection (halving's ranking, the controller's branch choice) uses the fixed validation set `T.VAL_IDX`. Every
  score below is the loss on 40 separate validation batches (`TEST_IDX`) that nothing selected on.

## Decision rules (`analyze.py`)

Differences are controller minus baseline on test loss, paired by seed, 10 seeds per model, with a 95% t-interval.
The controller starts from the default LR unless stated.

- H1 no worse: on every model, against both halving baselines, the interval's upper end is below the baseline's
  seed-to-seed SD.
- H2 better: on at least 2 of 3 models, against both halving baselines, the interval's upper end is below 0.
- H3 start does not matter: on every model, for both extreme starts, (extreme start minus default start) has its
  interval's upper end below the SD of the default-start results.
- Reported, not decisive: the controller against the best single LR in hindsight (which costs 6 x 2400 steps),
  the LRs halving picked, the controller's LR path, and runs that ended short of 2400 steps.

## Known limits

- One dataset, models under 2M parameters, 2400 steps. The LR is the only thing controlled.
- Halving over six LRs is a modest baseline. A better tuner, or a longer run where the search cost is a smaller
  share, would be harder to beat.
- The controller's 3x cost is charged in full. Halving also yields a ranking of LRs that can be reused on the next
  run; the controller yields nothing reusable.
- The margin was chosen on five pilot cases.
