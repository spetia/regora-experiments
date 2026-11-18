# E16 plan: does a referee keep a tuning loop honest?

PRE-REGISTERED before the test run. Written after E14; E15 was stopped for it.

Agents that tune models keep a change when it scores better. E14 suggested what that costs: picking the best of
eight changes from one run each claimed +0.38 points on the topic task where the true gain was -0.03 (post hoc).
A real loop is worse, because each accepted change becomes the baseline for the next. This measures the gap
between what such a loop claims and what it delivers, with and without a referee in front of "accept".

## Claims

1. A loop that accepts any change that scores better on one run claims much more than it delivers.
2. The same loop behind a referee (a replicated paired comparison that must give "real") claims what it delivers.
3. The referee does not cost the loop its real gains.

## What is fixed in advance, and what the pilot is for

- Model, data, fine-tuning and scoring: E12's, unchanged (SmolLM2-360M, topic and NLI, 2000 test items,
  deterministic kernels, control LR from E12).
- Grid: LR multiple {0.5, 0.71, 1.0, 1.41} x label smoothing {0, 0.1} x steps {200, 250, 300}: 24
  configurations. Every configuration is fine-tuned on seeds 301..312 on both tasks: 576 fine-tunes. Seeds
  301..308 are for the loops; 309..312 are held out and used only to measure true gain.
- A loop: starts at (1.0, 0, 250). For 10 rounds it proposes a configuration that differs from the current one in
  exactly one setting, chosen uniformly at random, and decides whether to move there.
  - Greedy: each configuration it has run has one score (one loop seed, drawn at random the first time). It moves
    if the proposal's score is higher. Its claimed gain is the last accepted score minus the start's score.
  - Refereed: the proposal and the current configuration are compared on the same loop seeds, 3 to 8 of them,
    adding one at a time. It moves if the 95% t interval of the paired differences is above zero, and stops
    adding seeds as soon as the interval excludes zero either way. Its claimed gain is the sum of the mean
    differences of the moves it made.
  - True gain, for both: accuracy of the final configuration minus the start's, averaged over the held-out seeds.
- Replay: 2000 loops of each kind per task from a fixed generator; each kind sees the same random draws (the
  proposals themselves differ once the loops are at different configurations).
- Pilot: seed 13, three configurations (the start and two opposite corners), both tasks, to check the grid runs.
  Seed 13 is not used below.

## Decision rules (`replay.py`)

- H1. On at least one task, the greedy loop's mean true gain is at most half of its mean claimed gain.
- H2. On both tasks, the refereed loop's mean claimed gain is within 0.003 (0.3 points of accuracy) of its mean
  true gain.
- H3. On both tasks, the refereed loop's mean true gain is at least the greedy loop's minus 0.002.
- Reported only: fine-tunes each loop used; moves made per loop; the share of loops that end worse than they
  started; the same quantities with the referee at 5 seeds fixed.

## What is expected

H1 on the topic task, where E14 found no change reliably better than the control. On NLI the LR x0.5 direction
is large and both loops should find it, so H1 may not hold there. H3 is the one that could embarrass the
referee: with at most 8 seeds it may refuse small real gains.

## Known limits

- The "agent" is a random proposer over a 24-point grid. A real agent proposes better changes, and also
  fabricates and games evaluations; this tests only honest noise.
- The referee here is the statistical rule in Python, not the Regora harness, and it re-tests after every seed
  with ordinary t intervals (optional stopping), so its own false-accept rate is above 5%.
- True gain rests on 4 held-out seeds per configuration. Averaged over 2000 loops the means are stable, but a
  single loop's true gain is noisy; no per-loop claim is made.
- The 2000 loops reuse the same 8 loop seeds and are not independent. One model, two tasks.
- The grid and both tasks overlap with E12 and E14 on other seeds, so which directions help is roughly known.
- Cost is not equal: the refereed loop uses several times the fine-tunes. H3 compares outcomes, not efficiency.

## Pilot outcome (`results/e16/pilot_*.json`, seed 13, three configurations per task)

`replay.py` was frozen and passed its self-check on synthetic tables before this was committed. It has not been
run on any E16 data. No rule or threshold was changed after the pilot.

- The grid runs on both tasks. A fine-tune takes about 175 s (topic) and 200 s (NLI) with both tasks sharing one
  A10G, so the 576 fine-tunes take roughly 16 hours.
- Accuracy on seed 13, topic: start 0.9180; (0.5, 0, 200) 0.9105; (1.41, 0.1, 300) 0.9220.
- Accuracy on seed 13, NLI: start 0.7855; (0.5, 0, 200) 0.8005; (1.41, 0.1, 300) 0.6050. The high-LR corner
  trains badly on NLI, as LR x1.41 did in E14; such configurations stay in the grid.
- One seed says nothing about H1 to H3.
