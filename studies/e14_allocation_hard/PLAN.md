# E14 plan: adaptive allocation on a harder table

PRE-REGISTERED before the test run. Written after E13, which could not separate the allocators: with four
changes and one far ahead, every allocator picked the best change almost at once. This repeats the test where
it can fail: eight changes, several of them close, on seeds nobody has looked at.

## Claims

1. Given the same number of training runs, LUCB picks the best change more often than equal allocation.
2. LUCB is at least as good as successive halving.
3. LUCB's early stop saves runs without costing correct picks. In E13 this was only observed, not claimed.
4. The status quo, one run of each change next to one run of the control and keep whichever scores highest
   (what a tracker's compare view shows), often picks the wrong change, and the engine with more runs does
   clearly better.

## What is fixed in advance, and what the pilot was for

- Fine-tuning and scoring: E12's, unchanged (SmolLM2-360M, topic and NLI, 4000 training and 2000 test items,
  250 steps, deterministic kernels, the seed sets the data order only). Control LR: E12's (topic 3.2e-4, NLI
  6.4e-4). Nothing is tuned here.
- Changes (8): LR x0.5, x0.71, x1.41; label smoothing 0.05, 0.1; 200 steps, 300 steps; the reworded question.
- Data: seeds 201..216, control and every change, both tasks: 288 fine-tunes. An observation is the paired
  difference in test accuracy, change minus control on one seed. `run.py table` writes `results/e14/tables.json`.
- Truth per task: the change with the highest mean difference over all 16 seeds, if positive.
- Cost: one run per observation, plus one control run for each seed any change has reached.
- Allocators: `regora-eie` at the commit recorded with this plan. Since E13, LUCB's warm-up goes round the
  changes instead of filling one at a time (the flaw E13 reported). Nothing else changed.
  - uniform: the change with the fewest observations goes next;
  - halving: every surviving change gets r more seeds, then the worse half is cut; r = max(2, budget / 18);
  - LUCB: 3 seeds each, then the leader or its challenger, whichever has fewer; stops when the leader's 95%
    t interval clears the challenger's.
- Replay: `regora allocate results/e14/tables.json --budgets 24,36,48,60,72,84,96,108,120,132,144` with 1000
  seed orders from the fixed generator. 144 runs is every change on every seed.
- Score: per budget, the share of orders in which the pick is the truth; then the mean over the 11 budgets.
- Pilot: seed 12, every arm, both tasks, to check that the new arms run and score. Its outcome is recorded at
  the end of this file. Seed 12 is not used below.

## Decision rules

- H1. LUCB's mean score is at least 0.05 above uniform's on both tasks.
- H2. LUCB's mean score is at least halving's on both tasks.
- H3. At budget 144, on both tasks, LUCB spends at most 75% of the runs uniform spends (mean over orders) and
  picks the truth in at least 95% of orders.
- H4. The status quo is the replay at a budget of 9 runs: one seed for the control and for each change, pick
  the highest. That is the uniform allocator's score from `regora allocate ... --budgets 9`. On both tasks,
  LUCB's score at 72 runs is at least 0.20 above it.
- Reported only, for the status quo (`run.py quo`, over the 16 seeds, no sampling): how often it adopts a change
  whose 16-seed mean difference is zero or negative (a false win), and how often it keeps the control.
- Reported only: the score at every budget; runs spent; each change's mean difference with its 95% t interval
  over 16 seeds; the gap between the best and second-best change.

## Known limits

- Four of the eight changes were run in E12 on other seeds, so which of them tend to help is roughly known
  (LR x0.5 was far ahead on NLI). The new seeds, the four new changes and their ranking are not.
- If one change is again far ahead on a task, that task cannot separate the allocators, and H1 fails there by
  construction. That is a property of the table, and it is reported, not repaired.
- Two tasks, one model, 16 seeds. The 1000 orders reshuffle the same 16 seeds, so the scores carry no interval.
- Truth is the best mean on 16 seeds; if the top two are within noise of each other, "correct" is partly luck.
- LUCB's stop uses ordinary t intervals, not anytime-valid ones. H3 measures its error rate here; it does not
  bound it elsewhere.
- A replay cannot show how the engine behaves when it proposes changes nobody has run.
- The status quo here is a decision rule, not a tool: no tracker was used, and a person reading curves may
  decide differently from "highest final accuracy". The tasks are public benchmarks, not a customer's work.
  H4 compares 9 runs with 72, so it shows what the extra runs buy, not that the engine is better per run.

## Pilot outcome (`results/e14/pilot_*.json`, seed 12, one run per arm and task)

Allocators under test: `regora-eie` commit 55982c9. No rule or threshold was changed after the pilot.

- Every arm ran and scored on both tasks. A fine-tune takes about 175 s (topic) and 230 s (NLI) with both tasks
  sharing one A10G, so the 288 fine-tunes take roughly 8 to 9 hours.
- Accuracy on seed 12, topic: control 0.9270; changes from 0.9135 (LR x0.5) to 0.9260 (smoothing 0.1). No change
  beat the control on this seed.
- Accuracy on seed 12, NLI: control 0.7980; LR x0.5 0.8040 and LR x0.71 0.8025 are close together at the top;
  LR x1.41 0.7445, 200 steps 0.7610 and 300 steps 0.7660 are well below.
- One seed says nothing about H1 to H4.
