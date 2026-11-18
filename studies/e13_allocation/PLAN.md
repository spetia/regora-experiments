# E13 plan: does adaptive allocation find the best change in fewer runs?

PRE-REGISTERED before the replay. Written after E12. The pitch says Regora will answer "what should we run
next?". This is the first test of an engine for that question in its narrowest form: several candidate changes
compete against one control, and the engine decides which change gets the next seed.

## Claim

Given the same number of training runs, an allocator that concentrates seeds on the leading change and its
strongest challenger (LUCB on paired differences, `regora-eie::allocate::Lucb`) picks the best change more often
than giving every change the same seeds, and at least as often as successive halving.

## What is fixed in advance

- Data: the E12 results (`results/e12/e12_{topic,nli}.json`), unchanged. Per task, 4 changes (half LR, reworded
  question, label smoothing, 300 steps) and 20 seeds. An observation is a paired difference in test accuracy,
  change minus control on one seed. `make_table.py` writes them to `results/e13/tables.json`.
- Truth per task: the change with the highest mean difference over all 20 seeds, if positive.
- Cost: one training run per observation, plus one control run for each seed any change has reached.
- Allocators, as implemented in `regora-eie` at the commit recorded in the write-up:
  - uniform: the change with the fewest observations goes next;
  - halving: every surviving change gets r more seeds, then the worse half is cut; r = max(2, budget / 10);
  - LUCB: 3 seeds each, then the leader or its challenger, whichever has fewer; stops when the leader's 95%
    t interval clears the challenger's.
- Replay: `regora allocate results/e13/tables.json` with its defaults: budgets 12, 20, ..., 100 runs and 1000
  random seed orders from a fixed generator. Every allocator sees the same orders.
- Score: for each budget, the share of orders in which the pick is the truth; then the mean over the 12 budgets.

## Decision rules

- H1. LUCB's mean score is at least 0.05 above uniform's on both tasks.
- H2. LUCB's mean score is at least halving's on both tasks.
- Reported only: the score at every budget, the runs each allocator actually spent, and how often each stops
  early.

## Known limits

- The data was seen before this plan was written: the 20-seed means from E12 are known (half LR is clearly best
  on NLI; label smoothing leads on topic by a small margin). The allocators were written and unit-tested on
  synthetic tables only and have not been run on this data. The thresholds were set without running them.
- Two tasks, four changes each, one model. The 1000 orders reshuffle the same 20 seeds, so they are not
  independent samples and the scores carry no interval.
- LUCB's stopping rule uses ordinary t intervals, not anytime-valid ones, so its error rate is not controlled.
- E12 found that sharing seeds does not reduce variance for fine-tunes, so the pairing itself is not expected
  to help here; only the allocation is under test.
- A replay cannot show how the engine behaves when it proposes changes nobody has run.
