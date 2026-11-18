# E13 — does adaptive allocation find the best change in fewer runs?

Date: 2026-11-14. Plan pre-registered in `experiments` (`studies/e13_allocation/PLAN.md`, commit "Pre-register
E13"). Code: `regora-eie` `src/allocate.rs` (commit a6abe93), `regora-cli` `src/allocate.rs` (19e6de3). Raw:
`experiments/results/e13/` (`tables.json`, `e13_replay.txt`).

## Setup

A replay on the E12 results, no new training. Per task (topic, NLI): 4 candidate changes and 20 seeds; an
observation is the paired difference in test accuracy, change minus control on one seed. Three allocators decide
which change gets the next seed: uniform, successive halving, and LUCB (leader against strongest challenger,
with an early stop). Cost is one run per observation plus one control run per seed reached. Budgets 12 to 100
runs, 1000 random seed orders. Score: how often the pick is the change that is best on all 20 seeds.

## Result

Share of orders in which the best change was picked:

| Task | Allocator | 12 | 20 | 28 | 36 | 44 | 52 | 60 to 100 | Mean | Runs spent at budget 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Topic | uniform | 0.61 | 0.80 | 0.84 | 0.92 | 0.94 | 0.98 | 0.99 to 1.00 | 0.924 | 100 |
| Topic | halving | 0.68 | 0.81 | 0.83 | 0.92 | 0.96 | 0.99 | 1.00 | 0.933 | 80 |
| Topic | LUCB | 0.86 | 0.78 | 0.90 | 0.94 | 0.98 | 1.00 | 1.00 | 0.952 | 71 |
| NLI | all three | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 100, 80, 49 |

- H1, LUCB's mean score is at least 0.05 above uniform's on both tasks: NOT SUPPORTED (+0.028 on topic, 0.000
  on NLI).
- H2, LUCB's mean score is at least halving's on both tasks: SUPPORTED (0.952 against 0.933 on topic; a tie at
  1.000 on NLI).

Reported only: LUCB stopped early. With the full budget available it spent 71 runs on topic and 49 on NLI and
still picked the best change in every order; uniform spent 100.

## What it shows

1. The test could not separate the allocators. On NLI the best change is so far ahead that every allocator is
   right from 12 runs. On topic all three reach 0.98 by 52 runs. Only the low budgets on one task carry signal.
2. LUCB was never worse than halving and was slightly ahead of uniform on topic, short of the 0.05 that was
   set in advance.
3. The early stop is where a saving showed: the same picks for 29% to 51% fewer runs than the full budget. This
   was a reported-only item, the stop uses ordinary t intervals, and its error rate is not controlled.

## A flaw found in the result

LUCB's 0.86 at budget 12 on topic is partly an artefact. Its warm-up fills one change to 3 seeds before starting
the next, and 12 runs pay for only three changes, so the fourth (300 steps) was never observed. Uniform at the
same budget observes all four. The warm-up should go round the changes. It is not fixed here: a fix followed by
a re-run on the same data would not be a test.

## Conclusion

Not shown. Adaptive allocation did not pick the best change clearly more often than equal allocation on this
data, and the data was too easy to tell. The engine for "what should we run next?" stays unproven. What is worth
carrying forward is the early stop, as a hypothesis.

Next test needs: harder tables (more changes, smaller gaps between the best two), data not seen before the
plan, the warm-up fixed, and an anytime-valid stopping rule so the stop's error rate means something.

Limits: two tasks, four changes, one model; the orders reshuffle the same 20 seeds and carry no interval; the
E12 means were known when the plan was written; a replay says nothing about proposing changes nobody has run.
