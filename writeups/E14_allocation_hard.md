# E14 — adaptive allocation on a harder table, and the status quo

Date: 2026-11-15. Plan pre-registered in `experiments` (`studies/e14_allocation_hard/PLAN.md`, commit
"Pre-register E14"). Allocators: `regora-eie` 55982c9. Raw: `experiments/results/e14/` (`e14_topic.json`,
`e14_nli.json`, logs, pilot, `tables.json`, `e14_replay.txt`).

## Setup

288 new fine-tunes of SmolLM2-360M (E12's procedure, unchanged) on topic and NLI: a control and 8 changes (LR
x0.5, x0.71, x1.41; label smoothing 0.05, 0.1; 200 and 300 steps; a reworded question) on seeds 201..216. An
observation is the paired difference in test accuracy, change minus control on one seed. Three allocators were
replayed on these tables at budgets of 24 to 144 training runs, over 1000 seed orders. Score: how often the pick
is the change that is best on all 16 seeds. The status quo is one run of each change and one control (9 runs),
keeping the highest.

## Result

Share of orders in which the best change was picked:

| Task | Allocator | 24 | 36 | 48 | 60 | 72 | 84 | 96 | 108 to 144 | Mean | Runs spent at budget 144 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Topic | uniform | 0.49 | 0.64 | 0.71 | 0.83 | 0.88 | 0.92 | 0.99 | 1.00 | 0.859 | 144 |
| Topic | halving | 0.55 | 0.74 | 0.75 | 0.88 | 0.98 | 0.96 | 1.00 | 1.00 | 0.896 | 112 |
| Topic | LUCB | 0.49 | 0.67 | 0.82 | 0.93 | 0.97 | 0.98 | 0.98 | 0.98 | 0.887 | 84 |
| NLI | uniform | 0.79 | 0.90 | 0.96 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 0.963 | 144 |
| NLI | halving | 0.78 | 0.85 | 0.79 | 0.92 | 0.98 | 0.97 | 1.00 | 1.00 | 0.935 | 112 |
| NLI | LUCB | 0.79 | 0.91 | 0.96 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 0.966 | 89 |

Status quo (9 runs): 0.32 on topic, 0.68 on NLI.

- H1, LUCB's mean score is at least 0.05 above uniform's on both tasks: NOT SUPPORTED (+0.028 on topic, +0.003
  on NLI). The same margins as E13.
- H2, LUCB's mean score is at least halving's on both tasks: NOT SUPPORTED (0.887 against 0.896 on topic; 0.966
  against 0.935 on NLI).
- H3, with the full budget LUCB spends at most 75% of the runs and is right in at least 95% of orders, on both
  tasks: SUPPORTED (58% of the runs and 0.98 on topic; 62% and 1.00 on NLI).
- H4, LUCB at 72 runs is at least 0.20 above the status quo on both tasks: SUPPORTED (0.97 against 0.32 on
  topic; 0.99 against 0.68 on NLI).

Reported only:

- Status quo, one decision per seed. Topic: it adopted a change in 12 of 16 seeds, and in 5 of those the change
  has a zero or negative mean over 16 seeds; it kept the control in 4. NLI: it always adopted LR x0.5 or x0.71,
  both of which help.
- Mean difference over 16 seeds. Topic: the best change is label smoothing 0.1 at +0.0013, 95% interval
  [-0.0004, +0.0030]; the second is +0.0003; four changes are reliably worse than the control. NLI: LR x0.5
  +0.030 [+0.018, +0.042], LR x0.71 +0.017 [-0.004, +0.038], label smoothing 0.1 +0.011 [+0.001, +0.022]; LR
  x1.41 -0.128.

## What it shows

1. Choosing where the next run goes did not beat spreading runs evenly. Two experiments now agree on that
   (E13, E14): LUCB's gain over uniform is about 0.03 on topic and nothing on NLI, and halving was ahead of it
   on topic. As a way to find the best change faster, the allocator is not supported.
2. Knowing when to stop did hold. LUCB reached the same answers with 58% to 62% of the runs. This was claimed in
   advance this time. The stop still uses ordinary t intervals, and on topic it was wrong in 2% of orders.
3. One run per change is a poor way to decide. On topic it picked the best change in a third of orders, and
   adopted a change that does not help in 5 of 16 seeds. On NLI, where the effects are ten times larger, it was
   right two times in three.
4. On topic nothing is established as better than the control: the best change's interval includes zero. So
   "picking the best" there partly means picking the largest of several near-zero effects, and the honest
   verdict for that task is "no change is shown to help". The allocators are not scored on saying that.

## Conclusion

The engine's value is not in choosing the next experiment; it is in the comparison and in stopping. Against
deciding from single runs, replicated paired comparison is far more often right (H4), which costs 8 times the
runs. Within replicated comparison, an early stop saves about 40% of them (H3). Smart allocation adds nothing
measurable on top (H1, H2). "What should we run next?" as a selection engine remains unsupported, now twice.

Limits: one model, two tasks, 16 seeds; the orders reshuffle the same seeds and carry no interval; four of the
eight changes were known from E12 on other seeds; truth on topic is within noise; H4 compares 9 runs with 72 and
no tracker or person was involved; the stop's error rate is measured here, not bounded.

## Held-out check (post hoc, 2026-11-16)

E14 scored a pick against the change that is best over all 16 seeds, the same seeds the replay draws from. This
check splits the 16 seeds at random into 8 selection and 8 held-out seeds (1000 splits); an allocator sees only the
selection seeds, and its pick is scored on the held-out ones. Code: `experiments/studies/e14_allocation_hard/heldout.py`
(a Python port of the three allocators, replayed against the committed E14 numbers within sampling noise);
output: `results/e14/e14_heldout.txt`. Not pre-registered.

At 72 runs (all 8 selection seeds), mean gain over the control, claimed on the selection seeds and delivered on the
held-out seeds:

| Task | Allocator | Claimed | Delivered | Pick has positive held-out gain | Runs spent |
|---|---|---|---|---|---|
| Topic | uniform | +0.0013 | +0.0010 | 86% | 72 |
| Topic | halving | +0.0013 | +0.0010 | 86% | 56 |
| Topic | LUCB | +0.0013 | +0.0009 | 82% | 51 |
| Topic | status quo (one run each) | +0.0028 | -0.0002 | 37% | 9 per decision |
| NLI | uniform | +0.0296 | +0.0294 | 99% | 72 |
| NLI | halving | +0.0295 | +0.0293 | 99% | 56 |
| NLI | LUCB | +0.0296 | +0.0291 | 99% | 51 |
| NLI | status quo (one run each) | +0.0310 | +0.0253 | 99% | 9 per decision |

- Replicated picks claim about what they deliver (topic +0.0013 against +0.0009 to +0.0010; NLI +0.0296 against
  +0.0291 to +0.0294), where the one-run status quo overstates on topic (+0.0028 claimed, -0.0002 delivered).
  The recovery in E14's headline scores is therefore not only the allocators recovering the ranking they were
  scored against.
- On topic the delivered gain is about one tenth of a point of accuracy, and the 16-seed interval for the best change
  includes zero, so "positive on held-out seeds" is weak evidence there: 82% to 86% of replicated picks were
  positive, against 37% for the status quo.
- Cost against delivered gain: LUCB at about 51 runs delivers the same gain as uniform at 72 (topic +0.0009 against
  +0.0010; NLI +0.0291 against +0.0294), about 30% fewer runs.
- Limits: the halves come from the same 16 seeds, the same changes, model and tasks, so this tests selection bias,
  not external validity; the held-out half has 8 seeds, so a pick's held-out gain is noisy and "best on held-out"
  (reported in the output) is a harsh criterion; 1000 splits reuse the same data.
