# E7 / E7b — diagnosing a run by intervention

Date: 2026-11-11. Plans pre-registered in `experiments` before each run (commits "Pre-register E7", "pre-register E7b").
Code: `experiments/studies/e7_diagnosis/`, `experiments/studies/e7b_confirmation/`.
Raw: `experiments/results/e7/` (`e7.json`, `e7b.json`, logs, `*_analysis.txt`).

This is the first test of the "competing explanations" loop (spec 5.4): keep several causes in play, run probes,
update, stop when one cause dominates. E1b to E6 tested config selection, which is a different claim.

## Setup

TinyGPT (875k params), MPS, constant LR after a 50-step warmup. Five conditions injected on purpose, so the true
cause is known: healthy, LR too high (3e-2), LR too low (3e-4), evaluation noise (metric read on one random batch),
too little capacity (width 32). Each case trains 300 steps, then four probes run from that checkpoint:

| Probe | Reading | Cost (baseline steps) |
|---|---|---|
| halve LR, 100 steps | metric minus a paired control branch | 100 |
| double LR, 100 steps | same | 100 |
| re-evaluate on 4x the batches | absolute change in the metric | about 27 (1 for the noisy-eval case) |
| 2x wider model, 400 steps from scratch | metric minus the control | 1600 |

The control branch costs 100 steps once. Two passive readings are free: the metric at step 300 and its change since
step 200. 10 seeds per study, 50 cases each. E7 used seeds 31..40, E7b seeds 41..50.

## E7: the pre-registered analysis failed

Readings cut into three bins at pooled terciles, add-one smoothed naive Bayes, leave-one-seed-out.

| Rule | Result |
|---|---|
| H1 diagnosis right on at least 40 of 50 with all probes | **Not supported**: 34 of 50 |
| H2 choosing by information gain saves compute | **Not supported**: no policy stopped early |
| H3 at least 85% right when 90% sure | Supported, on 6 confident cases only |

Watching alone (same model on the passive readings) scored 29 of 50.

The tercile cut points were computed on the same 50 cases that the leave-one-seed-out rule was scored on. No
threshold was frozen on separate data in E7.

## Post hoc, not evidence

Keeping the readings continuous (Student-t predictive density per reading and condition) on the same E7 data scored
46 of 50 with the four probes, and 48 of 50 once the passive readings were added. I chose this after seeing the data
and tried eight probe orders on it. Output: `e7_posthoc_analysis.txt`.

## E7b: confirmation on fresh seeds

From here on thresholds and probe logic are frozen before the test data exist. The continuous model was fitted on
the 50 E7 cases, frozen, and tested once on 50 fresh cases. Passive readings are
read first; probes follow until the top posterior reaches 0.9.

| Policy | Right | Mean cost | Probes used | Confident (>= 0.9) | Right when confident |
|---|---|---|---|---|---|
| Information gain per cost | 49 of 50 | 206 | 1.34 | 98% | 98% |
| Cheapest first | 49 of 50 | 238 | 1.36 | 98% | 98% |
| Random order (mean of 24) | 98% | 705 | 1.47 | 96% | 99% |
| All four probes | 49 of 50 | 1922 | 4 | 90% | 100% |
| Watching only | 35 of 50 | 0 | 0 | 28% | 100% |

| Rule | Result |
|---|---|
| H1 diagnosis right on at least 40 of 50 | **Supported**: 49 of 50 |
| H2 selection saves compute | **Weak**: 29% of random's cost, but 87% of cheapest-first's (needed 80% or less) |
| H3 at least 85% right when 90% sure | **Supported**: 98% |
| H4 probes add at least 10 points over watching | **Supported**: 98% against 70% |

The one miss in both studies is a noisy-evaluation case read as healthy.

## What it shows

- Paired probe outcomes identify an injected cause on unseen seeds, at about a tenth of the cost of running every
  probe (206 against 1922 steps), and about two thirds of one 300-step training run.
- The win came from keeping the readings continuous. Binning discarded the precision that pairing provides (E0b,
  E0c). The 34 of 50 failure was caused by my outcome coding, and the fix was confirmed out of sample.
- Confidence is honest here: confident diagnoses were right 98% of the time or more.
- Information-gain ordering is not shown to beat "cheapest probe first". With four probes whose informativeness and
  cost line up this way, a trivial ordering does as well. The spec's selection formula has no demonstrated advantage.

## What it does not show

- One small model, one dataset, one machine. The likelihoods were learned on the same model family they were tested
  on, which is the best case. E7c tests a larger model and transfer.
- Causes were single, injected and mutually exclusive. Real runs have mixed causes and causes outside the list; the
  method has no "none of the above" beyond the healthy condition.
- Watching uses the absolute metric level, which will not transfer across models or datasets.
- "Healthy" here means the base config, not an optimum: halving its LR still helped slightly over 100 steps.
