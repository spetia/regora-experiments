# E15 plan: what does the diagnosis rule say when the cause is mixed, or not one it knows?

PRE-REGISTERED before the test run. Written after E14 was started.

E9's rule knows three causes: noisy evaluation, LR too low, LR too high. Every test so far gave it runs with
exactly one of them, or none. A team's stalled run may have two at once, or a cause the rule has no name for.
The product must not answer those with confidence. E11 already found the rule unreliable on these models (25
of 60), so this measures how it fails, not whether it works.

## Claims

1. Given two causes at once, the rule names one that is really there.
2. Given a fault it has no name for, the rule does not name a cause.
3. Healthy runs are not told "LR too low" or "noisy evaluation" (E11's H4, repeated on new seeds).

## What is fixed in advance, and what the pilot is for

- Models, training, checkpoint, probe branches and stored fields: E11's, unchanged (25.6M GPT on enwik8,
  ResNet-18 on CIFAR-100; 1000 steps, then control, half-LR and double-LR branches on 3 data orders for 300
  steps; default kernels, as in E11's main run). Healthy LR: E11's tuning. No 1500-step follow-up.
- The rule: `studies.e9_longer_probes.analyze.diagnose`, called with its defaults, unchanged.
- Conditions, each on seeds 201..210 per task (50 cases per task):
  - healthy;
  - low and noisy: LR 1/12 of healthy, evaluated on one fresh batch;
  - high and noisy: LR 3x healthy, evaluated on one fresh batch;
  - heavy decay: weight decay 0.5 (language; its own value is 0) or 1e-2 (vision; its own is 5e-4), healthy
    LR. The rule has no name for this;
  - low and decay: LR 1/12 with the heavy decay. One named cause, one unnamed.
  A fault lasts the whole case, probe branches included.
- Pilot: seed 12, every condition, both tasks, to check that the cases build and that heavy decay does damage
  (a worse checkpoint loss than healthy). Its outcome is recorded at the end of this file. Seed 12 is not used
  below.

## Decision rules (`analyze.py`)

- H1. Over the 20 mixed cases per task (low and noisy, high and noisy), the rule names a cause present in that
  case in at least 80%.
- H2. Over the 10 heavy-decay cases per task, the rule names any cause (noisy evaluation, LR too low, LR too
  high) in at most 20%.
- H3. Over the 10 healthy cases per task, the rule says "LR too low" or "noisy evaluation" in at most 10%.
- Reported only: the rule's answers per condition; how often "low and decay" is read as LR too low; the
  checkpoint loss of each condition next to healthy.

## What is expected

- H1 should hold if the noise check fires first, as it did for 19 of 20 noisy cases in E11. That would also
  mean the LR fault goes unmentioned in every mixed case, which is reported.
- H2 is expected to fail on the language model: in E11 the rule read all 10 healthy language runs as "LR too
  high", because halving a constant LR acts as decay over 300 steps. Vision is open.
- H3 held in E11 (0 of 20).

## Known limits

Two models, 10 seeds per condition, one unnamed fault at one strength, chosen by us. A real unnamed cause (a data
bug, a bad architecture) may look nothing like heavy decay. The rule has no way to say "I do not know"; "healthy"
is its only other answer, and for a faulty run that is also wrong, only less misleading. Default kernels, so the
language model's branches carry GPU non-determinism (E11).

## Pilot outcome (`results/e15/pilot_*.json`, seed 12, one case per condition and task)

No rule, threshold or fault strength was changed after the pilot. The rule was not run on the pilot cases.

- Every condition built on both models. A case takes about 6 minutes on the language model and under 3 on
  vision (A10G, both tasks sharing it), so the 100 cases take roughly 6 hours.
- Heavy decay does damage, as the plan required: checkpoint loss 1.3388 against 1.2933 healthy on the language
  model, 2.8746 against 2.5770 on vision. On the language model the damage is small (0.046), and one seed
  cannot say how it compares with seed-to-seed spread; it is reported with the results.
- Low LR is far worse than healthy on both (1.73 and 3.35 at the checkpoint, read on one noisy batch).
- One seed says nothing about H1 to H3.

## Stopped (2026-11-15)

The test run was stopped 24 minutes in, after 8 of 100 cases (1 language, 7 vision), by a product decision:
Regora's direction moved to refereeing experiments, and diagnosis is no longer on the roadmap
(`wiki/roadmap.md`). The partial files are kept in `results/e15/` for the record. The rule was not run on them
and no claim is made from them. H1 to H3 are untested. Anyone resuming this must use fresh seeds.
