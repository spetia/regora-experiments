# E0b: replication of E0 with a standard schedule and several checkpoints

**Label: MEASURED, small. Plan committed before the run (`experiments/studies/e0b_replication/run.py` docstring, commit "pre-register E0b").**

## Setup
Same model/data as E0. Base LR 1e-3, warmup 50, cosine to 10% over 600 steps. 3 checkpoints (init seeds 0, 1, 2) at step 300. From each: 8 data seeds x LR multiplier {1, 0.5, 2}, 200 steps, shared batches across arms (CRN). Primary endpoint: mean val loss at h = 100, 150, 200 (chosen after E0, so a hypothesis, not a fresh pre-registration). Determinism repeat: bitwise identical at all 3 checkpoints. Raw: `experiments/results/e0b/e0b.json`, `e0b_analysis.txt`.

## Results
| LR mult | pooled effect (val loss vs 1x) | pooled paired SD | effect/SD | per-checkpoint effect/SD |
|---|---|---|---|---|
| 0.5x | +0.014 (worse) | 0.0057 | +2.5 | +5.6, +0.3, +3.0 |
| 2x | -0.040 (better) | 0.0080 | -4.9 | -4.5, -5.1, -5.2 |

- Control SD across data seeds: 0.011-0.014 (E0 with constant LR 3e-3: 0.03-0.06).
- Rank agreement of the {0.5x, 1x, 2x} ordering at h=50 / 100 / 150 vs h=200: **0.50 / 0.62 / 0.83** (chance 0.17). E0 gave 0.08-0.33.
- Normal-approximation n80 values below about 2 (pooled: 1.3 and 0.3) are not meaningful; read them as "a handful of paired replicates".

## Pre-registered decision rule
"Power-limited if pooled effect/SD < 1." **Not triggered for either multiplier** in this setup.

## Interpretation and limits
- E0's pessimistic reading (effect about the size of the noise, early ranks uninformative) does **not** replicate under a stable schedule. The most likely reason is that E0's constant LR of 3e-3 was near instability, which inflated noise. This is an explanation consistent with the data, not a tested cause.
- Effect heterogeneity across checkpoints is real for 0.5x (checkpoint 1: +0.002, effect/SD 0.3). Three checkpoints are too few to characterise it.
- The 2x LR arm is better here because this model at step 300-500 is undertrained, so larger effects are easy to see. Interventions with smaller true effects (e.g. weight decay, clip norm) are the more demanding test and have not been run.
- Still one tiny model, one corpus, one intervention type (LR scaling), CPU only. Nothing here speaks to scale.
- Consequence for the plan: successive halving (F4) and CRN-paired designs are not ruled out. Rank agreement at h=50 (0.5) is moderate. The next informative experiment is a small-effect intervention (weight decay, grad clip) to find the detection limit.
