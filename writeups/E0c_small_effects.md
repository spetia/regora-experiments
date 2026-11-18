# E0c: detection limit for small-effect interventions

**Label: MEASURED, small. Plan committed before the run (`experiments/studies/e0c_small_effects/run.py` docstring).**

Same setup as E0b (LR 1e-3 cosine, 3 checkpoints at step 300, 8 data seeds, 200-step branches, shared batches). Control curves reused from E0b; one control per checkpoint recomputed and matched bitwise (the script asserts it). Endpoint: mean val loss at h = 100, 150, 200. Raw: `experiments/results/e0c/e0c.json`, `e0c_analysis.txt`.

| Arm (vs control wd=0, clip=1.0) | pooled effect | pooled paired SD | effect/SD | per-checkpoint effects | MDE (n=8, 80% power) | Detectable (pre-registered rule) |
|---|---|---|---|---|---|---|
| wd = 0.1 | +0.0004 | 0.0002 | 2.4 | 0.0011, 0.0001, 0.0002 | 0.0002 | yes |
| wd = 1.0 | +0.0067 | 0.0017 | 4.0 | 0.0133, 0.0026, 0.0043 | 0.0017 | yes |
| clip = 0.25 | +0.0063 | 0.0035 | 1.8 | 0.0093, 0.0029, 0.0069 | 0.0035 | yes |

Positive effect = higher (worse) val loss than control. For reference, E0b's 2x LR effect was -0.040 and its paired SD 0.008.

## What this supports and what it does not
- Supported: with shared batches, branches from one checkpoint stay so correlated that the paired SD scales down with the perturbation size (0.0002 for the smallest perturbation vs 0.008 for a 2x LR change). Statistical detection of very small effects (down to about 2e-4 in val loss at n=8) is therefore feasible **in this setup**.
- **Detectable is not the same as useful.** A 4e-4 change in val loss on this model is practically negligible. The experiment shows that the measurement is not the bottleneck here. It says nothing about whether such an intervention matters.
- The "effect" is on one fixed 20-batch validation set. Whether it transfers to fresh data is unmeasured.
- The wd = 0.1 pooled effect is dominated by one checkpoint (0.0011 vs 0.0001, 0.0002). Three checkpoints cannot characterise that heterogeneity.
- Paired SD this small depends on exact determinism and shared batches. Real training with non-deterministic kernels, different hardware, or data pipelines that cannot replay batches will have a lower pairing correlation. That is untested here and is the main reason not to generalise.
- Still one tiny model, one corpus, CPU only.

## Consequence for the plan
The harness's value rests on **deterministic replay plus shared batch order**, which was achievable on CPU here. Making `deterministic_replay` and a shared-data-order capability first-class parts of the adapter contract is justified by this result (the spec already lists deterministic replay as a capability). A next test would be GPU/MPS non-determinism and how much pairing correlation survives it.
