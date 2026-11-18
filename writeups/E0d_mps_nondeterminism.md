# E0d (MPS): does device non-determinism destroy pairing?

**Label: MEASURED, small. Plan committed before the run (`experiments/studies/e0d_nondeterminism/run.py` docstring). CUDA run NOT done.**

Setup: E0b checkpoint 0 rebuilt on CPU, then the same control branch (same data seed) run 3 times on Apple MPS for each of 4 data seeds, 200 steps, torch 2.15.0. Raw: `experiments/results/e0d/e0d_mps.json`.

Results:
- No two repeats were bitwise identical, so MPS is not exactly deterministic here.
- Max absolute difference among repeat val-loss curves: 6.0e-8 (about one float32 ulp at this loss scale). Pooled endpoint SD across repeats: 4.0e-9.
- Pre-registered survival rule: every CPU effect from E0b/E0c (smallest: wd 0.1, 0.0011) exceeds the resulting MDE (about 1e-8) by orders of magnitude, so pairing survives.

Limits:
- Float-level noise at 200 steps says nothing about longer horizons, where chaotic amplification of tiny differences can grow. Horizon dependence was not tested.
- Only the control arm, one checkpoint, a tiny model. Larger models and mixed-precision training will be noisier.
- CUDA kernels (atomics in some backward ops, cuDNN algorithm selection) can behave differently from MPS. **Needs the CUDA run**, blocked on GPU access (AWS G-instance quota request pending in us-east-1; 4070 box offered but no access yet).
