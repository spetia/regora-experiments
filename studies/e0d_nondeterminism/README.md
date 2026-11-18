# E0d — GPU non-determinism

**Question.** Does run-to-run non-determinism on a GPU destroy the pairing?

**Design.** Repeat identical branches on the device and compare the repeat noise with the E0c effect sizes.

**Outcome.** On Apple MPS repeats differ by at most 6e-8 and are not bitwise identical; every E0c effect survives. **CUDA has not been run** (no GPU quota yet).

| | |
|---|---|
| Run | `uv run python -m studies.e0d_nondeterminism.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | none; `run.py` prints the result |
| Raw output | [`results/e0d/`](../../results/e0d) |
| Write-up | `research/experiment-results/E0d_mps_nondeterminism.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
