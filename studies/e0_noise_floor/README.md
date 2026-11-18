# E0 — seed noise floor and branch correlation

**Question.** How large is seed-to-seed noise, and does branching from a shared checkpoint with a shared data order reduce it?

**Design.** 6 from-scratch runs and 12 paired branches at three learning-rate multipliers from one checkpoint. Constant LR 3e-3, CPU.

**Outcome.** Inconclusive. Effects were 0.4–1.4 seed SDs and rank agreement was 0.08–0.33, so the rule was neither met nor cleared. The first run was invalid (optimizer state aliasing) and is kept, labelled, in the results folder.

| | |
|---|---|
| Run | `uv run python -m studies.e0_noise_floor.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e0_noise_floor.analyze` |
| Raw output | [`results/e0/`](../../results/e0) |
| Write-up | `research/experiment-results/E0_noise_floor.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
