# E0c — smallest detectable effect

**Question.** How small an intervention effect can paired branches resolve?

**Design.** Weight decay 0.1 and 1.0 and gradient clip 0.25 against control, paired by data seed, from the E0b checkpoints.

**Outcome.** All three were detectable: +0.0004 (SD 0.0002), +0.0067 (SD 0.0017) and +0.0063 (SD 0.0035). Detectable does not mean useful: the smallest is far below any practical threshold.

| | |
|---|---|
| Run | `uv run python -m studies.e0c_small_effects.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e0c_small_effects.analyze` |
| Raw output | [`results/e0c/`](../../results/e0c) |
| Write-up | `research/experiment-results/E0c_small_effects.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
