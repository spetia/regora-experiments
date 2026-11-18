# E1 — early ranking validity

**Question.** Does the ranking of configs early in training predict the final ranking?

**Design.** 3 learning rates × 2 weight decays × 3 seeds, 600 steps.

**Outcome.** Rule met (Kendall tau 0.87 at step 100, 1.00 from step 200, regret 0), but the grid had no crossing curves, so it is a weak test. Superseded by E1b.

| | |
|---|---|
| Run | `uv run python -m studies.e1_early_ranking.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e1_early_ranking.analyze` |
| Raw output | [`results/e1/`](../../results/e1) |
| Write-up | `research/experiment-results/E1_early_ranking.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
