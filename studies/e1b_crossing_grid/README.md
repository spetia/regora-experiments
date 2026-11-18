# E1b — early ranking on a grid with crossings

**Question.** Does successive halving still pick well when some configs start slowly?

**Design.** 4 learning rates × 2 warmups × 3 seeds, 1000 steps, with per-seed halving simulated at steps 250, 500 and 750.

**Outcome.** Rule narrowly failed (tau 0.71 vs 0.7 required; regret 0.0107 vs 0.01 allowed). Halving dropped the slow-starting winner in 2 of 3 seeds but never picked a bad config.

| | |
|---|---|
| Run | `uv run python -m studies.e1b_crossing_grid.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e1b_crossing_grid.analyze` |
| Raw output | [`results/e1b/`](../../results/e1b) |
| Write-up | `research/experiment-results/E1b_crossing_grid.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
