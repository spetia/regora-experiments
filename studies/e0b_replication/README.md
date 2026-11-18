# E0b — replication with a realistic schedule

**Question.** Does E0's pairing effect hold with a cosine schedule and several checkpoints?

**Design.** 3 checkpoints × 8 data seeds, LR 1e-3 with cosine decay, treatments 0.5× and 2× LR.

**Outcome.** Pairing works. Pooled effects were +2.5 and −4.9 control SDs (control SD about 0.012). Rank agreement rose from 0.50 at 50 steps to 0.83 at 150. One checkpoint showed almost no effect for 0.5× LR.

| | |
|---|---|
| Run | `uv run python -m studies.e0b_replication.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e0b_replication.analyze` |
| Raw output | [`results/e0b/`](../../results/e0b) |
| Write-up | `research/experiment-results/E0b_replication.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
