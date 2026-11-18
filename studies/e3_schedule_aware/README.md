# E3 — schedule-aware cut rule vs fixed halving

**Question.** Does a halving rule that will not cut a config until 200 steps after its warmup beat fixed halving?

**Design.** 3 learning rates × 3 warmups × 8 fresh seeds (72 runs), 1000 steps. Selectors are replayed on all 56
three-seed subsets and scored on the 5 held-out seeds.

**Outcome.** No. The schedule-aware rule was worse: regret 0.0062 at 58% compute, against 0.0028 at 44% for fixed halving.
E2's finding that later cuts help was not confirmed either.

| | |
|---|---|
| Run | `uv run python -m studies.e3_schedule_aware.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e3_schedule_aware.analyze` |
| Raw output | [`results/e3/`](../../results/e3) |
| Write-up | `research/experiment-results/E3_schedule_aware.md` in the main repo |

The plan and decision rules are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
