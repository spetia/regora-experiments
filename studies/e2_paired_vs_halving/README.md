# E2 — paired sequential test vs successive halving

**Question.** At equal compute, does dropping configs by a paired significance test beat rank-and-cut halving?

**Design.** The E1b grid with 8 fresh seeds (64 runs). Selectors are replayed offline on all 56 three-seed subsets.

**Outcome.** No. The paired test had regret 0.0180 at 47% compute against 0.0159 for the matching halving schedule. Delaying the second cut to step 750 reached regret 0.0011 at 56% compute, but that schedule was singled out after seeing the data.

| | |
|---|---|
| Run | `uv run python -m studies.e2_paired_vs_halving.run` (set `REGORA_DEV=mps` or `cuda` for GPU) |
| Analyse | `uv run python -m studies.e2_paired_vs_halving.analyze` |
| Raw output | [`results/e2/`](../../results/e2) |
| Write-up | `research/experiment-results/E2_paired_test_vs_halving.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run. All numbers are measured on one
small model and do not establish behaviour at scale.
