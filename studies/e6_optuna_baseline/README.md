# E6 — Regora selector vs Optuna at equal compute

**Question.** Does Regora's schedule-aware selector find a better config than a conventional optimiser given the same compute?

**Design.** Replay on the E3 table. Optuna runs sequentially in three variants (TPE with Hyperband, random with median
pruning, grid with successive halving), each with a step budget equal to the schedule-aware selector's compute. Regret is
measured on held-out seeds.

**Outcome.** No advantage by the pre-registered rule: the schedule-aware selector (regret 0.0062) is within the margin of
Optuna's grid variant (0.0074). Fixed halving (0.0028 at less compute) beat all three Optuna variants, but that was not
the pre-registered claim.

| | |
|---|---|
| Run and analyse | `uv run python -m studies.e6_optuna_baseline.run` (needs `cargo build -p regora-cli` first) |
| Raw output | [`results/e6/`](../../results/e6) |

Write-up: `research/experiment-results/E6_optuna_baseline.md` in the main repo. The plan and decision rule are in the docstring of `run.py`, committed before the E3 data existed.
