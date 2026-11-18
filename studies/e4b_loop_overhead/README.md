# E4b — telemetry overhead on a real training loop

**Question.** How much does recording telemetry slow down training?

**Design.** Three arms (no callback, read the loss, read the loss and record two signals), 200 steps each, 7 rounds in
rotated order with the first discarded.

**Outcome.** Not measurable: −1.25% with a 95% interval of −2.54% to +0.05% on Apple MPS, against run-to-run noise of
about ±1.3%. No records dropped.

| | |
|---|---|
| Run and analyse | `REGORA_DEV=mps uv run python -m studies.e4b_loop_overhead.run` |
| Raw output | [`results/e4b/`](../../results/e4b) |
| Write-up | `research/experiment-results/E4b_loop_overhead.md` in the main repo |

The plan and decision rule are in the docstring of `run.py`, committed before the run.
