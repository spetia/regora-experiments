# regora-experiments

Research experiments for Regora on a small character-level GPT. Each study's plan and decision rule are written in its
`run.py` docstring and committed before the run.

```text
tinygpt/    model, data and a deterministic training loop; `trainer.py` is the Regora harness adapter
scale/      larger models on real data: a 25M-parameter byte-level GPT on enwik8, a ResNet-18 on CIFAR-100
studies/    one folder per study: `run.py` produces the data, `analyze.py` applies the pre-registered rule
results/    raw outputs, one folder per study
demo/       example experiment and capability declaration for `regora run`
```

| Study | Question |
|---|---|
| `e0_noise_floor`, `e0b_replication` | How large is seed noise, and do paired checkpoint branches reduce it? |
| `e0c_small_effects` | How small an intervention effect can paired branches detect? |
| `e0d_nondeterminism` | Does GPU non-determinism break pairing? |
| `e1_early_ranking`, `e1b_crossing_grid` | Does early ranking of configs predict the final ranking? |
| `e2_paired_vs_halving` | Does a paired sequential test beat successive halving at equal compute? |
| `e3_schedule_aware` | Does a cut rule that protects configs during warmup beat fixed halving? |
| `e4b_loop_overhead` | What does telemetry cost a real training loop? |
| `e6_optuna_baseline` | Do Regora's selectors beat Optuna at equal compute? |
| `e12_paired_eval` | Is a difference between two fine-tunes real? Item pairing, training noise and seed pairing on a 360M LM (needs `transformers` and `datasets`) |
| `e13_allocation` | Does adaptive allocation of seeds find the best change in fewer runs? Replay of `regora allocate` on the E12 results |
| `e14_allocation_hard` | The same on eight changes and fresh seeds, plus one-run-per-change as the status quo |
| `e15_mixed_causes` | The diagnosis rule on mixed and unnamed causes. Stopped after 8 of 100 cases; untested |
| `e16_gate` | A tuning loop with and without a referee: gain claimed against gain delivered |

Run from this directory with the workspace environment:

```sh
REGORA_DEV=mps uv run python -m studies.e2_paired_vs_halving.run
uv run python -m studies.e2_paired_vs_halving.analyze
uv run python -m tinygpt          # self-check
uv run python -m scale            # self-check
```

Write-ups are in `writeups/`; dependencies are pinned in `pyproject.toml` and `uv.lock`.

**What can be re-run.** The analysis scripts and replays (`analyze.py`, `replay.py`, `heldout.py`, `scaling_curve/analyze*.py`)
re-run on the shipped `results/`. Re-running the training itself is not possible from this repository alone: `tinygpt/trainer.py`
imports the Regora harness (`regora` Python module) and E6 needs `regora-cli`, both in a private repository. `PREREGISTRATION.md`
lists each plan's commit hash and time from the private history.

Part of [regora-ai/regora](https://github.com/regora-ai/regora).

## License

All content in this repository (plans, code, data and write-ups) is released under
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0); the full text is in `LICENSE`. Cite the accompanying paper, "Is the Change Real? Paired Verdicts, Early
Stopping, and the Limits of Run Selection for ML Training Changes" (Spetia SI Labs, 2026).

This repository is a squashed snapshot of the research repository's `main` branch. Plans are the versions committed before
each run; see `PREREGISTRATION.md` for hashes and times from the private history.
