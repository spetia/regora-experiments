# E4b — telemetry overhead on a real training loop

Date: 2026-11-11. Plan and rule committed in `experiments/studies/e4b_loop_overhead/run.py` before the run.
Raw: `experiments/results/e4b/e4b_mps.json`, `e4b_mps.log`. All values measured, Apple MPS, shared laptop.

## Setup

TinyGPT, 200 steps per run, three arms per round in rotated order, 7 rounds with the first discarded:
no callback; a callback that only reads the loss (`sync`); the same callback recording two signals per step (`record`).

## Result

| Arm | Median ms/step |
|---|---|
| No callback | 60.58 |
| Read loss each step | 61.15 |
| Read loss and record two signals | 60.62 |

| Overhead | Mean | 95% interval |
|---|---|---|
| Telemetry (record vs sync) | −1.25% | −2.54% to +0.05% |
| Reading the loss (sync vs none) | +1.14% | −1.63% to +3.91% |

400 records per run, 0 dropped, 0 write errors.

## Verdict

**LOW by the pre-registered rule** (upper end of the interval below 1%).

## What it shows

Telemetry cost is below what this benchmark can resolve. The negative mean is noise, not a speed-up: run-to-run variation
on a shared machine is about ±1.3%, and two signals per step at roughly 90 ns each is about 0.0003% of a 61 ms step.
The cost of reading the loss each step is also not distinguishable from zero here.

## Limits

One small model with a slow step (61 ms). A loop with millisecond steps or hundreds of signals per step would be a harder
test. Memory overhead was not measured. CUDA, where a forced sync can cost more, was not measured.
