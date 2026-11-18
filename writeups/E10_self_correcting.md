# E10 — a run that corrects its own learning rate

Date: 2026-11-13. Plan pre-registered in `experiments` (`studies/e10_self_correcting/PLAN.md`, commit
"Pre-register E10"). Code: `experiments/studies/e10_self_correcting/`. Raw: `experiments/results/e10/`
(`e10_slim.json`, `e10_stout.json`, `e10_lanky.json`, logs, pilots, `e10_analysis.txt`).

## Setup

Controller: every 300 steps, train three branches from the same state on the same data at 0.5x, 1x and 2x the
current LR and continue from the one with the lowest validation loss. 2400 steps, so 7200 steps of compute.

Baseline at the same 7200 steps: successive halving over six LRs {5e-4 .. 1.6e-2} (keep 3 at step 600, 2 at 1200,
1 at 1800), once with constant LR and once with cosine decay.

Three models not seen before (layers x width): slim 5x96, stout 3x256, lanky 8x128. Seeds 91..100. The controller
starts from 2e-3 (default), and from 5e-4 and 1.6e-2 to test robustness. Selection uses the usual validation set;
all scores are on 40 separate validation batches. A10G, CUDA.

## Result

Test loss, controller minus baseline, paired over 10 seeds, 95% interval. Negative favours the controller.

| Model | vs halving, constant LR | vs halving, cosine | start 5e-4 vs default | start 1.6e-2 vs default |
|---|---|---|---|---|
| slim | -0.032 ± 0.042 | -0.010 ± 0.026 | +0.039 ± 0.025 | +0.019 ± 0.030 |
| stout | +0.082 ± 0.045 | +0.028 ± 0.044 | -0.019 ± 0.080 | +0.173 ± 0.070 |
| lanky | +0.057 ± 0.027 | +0.031 ± 0.037 | +0.025 ± 0.041 | +0.085 ± 0.043 |

| Rule | Result |
|---|---|
| H1: no worse than halving on every model | **Not supported**: holds on slim only |
| H2: better than halving on at least 2 of 3 models | **Not supported**: on none |
| H3: the starting LR does not matter on every model | **Not supported**: on none |

Seeds on which the controller beat halving (constant, cosine): slim 7 and 6 of 10, stout 1 and 4, lanky 0 and 3.

## What it shows

- **A greedy LR controller does not replace an LR search.** On two of three models it is clearly worse than
  halving over constant LRs, the weaker baseline, and it is never reliably better than either.
- **The cause is the one the pilot showed.** From the default start the median LR path rises for one or two
  segments and then halves on almost every segment, ending at 1/16 of the starting LR or lower in 19 of 30
  runs (most often 1.25e-4). Halving the LR nearly always wins the next 300 steps, so a rule that keeps the 300-step winner decays
  too early. This is the same transient that produced E9's false "LR too high" diagnoses.
- **It does not recover from a bad start.** Starting at 1.6e-2 costs 0.17 on stout and 0.08 on lanky against the
  default start. Starting low costs less, but the controller only doubles once per segment and stops climbing
  early.
- No run diverged or ended short of 2400 steps.
- Halving itself was a good tuner here: it picked the hindsight-best constant LR on 24 of 30 seeds.

## Conclusion

All three registered claims failed, as the plan predicted from the pilot. Short branches from a checkpoint read
the direction of a large LR fault (E9), but a branch's result after 300 steps is biased towards lower LRs, so it
cannot be used as-is to steer the LR through a run.

Post hoc, not tested: the bias is specific to comparing branches at different LRs over a short horizon. Remedies
to test separately are a longer horizon before judging, judging a branch after returning it to the common LR,
or restricting the controller to raising the LR and leaving decay to a schedule.

## Limits

One dataset; models under 2M parameters; 2400 steps; one controller design with one margin chosen on five pilot
cases. The result is about this greedy rule, not about every way of using branches to set the LR.
