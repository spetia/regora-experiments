# E2 — paired sequential test vs successive halving

Date: 2026-11-11. Plan and rule committed in `experiments/studies/e2_paired_vs_halving/run.py` before the run.
Raw: `experiments/results/e2/e2.json`, `e2_analysis.txt`. All values measured.

## Setup

Same grid as E1b: LR {1e-3, 2e-3, 4e-3, 8e-3} × warmup {50, 300}, 1000 steps, TinyGPT on MPS. 8 fresh seeds (3–10), 64 runs.
Selectors are replayed offline on all 56 three-seed subsets. Truth is the 8-seed mean final loss.

- **SH(r1, r2, r3):** rank by 3-seed mean, keep 4, 2, 1 at the three rungs.
- **F3:** at steps 250, 500, 750 drop a config when its paired difference to the leader is significant (one-sided 95% t, 3 seeds).

## Result

| Selector | Compute | Mean regret | P(picks best) |
|---|---|---|---|
| F3 (paired test) | 0.466 | 0.0180 | 0.27 |
| SH(250, 500, 750) | 0.469 | 0.0159 | 0.34 |
| SH(250, 500, 1000) | 0.500 | 0.0149 | 0.43 |
| SH(250, 750, 1000) | 0.562 | **0.0011** | **0.96** |
| SH(500, 750, 1000) | 0.688 | 0.0011 | 0.96 |
| Full training | 1.000 | 0.0000 | 1.00 |

Random pick: regret 0.1246. Seed SD of the final loss: 0.027–0.045. Best config: LR 8e-3 with warmup 300 (2.2690), ahead of
LR 4e-3 with warmup 50 (2.2905).

## Verdict

**By the letter of the rule F3 is "supported": no halving schedule has both lower-or-equal compute and lower-or-equal regret.
That is a technicality and I do not accept it as support.** F3 used 0.3% less compute than SH(250, 500, 750) and had higher
regret (0.0180 vs 0.0159). The rule should have required a margin. On the substance, F3 did not beat halving.

## What it shows

1. **The paired test does not fix the slow-starter problem.** The eventual winner is genuinely behind at steps 250 and 500,
   so a test of the *current* gap drops it with confidence. The error is bias (curves cross), not noise. More seeds or a
   stricter threshold would not help.
2. **Cut timing is what matters on this grid.** Moving the second cut from step 500 to 750 took regret from 0.0159 to 0.0011
   and P(best) from 0.34 to 0.96, for 56% of full compute instead of 47%. This comparison is post hoc: the schedule family was
   pre-registered as a baseline, but singling out the best member was not. It needs confirmation on a different grid.
3. **Pairing across configs helps only modestly.** At step 250 the median SD of a config difference was 0.048 paired vs 0.074
   unpaired. Pairing is far stronger for branches from a shared checkpoint (E0b, E0c) than for from-scratch configs.

## Limits

Subsets overlap, so the 56 replays are not independent. Truth includes each selector's own seeds. One model, one dataset, one
grid, with a single slow-starting winner; the timing result may be specific to warmup 300.

## Consequence

- Do not build the first selector around a significance test on current loss. It is confidently wrong when curves cross.
- The open question is now prediction: when to cut depends on whether a config's curve will cross. E3 should test a cut rule
  that is aware of schedule state (no cut before every surviving config has finished warmup plus a margin) against the best
  fixed schedule, on a grid with different warmups and schedule lengths.
