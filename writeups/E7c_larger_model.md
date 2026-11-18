# E7c — diagnosis on a larger model, and transfer from the small one

Date: 2026-11-12. Plan pre-registered in `experiments` (commit "Pre-register E7c") after a one-seed pilot that
changed no setting. Code: `experiments/studies/e7c_larger_model/`. Raw: `experiments/results/e7/e7c.json`,
`e7c.log`, `e7c_analysis.txt`. Background and method: `E7_diagnosis.md`.

## Setup

The E7 study unchanged, on a 6-layer, width-256 model (about 5.5x the parameters), Tesla T4, CUDA, torch 2.15.0.
Learning rates were not retuned. 20 fresh seeds 51..70, 100 cases. The E7b analyzer was used unchanged.

- WITHIN: fit on seeds 51..60, test on the 50 cases of seeds 61..70.
- TRANSFER: fit on the 50 small-model E7 cases (MPS), test on the same 50 cases, probes only.

## Result

| | Right (all probes) | Info gain per cost: right / cost | Cheapest first: right / cost | Watching only | Right when >= 0.9 sure |
|---|---|---|---|---|---|
| WITHIN | **45 of 50** | 45 of 50 / 442 | 43 of 50 / 342 | 33 of 50 | 90% |
| TRANSFER | **32 of 50** | 31 of 50 / 1270 | 31 of 50 / 1270 | not applicable | 62% |

| Rule | Result |
|---|---|
| H1 within: at least 40 of 50 | **Supported**: 45 of 50 |
| H2 within: selection saves compute | **Weak**: 54% of random's cost, but more than cheapest-first's |
| H3 within: at least 85% right when 90% sure | **Supported**: 90% |
| H4 within: probes add at least 10 points over watching | **Supported**: 90% against 66% |
| H5 transfer: at least 40 of 50 | **Not supported**: 32 of 50 |

## What it shows

- With labelled examples from the same model, diagnosis holds on a larger model and a different device (45 of 50).
  Accuracy is lower than on the small model (49 of 50); the misses are mostly noisy evaluation read as healthy (3).
- Likelihoods learned on the small model do not carry over. Worse, they are confidently wrong: 62% right among
  diagnoses made with at least 90% confidence. A product that shipped fixed likelihoods would mislead.
- The transfer failure is concentrated. All 10 healthy cases were read as "LR too high". On the larger model the
  unchanged base LR of 4e-3 behaves that way: halving it helps and doubling it hurts far more than on the small
  model. "LR too low" (10 of 10) and capacity (8 of 10) transferred; their probe signatures kept their sign.
- Information-gain ordering again shows no advantage over cheapest-first.

## What it does not show

- Whether any calibration short of full relabelling rescues transfer (for example scaling readings by the run's own
  paired noise, or classifying by the sign pattern of the probe effects). Untested; any such analysis of this data
  would be post hoc.
- Anything about real, mixed or unlisted causes, or models beyond a few million parameters.
- The healthy-as-LR-too-high reading may be a correct description of the larger model rather than an error, since
  the base LR was never tuned for it. The pre-registered labels count it as wrong, and so does this note.
