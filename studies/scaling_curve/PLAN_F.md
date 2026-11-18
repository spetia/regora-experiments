# Stage F plan, the gate test: how much of an automated tuning loop's "progress" is real?

PRE-REGISTERED, as an addition to `PLAN.md`. Written after stages A to C and the literature check
(`LITERATURE.md`) and before any stage-F run; stages D and E were running or queued and no result of theirs had
been read. Ran before this file was committed: `stage_f.py smoke`, a mechanics check on the CPU with a tiny
configuration.

## Why

Loops that propose a change, score it once and keep it if the score went up are how automated research agents
and weight-editing procedures work. The literature check found many reports that such loops produce results
that do not hold, and no measurement of what a paired, replicated acceptance rule changes inside one. This
measures it on a case small enough to know the truth.

## Setup

`study.py` unchanged; the loops are in `stage_f.py`. Model 4x128 (0.88M), base learning rate 0.004 (what
`A_s1` tuned to), weight decay 0, bfloat16, deterministic kernels, batch 32. Per replicate: one checkpoint at
step 600 at the base configuration; a configuration is scored by its validation loss 200 steps after the
checkpoint on a given data order.

A scripted proposer draws each candidate from the incumbent: with probability 0.2 a null (the incumbent with 1
example in 32 swapped, so no systematic effect), 0.4 a learning-rate move (x0.8, x0.9, x1.1 or x1.25, kept
within 0.25x to 4x), 0.4 a new weight decay (0, 0.01, 0.03 or 0.1). Both loops use the same random stream.

- **Keep-if-better loop**: 96 rounds. One run of the candidate on a fresh order; kept if its loss is below the
  best single score so far. Claimed gain: the first score minus the best score.
- **Gated loop**: 12 rounds. Candidate and incumbent on 4 fresh orders each, paired; kept if the 95% t interval
  of the paired difference is below zero (the harness's "real" verdict). Claimed gain: the sum of the mean
  paired improvements of the changes kept.
- Both use 96 training runs (plus one starting run for the first loop).
- **Truth**: the base and each loop's final configuration on 12 further orders that no loop used; the true gain
  is the mean paired improvement over the base.

8 replicates (checkpoint and proposal seeds differ). Estimated GPU time: about 2 hours.

## Decision rules

Over the 8 replicates, with 95% t intervals. `stage_f.py analyze`, committed with this plan, computes them.

- **F1 (the simple loop overclaims).** The mean of claimed minus true gain for the keep-if-better loop is above
  zero, with its interval excluding zero.
- **F2 (the gate removes most of it).** The gated loop's mean claimed minus true gain is, in absolute value, at
  most half the keep-if-better loop's. Uninformative if F1 is not supported.
- **F3 (nulls).** The gated loop keeps at most 5% of the null candidates it is shown. The keep-if-better loop's
  share is reported.
- **F4 (which loop ends better on the same compute).** The mean of true gain, gated minus keep-if-better, with
  its interval: gated better, keep-if-better better, or unresolved. No direction is predicted. The simple loop
  tries eight times as many candidates, so it may well end ahead.

## Known limits, stated in advance

- A scripted proposer over two knobs is not an agent, and a tiny model at 200 steps is not a real task. The
  test isolates the acceptance rule; it says nothing about fabrication, leakage or reward hacking, which the
  literature reports as the larger failures.
- Lowering the learning rate helps at 200 steps in this setup (stage A), so real gains exist to be found. With a
  flat landscape the overclaim would be all there is; with a steep one it would be small beside the true gain.
- The keep-if-better loop compares against its best single score, which is the usual practice and the reason it
  overclaims; an agent that re-ran its incumbent each round would overclaim less.
- 8 replicates. The gated loop has 12 rounds, so it sees few nulls (about 19 expected in all), and F3 is coarse.
- Short-horizon gain is not final-training gain (E11).
