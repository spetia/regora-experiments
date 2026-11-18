# E16 — does a referee keep a tuning loop honest?

Date: 2026-11-16. Plan pre-registered in `experiments` (`studies/e16_gate/PLAN.md`, commit "Pre-register E16"). Code:
`experiments/studies/e16_gate/` (`run.py`, `replay.py`, frozen before the run). Raw: `experiments/results/e16/`
(`e16_topic.json`, `e16_nli.json`, logs, pilot, `e16_replay.txt`).

## Setup

SmolLM2-360M, E12's procedure unchanged, on topic and NLI. A grid of 24 configurations (LR x {0.5, 0.71, 1.0, 1.41}
times label smoothing {0, 0.1} times steps {200, 250, 300}), each fine-tuned on seeds 301..312 on both tasks: 576
fine-tunes. Seeds 301..308 serve the loops; 309..312 are held out and used only to measure true gain.

A loop starts at (1.0, 0, 250) and runs 10 rounds. Each round proposes a configuration that differs from the current
one in exactly one setting, chosen uniformly at random.

- Greedy: one run per configuration; moves if the proposal's score is higher. Claimed gain is the last accepted
  score minus the start's score.
- Refereed: compares proposal and current configuration on the same loop seeds, adding one at a time from 3 to 8,
  and moves if the 95% t interval of the paired differences is above zero (stops adding seeds once the interval
  excludes zero either way). Claimed gain is the sum of the mean differences of the moves made.
- True gain, for both: accuracy of the final configuration minus the start's, averaged over the 4 held-out seeds.

Replay: 2000 loops of each kind per task from a fixed generator. A third loop, the referee with 5 seeds fixed, is
reported only. The fine-tunes took about 14.2 h (topic) and 16.0 h (NLI) of wall clock with both tasks sharing one
A10G.

## Result

Means over 2000 loops. "Fine-tunes" counts distinct fine-tuned runs a loop used; "ends worse" is the share of loops
whose true gain is below zero.

| Task | Loop | Claimed | True | Fine-tunes | Moves | Ends worse |
|---|---|---|---|---|---|---|
| Topic | greedy | +0.0051 | -0.0014 | 6.8 | 1.5 | 72% |
| Topic | refereed | +0.0016 | -0.0003 | 49.4 | 0.6 | 37% |
| Topic | refereed, 5 seeds (reported only) | +0.0005 | -0.0001 | 40.4 | 0.2 | 13% |
| NLI | greedy | +0.0523 | +0.0209 | 7.3 | 2.5 | 8% |
| NLI | refereed | +0.0384 | +0.0205 | 51.7 | 1.1 | 0% |
| NLI | refereed, 5 seeds (reported only) | +0.0196 | +0.0133 | 42.0 | 0.6 | 0% |

- H1, on at least one task the greedy loop delivers at most half of the gain it claims: SUPPORTED, on both tasks.
  On topic it delivers a small loss against a claimed +0.0051; on NLI 40% of the claim (+0.0209 of +0.0523).
- H2, on both tasks the refereed loop's claim is within 0.003 of its true gain: NOT SUPPORTED. Topic is within
  tolerance (0.0019 apart). On NLI the refereed claim is +0.0384 against a delivered +0.0205, a gap of 0.0179.
- H3, on both tasks the refereed loop's true gain is at least the greedy loop's minus 0.002: SUPPORTED (topic
  -0.0003 against -0.0014; NLI +0.0205 against +0.0209).

Reported only:

- The referee made 0.6 moves per loop on topic and 1.1 on NLI, against 1.5 and 2.5 for greedy, and used about 7
  times the fine-tunes.
- Where there is nothing to find (topic), the greedy loop ended worse than it started in 72% of loops, the referee
  in 37%.
- Fixing the referee at 5 seeds brought its claim closer to its true gain on NLI (+0.0196 against +0.0133, a gap of
  0.0063) but gave up a third of the real gain (+0.0133 against +0.0209).

## What it shows

1. The keep-if-better loop overstates what it delivers, as predicted. On the task with no real effect it claims a
   gain and delivers a small loss; on the task with a real effect it claims 2.5 times what it delivers. The plan
   expected H1 only on topic; it also held on NLI.
2. The referee removes most of the false progress where there is none (topic claim +0.0016, true -0.0003) and does
   not cost the real gain (H3), at about 7 times the runs.
3. The referee's claim is still too high where there is a real effect (NLI, 1.9 times the delivered gain), so
   "the referee reports what it delivers" is not shown. The plan names re-testing after every seed with ordinary t
   intervals as a likely cause, so that its false-accept rate is above 5%. A second plausible cause is that the
   claimed gain is a sum of per-move estimates, each selected for having a large interval-clearing difference.
   Neither was tested here.

## Conclusion

A referee in front of "accept" keeps a tuning loop from making false progress and does not cost it real progress
on these two tasks, but it does not make the loop's claimed gain trustworthy: on NLI the claim stays nearly twice
the delivered gain. Two of three pre-registered claims hold.

Limits: the proposer is a random walk over a 24-point grid, not an agent, and this tests honest noise only, not
fabrication or evaluator tampering; the referee is a Python statistical rule, not the Regora harness, and it
re-tests after every seed; true gain rests on 4 held-out seeds per configuration (stable averaged over 2000 loops,
noisy for one loop); the 2000 loops reuse the same 8 loop seeds and are not independent; one model, two tasks; the
grid and both tasks overlap with E12 and E14 on other seeds; the referee costs several times the fine-tunes, and
H3 compares outcomes, not efficiency. No rule or threshold was changed after the pilot, and the replay was run once.
