# Literature check: what is known, what undercuts Regora, where the opening is

Date: 2026-11-15. Not pre-registered work; a reading note.

**Method and its limits.** OpenAlex API: 90 searches (200 best matches each) across seed noise, trajectory
divergence, paired and cheap evaluation, sequential testing, hyperparameter search, research agents, model
updates, weight editing and data attribution, plus the works citing 8 key papers. 16,103 unique records, 15,175
with abstracts, ranked by keyword match, citations and recency. About 500 titles were scanned and about 45
abstracts read. **No full text was read**, so every number below is the authors' claim as stated in an abstract.
The weight-space-programming post was read through an automated summary of the page. A search like this finds
what its queries ask for; it cannot show that something does not exist.

## 1. Our stage A to C findings are already in the literature

- **Small perturbations make training runs diverge, mostly early in training.** "The Butterfly Effect: Neural
  Network Training Trajectories Are Highly Sensitive to Initial Conditions" (2025, arXiv 2506.13234): in an
  initial chaotic phase even extremely small perturbations make identical runs diverge, "an effect that
  diminishes rapidly over training time". This is what `C_age150` against `C_age2400` shows, and what D4 tests.
- **Any source of randomness has about the same effect.** "Nondeterminism and Instability in Neural Network
  Optimization" (2021, arXiv 2103.04514): all sources of nondeterminism have similar effects on model
  diversity, and one-bit changes in initial parameters lead to very different models.
- **Later checkpoints are stable.** "Linear Mode Connectivity and the Lottery Ticket Hypothesis" (2019) and
  "Scaling Linear Mode Connectivity and Merging to Billion Parameter Pretrained Transformers" (2026, arXiv
  2606.23607). "First-Order Predictable but Pairwise Fragile" (2026, arXiv 2607.16821): around a trained
  checkpoint, single small perturbations are first-order predictable up to scale 1e-2 on nine transformers from
  82M to 7B, with a registered protocol.

Consequence: the fading of seed pairing is not a discovery to sell. The same literature predicts pairing should
hold well when branching from a trained model, which is the fine-tuning case, for changes that are small. E12
found about 1x there, but only tried large changes. That is still untested.

## 2. What undercuts parts of the current plan

- **The statistics are public and packaged.** "Adding Error Bars to Evals" (2024, arXiv 2411.00640, 225
  citations) sets out paired differences, clustered standard errors and power for model evaluations. "evalci"
  (2026, arXiv 2607.04429) is a pip-installable library doing paired tests, intervals, power and multiple-
  comparison correction from a per-item results table. "Statistical Methods for Multiple Language Model
  Comparison on a Shared Evaluation" (2026, arXiv 2608.22659) extends it to many models. Item pairing as a
  method is therefore not a moat.
- **Cheaper evaluation is a crowded research area.** tinyBenchmarks (2024), "Efficient Evaluation of LLM
  Performance with Statistical Guarantees" (2026, up to 5x fewer queries), prediction-powered inference
  (Science 2023) and its GLIDE library (2026), anytime-valid stopping (AV-AIVAT, 2026).
- **Choosing the next experiment is hard to make pay.** Hyperband and ASHA remain the standard. "When Is an LLM
  Worth It for Hyperparameter Optimization?" (2026, arXiv 2606.21641) finds, budget-matched with paired tests,
  that an LLM advisor adds +0.40 points over a default configuration and nothing on held-out data. This agrees
  with E13 and E14: the selection engine is unsupported.
- **Someone has already done a paired-seed audit at scale.** "Auditing PEFT Claims" (AACL 2026): published
  improvements over LoRA audited with paired seeds, bootstrap intervals, sign-flip tests, Holm correction and
  precommitted rules across 110M to 7B models; 2 of 15 confirmatory claims survive and 4 reverse. This shows
  the method produces results people care about at real scale, and that it is not unique to us.

## 3. Where the opening is: nobody referees the experiments AI agents run

Agents that run ML research end to end are now mainstream ("Towards end-to-end automation of AI research",
Nature 2026; AI Scientist v2; MLE-bench; PostTrainBench; "Recursive self-improvement of AI research agents",
2026, whose loop "keeps the changes that perform best on hidden evaluations"). The 2026 literature on them says
the same thing from several directions:

| Paper | Claim in the abstract |
|---|---|
| MLR-Bench (NeurIPS 2025) | coding agents produce fabricated or invalidated experimental results "in 80% of the cases" |
| MLReplicate (2026, arXiv 2605.16616) | human reviewers found hallucinated results and reproducibility failures in all six systems; 59% of the automated reviews that accepted a paper contained fabricated or unsupported claims |
| "The more you automate, the less you see" (PNAS 2026) | data leakage, metric misuse and post hoc selection bias in open-source AI scientists, "largely invisible at the level of the final manuscript"; calls for auditing the workflow artifacts |
| "Reward Hacking Challenges Oversight of Autonomous Research Agents" (2026, arXiv 2609.28614) | spontaneous reward hacking in 30.5% of open-ended research-pipeline tasks across 17 models; an LLM review panel misses 6.5% of confirmed hacks, and evasion grows with feedback |
| "Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap" (2026, arXiv 2608.05179) | of 24 runnable systems 38% release seeds or traces; "no LLM-era system in the corpus demonstrates an externally validated in-loop oracle" |
| "Workflow Closure Is Not Scientific Closure" (2026, arXiv 2605.26200) | audit of 21 systems: internal self-evaluation replaces independent validation; asks for "autonomous execution under non-autonomous epistemic control" |
| "AI scientists produce results without reasoning scientifically" (2026, arXiv 2604.18805) | 25,000 agent runs: evidence ignored in 68% of traces |
| "On Randomness in Agentic Evals" (2026, arXiv 2602.07150) | 60,000 trajectories: single-run scores vary by 2.2 to 6.0 points, so reported gains of 2 to 3 points may be noise |

The weight-space-programming post is the same pattern without a training loop: its Go procedure "keeps a change
only if it wins more", its headline table has no intervals, and the authors report more than a hundred
follow-up efforts. More ways to produce candidate changes, no referee.

What Regora already is fits that gap closely: the harness, not the proposer, runs the experiment and records it;
the proposer cannot approve its own experiment; interventions are typed, so no generated code runs; a verdict
can be "unresolved"; there is an audit log. That is an external in-loop oracle for one narrow kind of claim
("this training change helped"). The product sentence would be: **the referee for experiments that agents
run**, with a person's own experiments as the same product at lower volume.

## 4. Reasons this could still be wrong

- No customer has been asked. Labs that run agent fleets may build this in-house; "Toward Auditable AI
  Scientists" (2026) and FARS (2026) already keep auditable artifacts.
- The failures above are mostly fabrication, leakage and reward hacking, not seed noise. A referee has to own
  the held-out data and the evaluation, which the harness does only in part today. Typed interventions prevent
  hacking and also exclude the arbitrary code changes agents want to make.
- Seed pairing, Regora's original mechanism, is the weakest part of the evidence (stages A to C, E12).
- Everything here is abstract-level reading.

## 5. What would test it

1. **The gate test (laptop).** A scripted proposer hill-climbs on the small GPT for a fixed number of rounds,
   two ways: keep any change that scores better on one run (what agent loops do), or keep only what a
   replicated paired comparison calls real. Score both by the true gain of the final configuration on fresh
   seeds against the gain each loop claims. It measures how much "progress" is noise and what the gate costs.
2. **Fine-tuning dose-response (cloud).** Small changes on E12's fine-tunes with shared seeds: the literature
   in section 1 predicts pairing holds there for small changes; E12 only tried large ones.
3. **An audit of a public claim.** The edited Qwen3.5-0.8B from the post is public; a paired evaluation with
   intervals and the share of answers that finish is the experiment the cofounder's note already describes.
4. **Ten conversations** with people who run agent fleets or large sweeps. No experiment replaces this.

## 6. Addendum, afternoon of 2026-11-15

Prompted by a critique: "Free tools already do the same statistics (seqbench, evalci). Run forking already ships
in W&B, Trainy and OpenPipe ART." Checked against the tools' own pages:

- **seqbench** (R package on CRAN, `castlaboratory/seqbench`): anytime-valid sequential comparison of two
  algorithms on paired losses with a practical-equivalence margin; it declares superiority, equivalence or
  "continue", treats seeds as clusters and accounts for the budget. That is the harness's comparison verdict,
  with stronger guarantees than its fixed-sample paired t interval.
- **Run forking**: W&B `fork_from` (a new run continues a parent's logged history from a step; restoring the
  training state is left to the user), Trainy Pluto forking (preview, the same idea), OpenPipe ART
  `_experimental_fork_checkpoint` (copies a checkpoint to start a new RL training run), Together AI training
  checkpoints. Branching mechanics are a commodity.
- **"Separating Real Improvements from Lucky Runs in Autonomous ML Research"** (NeurIPS 2026, Workshop for
  Autonomous Machine Learning Research; read from the poster page only): nanoGPT, 24 frozen candidates including
  9 exact nulls, 749 runs, seeds the agent cannot see. The best of 32 runs of one program shows an apparent gain
  of 0.00544 val_bpb, 2.7 times their meaningful-effect threshold, from selection alone; of 15 non-null
  candidates 8 replicated, 3 failed and 4 were inconclusive; small gains near the threshold stayed unresolved
  after 24 to 33 sealed evaluations. Stage F asks the same question with a scripted proposer, so it is now a
  replication, not a new result. Related: "Phantom Gains" (2026); "Auditing the Product, Not the Recursion"
  (2026: of 90 studies describing recursive self-improvement, 22% audit two or more reruns of the loop).

What this changes: the statistics, the branching and the referee idea are all available or published. What none
of them provides is how to design a comparison so that it is cheap and still answers the right question. Stage D
found one such relation: on a 4.9M model the learning rate during a branch changes the cost of a paired
comparison by two to three orders of magnitude. Theory and small-model work already tie chaos in training to the
learning rate ("Universal Sharpness Dynamics in Neural Network Training", 2023: a period-doubling route to chaos
as the learning rate increases; "Dynamical stability and chaos in artificial neural network trajectories along
training", 2024). Not found in the abstracts read: using it to set the cost of experiments, or measurements of
how cheap and how valid low-learning-rate comparisons are for real decisions. Labs make data decisions in
low-learning-rate micro-anneals (Llama 3; OLMo 2; Olmo 3); DataDecide (2025) and Signal and Noise (2025) measure
the decision accuracy of small experiments without shared-order designs. Stage G (`PLAN_G.md`) tests it.

## 7. Addendum, evening of 2026-11-15 (after stage G)

- **"Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice"** (Wang, Wu, Lyu,
  Zou, Song, Jia, Mittal; ICLR 2026; arXiv 2512.24503; read from the arXiv HTML, first half). With 23 data
  recipes, proxies of 70M to 125M parameters trained at the standard learning rate rank them against tuned 1B
  targets with Spearman below 0.75. At a learning rate 1 to 2 orders of magnitude smaller the correlation is
  above 0.92. Their theory, for random-feature models, has the deterministic drift dominate the variance at small
  learning rates. Three seeds per recipe, with standard errors and bootstrap intervals; the half that was read
  says nothing about pairing runs across recipes. This is the lever stage D found (the learning rate during a
  comparison) applied to validity across scale rather than to the cost of a paired comparison. A
  low-learning-rate data test may be both cheaper and more transferable. Neither study shows both at once.
- **"Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting"** (LARE; arXiv
  2610.32141, October 2026; abstract only). Re-weighting each domain's loss with its data exposure held fixed acts
  as a control that mixture ratios do not capture. So re-weighting is not a free stand-in for re-sampling. Stage
  H tests whether it is a good enough stand-in for a cheap paired comparison. Also: DoReMi trains its proxy with
  re-weighted losses; "Dynamic Gradient Alignment" (2024) argues that re-weighting to simulate re-sampling adds
  variance to its domain-weight estimates.
- Searched (the harvest, and the web twice) and not found: comparing two data mixtures on shared batches through
  loss weights, or common random numbers in data-mixture ablations. A fixed data order across ablation runs is
  standard practice (for example the Reflection Beam technical report, seen only as a search-result excerpt), as
  a control rather than as a measured saving.
