"""E7b: does diagnosis with continuous likelihoods hold on fresh seeds?

PRE-REGISTERED before running. E7 failed its rule (34 of 50) with readings cut into three bins. A post hoc re-analysis
of the same E7 data, keeping the readings continuous, scored 46 of 50 with the four probes and 47 of 50 when the
passive readings (metric level and trend) were added. That analysis was chosen after seeing the data, and I tried
eight probe orders on it, so it proves nothing. E7b tests it once, out of sample.

Data
- Same conditions, probes, costs and code as E7 (studies/e7_diagnosis/run.py), on 10 fresh seeds 41..50. MPS.

Evaluation (analyze.py)
- The model is fitted on the 50 E7 cases only and then frozen. For each reading and condition: a Student-t predictive
  density with the sample mean, scale sd * sqrt(1 + 1/n) and n - 1 degrees of freedom (n = 10). Non-finite readings
  are replaced by the largest finite E7 value of that reading plus one.
- Each fresh case starts from a uniform prior. The two passive readings (reported metric at step 300, change since
  step 200) are free and always read first. Probes follow one at a time until the top posterior reaches 0.9 or none
  are left. Diagnosis = top posterior.
- Policies for the probe order: EIG_PER_COST (expected information gain per unit cost, the expectation taken over 25
  quantiles of each condition's predictive density), CHEAPEST, RANDOM (mean over all 24 orders), ALL (every probe, no
  early stop), PASSIVE (no probes).

Decision rules, all on the 50 fresh cases
- H1 diagnosis works: ALL is right on at least 40 of 50.
- H2 selection saves compute: EIG_PER_COST costs at most 0.8x RANDOM's mean cost with accuracy no more than 0.04
  below RANDOM's. STRONG if the same holds against CHEAPEST, otherwise WEAK.
- H3 honest confidence: among EIG_PER_COST stops with top posterior >= 0.9, at least 0.85 are right.
- H4 probes add to watching: ALL accuracy exceeds PASSIVE accuracy by at least 0.10.

Known limits: as E7. One small model, injected single causes, likelihoods learned on the same model family and
machine. PASSIVE uses the absolute metric level, which would not transfer to a different model or dataset.
"""

from studies.e7_diagnosis.run import main

if __name__ == "__main__":
    main(range(41, 51), "e7b")
