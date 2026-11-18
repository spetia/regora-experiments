"""E7c: does diagnosis by intervention hold on a larger model, and do the learned likelihoods transfer to it?

PRE-REGISTERED before the full run, after a one-seed pilot (seed 2, not used below) that only checked timing and that
no condition crashed. No setting was changed after the pilot. E7b (fresh seeds, small model) was still running when
this was written, so its outcome did not shape this plan.

Data
- Same conditions, probes, costs and code as E7, on a model with 6 layers and width 256 (capacity condition: width
  64; the wider probe doubles each). Learning rates are NOT retuned for the larger model: "healthy" is still 4e-3,
  and in the pilot doubling it already hurt. Tesla T4, CUDA, 20 fresh seeds 51..70.

Evaluation (analyze.py), using the E7b analyzer unchanged (Student-t likelihoods, passive readings first, stop at 0.9)
- WITHIN: fit on seeds 51..60, test on the 50 cases of seeds 61..70.
- TRANSFER: fit on the 50 E7 cases (small model, MPS), test on the same 50 cases, probes only. The passive readings
  are left out because the absolute metric level is model-specific.

Decision rules
- H1 to H4 exactly as in E7b, applied to WITHIN.
- H5 transfer: TRANSFER's ALL policy is right on at least 40 of the 50 test cases. This is the claim that matters for
  a product, since a new customer's model has no labelled failures to fit on.

Known limits: still one architecture family, one dataset, single injected causes. A 5.5x larger model is not "scale".
The corpus is the Python standard library of the local interpreter, so the Linux corpus differs slightly from E7's.
"""

import sys

import studies.e7_diagnosis.run as e7

e7.N_LAYER = 6
e7.FAULTS = {
    name: cfg | dict(width=2 * cfg["width"]) for name, cfg in e7.FAULTS.items()
}

if __name__ == "__main__":
    if sys.argv[1:] == ["pilot"]:
        e7.main((2,), "e7c_pilot")
    else:
        e7.main(range(51, 71), "e7c")
