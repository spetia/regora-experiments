"""E7c analysis: within-model diagnosis on the larger model, and transfer from the small model. Rules are in run.py."""

import json

from studies.e7b_confirmation.analyze import empty_rows, report, score

if __name__ == "__main__":
    small = json.load(open("results/e7/e7.json"))["cases"]
    large = json.load(open("results/e7/e7c.json"))["cases"]
    fit_cases = [c for c in large if c["seed"] <= 60]
    test_cases = [c for c in large if c["seed"] > 60]
    faults = sorted({c["fault"] for c in large})

    print(
        f"WITHIN: fitted on {len(fit_cases)} larger-model cases, tested on {len(test_cases)}.\n"
    )
    within = empty_rows()
    score(fit_cases, test_cases, within)
    report(within, faults)

    print(
        f"\n\nTRANSFER: fitted on {len(small)} small-model cases, probes only. Only H5 applies below.\n"
    )
    transfer = empty_rows()
    score(small, test_cases, transfer, passive=())
    report(transfer, faults)
    right = sum(truth == pick for truth, pick, *_ in transfer["ALL"])
    print(
        f"\nH5 transfer (ALL right on >= 40 of 50): {right} of {len(test_cases)}: {'SUPPORTED' if right >= 40 else 'NOT SUPPORTED'}"
    )
