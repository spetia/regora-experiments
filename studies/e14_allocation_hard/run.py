"""E14 data collection: eight candidate changes per task, on seeds nobody has looked at. The plan is in PLAN.md.

The fine-tuning and scoring are E12's, unchanged; only the arms, the seeds and the output differ. The control LR is
the one E12 tuned, so nothing is tuned here.

Usage (with REGORA_DEV=cuda REGORA_EXACT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8):
       python -m studies.e14_allocation_hard.run pilot TASK
       python -m studies.e14_allocation_hard.run run TASK     (a killed run resumes from its partial file)
       python -m studies.e14_allocation_hard.run table        (writes the tables `regora allocate` replays)
       python -m studies.e14_allocation_hard.run quo          (the status quo's picks, one per seed)
"""

import json
import os
import sys

ARMS = {
    "control": dict(),
    "lr_x0.5": dict(lr_mult=0.5),
    "lr_x0.71": dict(lr_mult=0.71),
    "lr_x1.41": dict(lr_mult=1.41),
    "smoothing_0.05": dict(smoothing=0.05),
    "smoothing_0.1": dict(smoothing=0.1),
    "steps_200": dict(steps=200),
    "steps_300": dict(steps=300),
    "prompt": dict(prompt=1),
}
CONTROL_LR = {
    "topic": 3.2e-4,
    "nli": 6.4e-4,
}  # from E12's tuning (results/e12/tune_*.json)
PILOT_SEEDS, TEST_SEEDS = (12,), range(201, 217)
OUT = "results/e14"


def status_quo(diffs):
    """One seed per decision: the index of the change with the highest difference on that seed, or None if no
    change beats the control there. Returns one pick per seed."""
    picks = []
    for seed in range(len(diffs[0])):
        best = max(range(len(diffs)), key=lambda c: diffs[c][seed])
        picks.append(best if diffs[best][seed] > 0 else None)
    return picks


def report_status_quo(t):
    means = [sum(d) / len(d) for d in t["diffs"]]
    picks = status_quo(t["diffs"])
    false_wins = sum(p is not None and means[p] <= 0 for p in picks)
    print(
        f"{t['name']}: over {len(picks)} seeds the status quo adopts a change that does not help in "
        f"{false_wins}, keeps the control in {picks.count(None)}; picks: "
        + ", ".join("control" if p is None else t["changes"][p] for p in picks)
    )


if __name__ == "__main__":
    # two changes, three seeds: seed 0 favours the bad change, seed 1 the good one, seed 2 neither
    assert status_quo([[0.2, -0.1, -0.1], [0.1, 0.3, -0.2]]) == [0, 1, None]

    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:] == ["table"]:
        from studies.e13_allocation.make_table import table

        tables = [table(name, f"{OUT}/e14", ARMS) for name in CONTROL_LR]
        json.dump(tables, open(f"{OUT}/tables.json", "w"))
    elif sys.argv[1:] == ["quo"]:
        for t in json.load(open(f"{OUT}/tables.json")):
            report_status_quo(t)
    else:
        import scale
        from studies.e12_paired_eval import run as e12

        stage, name = sys.argv[1:]
        assert scale.EXACT or scale.DEV != "cuda", "run with REGORA_EXACT=1"
        # E12's collector reads its arms and output directory from its module
        e12.ARMS, e12.OUT = ARMS, OUT
        seeds = dict(pilot=PILOT_SEEDS, run=TEST_SEEDS)[stage]
        prefix = dict(pilot="pilot", run="e14")[stage]
        e12.collect(name, dict(lr=CONTROL_LR[name]), seeds, f"{prefix}_{name}")
