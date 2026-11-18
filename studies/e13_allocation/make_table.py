"""E13: turn the E12 results into the tables `regora allocate` replays. The plan is in PLAN.md.

Usage: python -m studies.e13_allocation.make_table
"""

import json

from studies.e12_paired_eval.run import ARMS, TASKS

OUT = "results/e13/tables.json"


def table(name, prefix="results/e12/e12", arms=ARMS):
    runs = json.load(open(f"{prefix}_{name}.json"))["runs"]
    acc = {
        (r["arm"], r["seed"]): r["correct"].count("1") / len(r["correct"]) for r in runs
    }
    seeds = sorted({r["seed"] for r in runs})
    changes = [arm for arm in arms if arm != "control"]
    return dict(
        name=name,
        changes=changes,
        diffs=[[acc[arm, s] - acc["control", s] for s in seeds] for arm in changes],
    )


if __name__ == "__main__":
    json.dump([table(name) for name in TASKS], open(OUT, "w"))
    print(f"wrote {OUT}")
