"""Post hoc, not pre-registered: what a data decision costs in stage G, in runs.

G1 compares how much pairing saves inside each design. A user cares about the total: runs needed to tell a data
change from the control. For every signal arm and every short reading this prints the runs per arm for 80% power
at two-sided 5% (`analyze.stats`, E0's formula, floor of 2), unpaired and paired, and whether the reading's mean
effect points the same way as the truth (the long continuation at step 1600, all pairs).

    python posthoc_g.py > results/posthoc_G.txt
"""

import analyze
from stage_g import NAME, SIGNAL_ARMS, geomean, table

SHORT = [
    ("anneal", 100),
    ("anneal", 200),
    ("anneal", 400),
    ("long", 100),
    ("long", 200),
    ("long", 400),
]


def label(design, reading):
    return f"{'anneal' if design == 'anneal' else 'const'}-{reading}"


def main():
    d = analyze.load(NAME)
    truth_reading = ("long", d["config"]["long_reads"][-1])
    print(
        "Post hoc. Runs per arm for 80% power (unpaired / paired); '!' marks a reading whose mean effect points "
        "against the truth.\n"
    )
    print(f"{'arm':8} {'truth':>9}  " + " ".join(f"{label(*dz):>13}" for dz in SHORT))
    cost = {dz: ([], []) for dz in SHORT}
    for a in SIGNAL_ARMS:
        truth = analyze.stats(
            table(d, "control", *truth_reading), table(d, a, *truth_reading)
        )
        cells = []
        for dz in SHORT:
            s = analyze.stats(table(d, "control", *dz), table(d, a, *dz))
            flip = "!" if (s["effect"] > 0) != (truth["effect"] > 0) else " "
            cells.append(f"{s['n_unpaired']:6.0f}/{s['n_paired']:4.0f}{flip}")
            cost[dz][0].append(s["n_unpaired"])
            cost[dz][1].append(s["n_paired"])
        print(f"{a:8} {truth['effect']:+.5f}  " + " ".join(f"{c:>13}" for c in cells))
    print(
        f"\n{'geomean':18} "
        + " ".join(
            f"{geomean(cost[dz][0]):6.1f}/{geomean(cost[dz][1]):4.1f} " for dz in SHORT
        )
    )


if __name__ == "__main__":
    main()
