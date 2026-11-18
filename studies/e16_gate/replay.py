"""E16 replay: tuning loops over the grid, with and without a referee. Decision rules are in PLAN.md.

Usage: python -m studies.e16_gate.replay [selftest | results.json ...]
"""

import json
import random
import statistics
import sys

from studies.e9_longer_probes.analyze import supported
from studies.e16_gate.run import GRID, SEARCH_SEEDS, START, TRUTH_SEEDS, name_of

ROUNDS, LOOPS = 10, 2000
MIN_SEEDS, MAX_SEEDS = 3, 8
T95 = {
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
}  # two-sided, by degrees of freedom
MAX_KEPT, CLAIM_TOLERANCE, MAX_LOSS = 0.5, 0.003, 0.002


def neighbours(config):
    """Configurations that differ from `config` in exactly one setting."""
    return [c for c in GRID if sum(a != b for a, b in zip(c, config)) == 1]


def interval(diffs):
    mean = statistics.mean(diffs)
    half = T95[len(diffs) - 1] * statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, half


def loop(acc, draws, referee, search, truth):
    """One tuning loop. `draws` holds, per round, a number in [0, 1) that picks the proposal and a generator for
    seeds; `referee` is None for the greedy loop or (min seeds, max seeds). Returns (claimed, true, runs, moves).
    """
    current, claimed, moves, used, single = START, 0.0, 0, set(), {}

    def one_run(config, rng):
        # the greedy loop knows each configuration by a single run
        if config not in single:
            single[config] = rng.choice(search)
            used.add((config, single[config]))
        return acc[config, single[config]]

    for pick, rng in draws:
        options = neighbours(current)
        proposal = options[int(pick * len(options))]
        if referee is None:
            gain = one_run(proposal, rng) - one_run(current, rng)
            accept = gain > 0
        else:
            low, high = referee
            diffs, accept = [], False
            for seed in rng.sample(search, high):
                used.update({(proposal, seed), (current, seed)})
                diffs.append(acc[proposal, seed] - acc[current, seed])
                if len(diffs) < low:
                    continue
                gain, half = interval(diffs)
                if abs(gain) > half:  # settled either way: stop adding seeds
                    accept = gain > 0
                    break
        if accept:
            current, claimed, moves = proposal, claimed + gain, moves + 1
    held_out = lambda c: statistics.mean(acc[c, s] for s in truth)
    return claimed, held_out(current) - held_out(START), len(used), moves


def replay(acc, search=tuple(SEARCH_SEEDS), truth=tuple(TRUTH_SEEDS), loops=LOOPS):
    """kind -> (mean claimed, mean true, mean runs, mean moves, share of loops ending worse than they started)."""
    kinds = dict(greedy=None, refereed=(MIN_SEEDS, MAX_SEEDS), refereed_5=(5, 5))
    out = {}
    for kind, referee in kinds.items():
        master = random.Random(16)  # every kind sees the same random draws
        results = []
        for _ in range(loops):
            draws = [
                (master.random(), random.Random(master.random())) for _ in range(ROUNDS)
            ]
            results.append(loop(acc, draws, referee, search, truth))
        claimed, true, runs, moves = (
            statistics.mean(r[i] for r in results) for i in range(4)
        )
        out[kind] = (claimed, true, runs, moves, sum(r[1] < 0 for r in results) / loops)
    return out


def report(tables):
    h1, h2, h3 = False, True, True
    for task, acc in tables.items():
        result = replay(acc)
        print(f"\n{task}")
        print("  loop         claimed     true    fine-tunes  moves  ends worse")
        for kind, (claimed, true, runs, moves, worse) in result.items():
            print(
                f"  {kind:11s} {claimed:+.4f}  {true:+.4f}  {runs:10.1f}  {moves:5.1f}  {worse:9.0%}"
            )
        greedy, refereed = result["greedy"], result["refereed"]
        h1 |= greedy[0] > 0 and greedy[1] <= MAX_KEPT * greedy[0]
        h2 &= abs(refereed[0] - refereed[1]) <= CLAIM_TOLERANCE
        h3 &= refereed[1] >= greedy[1] - MAX_LOSS
    print(
        f"\nH1 on at least one task the greedy loop delivers at most {MAX_KEPT:.0%} of the gain it claims: {supported(h1)}"
    )
    print(
        f"H2 on both tasks the refereed loop's claim is within {CLAIM_TOLERANCE} of its true gain: {supported(h2)}"
    )
    print(
        f"H3 on both tasks the refereed loop's true gain is at least the greedy loop's minus {MAX_LOSS}: {supported(h3)}"
    )
    return h1, h2, h3


def selftest():
    rng = random.Random(0)
    seeds = list(range(12))
    search, truth = tuple(seeds[:8]), tuple(seeds[8:])
    # no configuration is better than another: every gain a loop claims is noise
    flat = {(c, s): 0.9 + rng.gauss(0, 0.005) for c in GRID for s in seeds}
    result = replay(flat, search, truth, loops=300)
    assert result["greedy"][0] > 0.004 and abs(result["greedy"][1]) < 0.002, result[
        "greedy"
    ]
    assert (
        abs(result["refereed"][0]) < 0.002
        and result["refereed"][3] < result["greedy"][3]
    )
    # a lower LR multiple is clearly better: both loops should find gain, and the referee should report it truly
    slope = {
        (c, s): 0.9 - 0.03 * c[0] + rng.gauss(0, 0.002) for c in GRID for s in seeds
    }
    result = replay(slope, search, truth, loops=300)
    assert (
        result["refereed"][1] > 0.005
        and abs(result["refereed"][0] - result["refereed"][1]) < 0.003
    )
    assert len(neighbours(START)) == 3 + 1 + 2
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        paths = sys.argv[1:] or [
            f"results/e16/e16_{task}.json" for task in ("topic", "nli")
        ]
        tables = {}
        for path in paths:
            data = json.load(open(path))
            by_name = {name_of(c): c for c in GRID}
            tables[data["config"]["task"]] = {
                (by_name[r["arm"]], r["seed"]): r["correct"].count("1")
                / len(r["correct"])
                for r in data["runs"]
            }
        report(tables)
