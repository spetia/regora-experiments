"""Analyse E3 per the rules in run.py.

Usage: python -m studies.e3_schedule_aware.analyze [results.json] [seeds-per-selector]
"""

import itertools
import json
import math
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/e3/e3.json"
seeds_per_selector = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MARGIN = 0.005
COMPUTE_SLACK = 0.02
WARMUP_GUARD = 200
TARGETS = (4, 2, 1)

data = json.load(open(path))
config, curves = data["config"], data["curves"]
configs = list(itertools.product(config["lrs"], config["warms"]))
seeds = list(config["seeds"])
total_steps = config["steps"]
n_configs = len(configs)


def curve(cfg, seed):
    return curves[f"{cfg[0]}|{cfg[1]}|{seed}"]


eval_steps = [step for step, _ in curve(configs[0], seeds[0])]
# loss[config, seed, eval index]; a diverged run counts as a very large finite loss
loss = np.array(
    [
        [[v if math.isfinite(v) else 1e9 for _, v in curve(cfg, s)] for s in seeds]
        for cfg in configs
    ]
)
at = {step: eval_steps.index(step) for step in (250, 500, 750, 1000)}


def mean_at(i, subset, step):
    return loss[i, subset, at[step]].mean()


def fixed(subset, rungs):
    alive, used, prev = list(range(n_configs)), 0, 0
    for rung, keep in zip(rungs, TARGETS):
        used += len(alive) * (rung - prev)
        prev = rung
        alive = sorted(alive, key=lambda i: mean_at(i, subset, rung))[:keep]
    used += len(alive) * (total_steps - prev)
    return alive[0], used / (n_configs * total_steps)


def aware(subset):
    alive, used, prev = list(range(n_configs)), 0, 0
    for rung, keep in zip((250, 500, 750), TARGETS):
        used += len(alive) * (rung - prev)
        prev = rung
        protected = [i for i in alive if rung < configs[i][1] + WARMUP_GUARD]
        cuttable = sorted(
            (i for i in alive if i not in protected),
            key=lambda i: mean_at(i, subset, rung),
        )
        alive = protected + cuttable[: max(1, keep - len(protected))]
    used += len(alive) * (total_steps - prev)
    pick = min(alive, key=lambda i: mean_at(i, subset, 1000))
    return pick, used / (n_configs * total_steps)


def full(subset):
    return min(range(n_configs), key=lambda i: mean_at(i, subset, 1000)), 1.0


selectors = {
    f"FIXED{rungs}": (lambda subset, rungs=rungs: fixed(subset, rungs))
    for rungs in itertools.combinations((250, 500, 750, 1000), 3)
}
selectors["FULL"] = full
selectors["AWARE"] = aware

subsets = [
    list(c) for c in itertools.combinations(range(len(seeds)), seeds_per_selector)
]
results = {}
for name, select in selectors.items():
    computes, regrets, hits = [], [], []
    for subset in subsets:
        heldout = [j for j in range(len(seeds)) if j not in subset]
        truth = loss[:, heldout, -1].mean(1)
        pick, compute = select(subset)
        computes.append(compute)
        regrets.append(truth[pick] - truth.min())
        hits.append(pick == int(truth.argmin()))
    results[name] = (np.mean(computes), np.mean(regrets), np.mean(hits))

all_seed_final = loss[:, :, -1].mean(1)
print(
    f"{path}: {n_configs} configs, {len(seeds)} seeds, {len(subsets)} subsets of {seeds_per_selector}"
)
print("all-seed mean final:", dict(zip(map(str, configs), np.round(all_seed_final, 4))))
print("seed SD of final:", np.round(loss[:, :, -1].std(1, ddof=1), 4))
print(
    f"random-pick regret (all seeds): {all_seed_final.mean() - all_seed_final.min():.4f}"
)
crossings = sum(
    any(
        (loss[a, :, j].mean() < loss[b, :, j].mean())
        != (all_seed_final[a] < all_seed_final[b])
        for j in range(len(eval_steps) - 1)
    )
    for a, b in itertools.combinations(range(n_configs), 2)
)
print(f"crossing pairs: {crossings} of {n_configs * (n_configs - 1) // 2}")

print(f"\n{'selector':<24}{'compute':>9}{'regret':>9}{'P(best)':>9}")
for name, (compute, regret, hit) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"{name:<24}{compute:>9.3f}{regret:>9.4f}{hit:>9.2f}")

late, early = results["FIXED(250, 750, 1000)"][1], results["FIXED(250, 500, 750)"][1]
print(
    f"\nH1: late-cut regret {late:.4f} vs early-cut {early:.4f} -> {'CONFIRMED' if early - late >= MARGIN else 'NOT confirmed'}"
)

aware_compute, aware_regret, _ = results["AWARE"]
comparable = {
    name: regret
    for name, (compute, regret, _) in results.items()
    if name != "AWARE" and compute <= aware_compute + COMPUTE_SLACK
}
if all(regret - aware_regret >= MARGIN for regret in comparable.values()):
    verdict = "SUPPORTED"
elif any(aware_regret - regret >= MARGIN for regret in comparable.values()):
    verdict = "WORSE"
else:
    verdict = "EQUIVALENT"
print(f"H2: AWARE vs comparable fixed schedules {sorted(comparable)} -> {verdict}")
