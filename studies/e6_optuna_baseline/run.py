"""E6: does Regora's selector beat a conventional optimiser (Optuna) at the same compute budget?

PRE-REGISTERED before the E3 data existed. This is the spec's success criterion: a measurable advantage over at least
one meaningful baseline.

Data: the E3 table (results/e3/e3.json): 9 configs, 8 seeds, 1000 steps. No new training runs.

Method (replay)
- For each 3-seed subset (56 subsets) an optimiser sees the mean loss curve of those seeds for whichever config it asks
  for, one evaluation step at a time. Regret is measured on the 5 held-out seeds, as in E3.
- Regora: `regora replay` with the default ladder (rungs 250, 500, 750 keeping 4, 2, 1). Both of its selectors are
  reported: schedule-aware (guard 200) and fixed halving.
- Optuna, run sequentially, each study given a step budget equal to the schedule-aware selector's mean compute:
    tpe-hyperband : TPESampler + HyperbandPruner(min_resource=250, max_resource=1000, reduction_factor=2)
    random-median : RandomSampler + MedianPruner(n_startup_trials=2, n_warmup_steps=250)
    grid-sha      : GridSampler (each config once, shuffled) + SuccessiveHalvingPruner(min_resource=250, reduction_factor=2)
  3 sampler seeds per subset. A trial costs the steps it ran before pruning. When the budget runs out the running trial
  is cut off. Optuna's answer is the best completed trial; if no trial completed, the trial with the lowest last value.
  Pruner settings were chosen to match the ladder's first rung and were not tuned on any data.

Decision rule
- ADVANTAGE     if schedule-aware regret is lower than every Optuna variant's by at least 0.005;
- NO ADVANTAGE  if some Optuna variant's regret is lower than or within 0.005 of schedule-aware's.
  Fixed halving is reported the same way but is not part of the rule.

Known limits: Optuna is designed for large or continuous search spaces and parallel trials; a 9-point grid run
sequentially is not its best case. Subsets overlap. One model, one dataset. A pass here is a necessary first step, not
proof of general superiority.
"""

import itertools
import json
import math
import re
import subprocess
import sys

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

path = sys.argv[1] if len(sys.argv) > 1 else "results/e3/e3.json"
SAMPLER_SEEDS = (0, 1, 2)
MARGIN = 0.005

data = json.load(open(path))
config, curves = data["config"], data["curves"]
learning_rates, warmups, seeds, total_steps = (
    config["lrs"],
    config["warms"],
    config["seeds"],
    config["steps"],
)
configs = list(itertools.product(learning_rates, warmups))
eval_steps = [step for step, _ in curves[f"{configs[0][0]}|{configs[0][1]}|{seeds[0]}"]]
step_size = eval_steps[0]
loss = np.array(
    [
        [
            [v if math.isfinite(v) else 1e9 for _, v in curves[f"{lr}|{warm}|{s}"]]
            for s in seeds
        ]
        for lr, warm in configs
    ]
)


def regora_replay():
    """Run the Regora CLI and return {selector name: (compute, regret, hit rate)}."""
    out = subprocess.run(
        ["../target/debug/regora", "replay", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = re.findall(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", out, re.M)
    return {name: (float(c), float(r), float(h)) for name, c, r, h in rows}


def make_study(variant, sampler_seed):
    if variant == "tpe-hyperband":
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=250, max_resource=total_steps, reduction_factor=2
        )
    elif variant == "random-median":
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=250)
    else:
        sampler = optuna.samplers.GridSampler(
            {"lr": learning_rates, "warm": warmups}, seed=sampler_seed
        )
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=250, reduction_factor=2
        )
    return optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)


def run_optuna(variant, subset, sampler_seed, budget_steps):
    """Return (index of the chosen config, steps used)."""
    study = make_study(variant, sampler_seed)
    used = 0
    last_value = {}  # trial number -> (config index, last reported loss)

    def objective(trial):
        nonlocal used
        index = configs.index(
            (
                trial.suggest_categorical("lr", learning_rates),
                trial.suggest_categorical("warm", warmups),
            )
        )
        mean_curve = loss[index][subset].mean(0)
        for step, value in zip(eval_steps, mean_curve):
            if used + step_size > budget_steps:
                study.stop()
                raise optuna.TrialPruned()
            used += step_size
            last_value[trial.number] = (index, float(value))
            trial.report(float(value), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(mean_curve[-1])

    study.optimize(objective, n_trials=500)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = min(completed, key=lambda t: t.value)
        return last_value[best.number][0], used
    return min(last_value.values(), key=lambda pair: pair[1])[0], used


regora = regora_replay()
aware_name = next(name for name in regora if name.startswith("schedule-aware"))
budget_steps = int(regora[aware_name][0] * len(configs) * total_steps)

subsets = [list(c) for c in itertools.combinations(range(len(seeds)), 3)]
print(
    f"{path}: {len(configs)} configs, {len(seeds)} seeds, {len(subsets)} subsets; Optuna budget {budget_steps} steps"
)
print(f"\n{'method':<28}{'compute':>9}{'regret':>9}{'P(best)':>9}")
for name in (aware_name, "fixed-halving"):
    print(
        f"regora {name:<21}{regora[name][0]:>9.3f}{regora[name][1]:>9.4f}{regora[name][2]:>9.2f}"
    )

optuna_regret = {}
for variant in ("tpe-hyperband", "random-median", "grid-sha"):
    computes, regrets, hits = [], [], []
    for subset, sampler_seed in itertools.product(subsets, SAMPLER_SEEDS):
        heldout = [j for j in range(len(seeds)) if j not in subset]
        truth = loss[:, heldout, -1].mean(1)
        pick, used = run_optuna(variant, subset, sampler_seed, budget_steps)
        computes.append(used / (len(configs) * total_steps))
        regrets.append(truth[pick] - truth.min())
        hits.append(pick == int(truth.argmin()))
    optuna_regret[variant] = float(np.mean(regrets))
    print(
        f"optuna {variant:<21}{np.mean(computes):>9.3f}{np.mean(regrets):>9.4f}{np.mean(hits):>9.2f}"
    )

aware_regret = regora[aware_name][1]
advantage = all(regret - aware_regret >= MARGIN for regret in optuna_regret.values())
print(
    f"\nschedule-aware vs Optuna: {'ADVANTAGE' if advantage else 'NO ADVANTAGE'} by the pre-registered rule"
)
