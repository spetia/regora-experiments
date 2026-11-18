"""E8 data collection. The plan and decision rules are in PLAN.md next to this file.

Usage: python -m studies.e8_label_free.run tune|pilot|run
"""

import json
import statistics
import sys
import time

import torch

import studies.e7_diagnosis.run as e7
import tinygpt as T

MODELS = {"tiny": (2, 96), "mid": (6, 256), "deep": (8, 192)}  # layers, width
PILOT_MODEL = {
    "base": (4, 128)
}  # the E7 model, used only to check the rule before the plan was fixed
CONDITIONS = {
    "healthy": dict(),
    "lr_high": dict(lr_mult=8.0),
    "lr_low": dict(lr_mult=1 / 12),
    "eval_noise": dict(noisy_eval=True),
    "capacity": dict(width_div=4),
    "mild_high": dict(lr_mult=3.0),
    "mild_low": dict(lr_mult=1 / 4),
}
TUNE_LRS, TUNE_SEEDS = (1e-3, 2e-3, 4e-3, 8e-3), (3, 4)
CHECKPOINT_STEP, PROBE_STEPS, REPLICATES, NOISE_READS = 300, 100, 3, 4
FIXED_EVAL_BATCHES = 20


def train(state, data_seed, steps, lr, lr_mult=1.0):
    return T.run(state, data_seed, steps, lr, lr_mult=lr_mult, eval_every=e7.NO_EVAL)[0]


def tune(models, lrs=TUNE_LRS):
    """Pick each model's healthy LR: lowest mean validation loss after 400 steps over TUNE_SEEDS."""
    chosen = {}
    for name, (layers, width) in models.items():
        e7.N_LAYER = layers
        e7.set_width(width)
        scores = {}
        for lr in lrs:
            finals = [
                e7.loss_on(
                    train(T.new_state(s), 3000 + s, CHECKPOINT_STEP + PROBE_STEPS, lr),
                    T.VAL_IDX,
                )
                for s in TUNE_SEEDS
            ]
            scores[lr] = statistics.mean(finals)
            print(f"tune {name} lr {lr}: {scores[lr]:.4f}", flush=True)
        chosen[name] = dict(
            lr=min(scores, key=scores.get),
            scores={str(k): v for k, v in scores.items()},
        )
    return chosen


def run_case(model, layers, width, healthy_lr, fault, seed):
    cfg = CONDITIONS[fault]
    lr = healthy_lr * cfg.get("lr_mult", 1.0)
    width //= cfg.get("width_div", 1)
    noisy = cfg.get("noisy_eval", False)
    eval_batches = 1 if noisy else FIXED_EVAL_BATCHES
    eval_rng = torch.Generator().manual_seed(7000 + seed)

    def fresh(batches):
        return torch.randint(
            0, len(T.VAL) - T.CTX - 1, (batches * T.BATCH,), generator=eval_rng
        )

    def reported(state):
        return e7.loss_on(state, fresh(1) if noisy else T.VAL_IDX)

    e7.N_LAYER = layers
    e7.set_width(width)
    at_200 = train(T.new_state(seed), 3000 + seed, CHECKPOINT_STEP - 100, lr)
    metric_200 = reported(at_200)
    checkpoint = train(at_200, 3000 + seed, 100, lr)
    metric_300 = reported(checkpoint)
    noise_reads = [reported(checkpoint) for _ in range(NOISE_READS)]
    reeval = abs(e7.loss_on(checkpoint, fresh(4 * eval_batches)) - metric_300)
    # replicate 0 continues the run's own data order; the others branch onto fresh data orders
    branch_seeds = [3000 + seed] + [9000 + 10 * seed + k for k in range(1, REPLICATES)]
    control, halve, double = [], [], []
    for data_seed in branch_seeds:
        control.append(reported(train(checkpoint, data_seed, PROBE_STEPS, lr)))
        halve.append(
            reported(train(checkpoint, data_seed, PROBE_STEPS, lr, 0.5)) - control[-1]
        )
        double.append(
            reported(train(checkpoint, data_seed, PROBE_STEPS, lr, 2.0)) - control[-1]
        )
    e7.set_width(2 * width)
    wide = train(T.new_state(seed), 3000 + seed, CHECKPOINT_STEP + PROBE_STEPS, lr)
    return dict(
        model=model,
        fault=fault,
        seed=seed,
        lr=lr,
        eval_batches=eval_batches,
        metric_200=metric_200,
        metric_300=metric_300,
        noise_reads=noise_reads,
        reeval_4x=reeval,
        control=control,
        halve_lr=halve,
        double_lr=double,
        wider=reported(wide) - control[0],
    )


def collect(models, healthy, seeds, conditions, name):
    out = dict(
        config=dict(models=models, healthy=healthy, seeds=tuple(seeds), device=T.DEV),
        cases=[],
    )
    start = time.time()
    for model, (layers, width) in models.items():
        for seed in seeds:
            for fault in conditions:
                case = run_case(model, layers, width, healthy[model]["lr"], fault, seed)
                out["cases"].append(case)
                print(
                    f"[{time.time() - start:5.0f}s] {model:5s} {fault:10s} seed {seed}: "
                    f"halve {statistics.mean(case['halve_lr']):+.4f} double {statistics.mean(case['double_lr']):+.4f} "
                    f"wider {case['wider']:+.4f}",
                    flush=True,
                )
                json.dump(out, open(f"results/e8/{name}_partial.json", "w"))
    json.dump(out, open(f"results/e8/{name}.json", "w"))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        json.dump(tune(MODELS), open("results/e8/tune.json", "w"), indent=1)
    elif stage == "pilot":
        five = [c for c in CONDITIONS if not c.startswith("mild")]
        collect(PILOT_MODEL, {"base": dict(lr=4e-3)}, (5, 6, 7, 8), five, "pilot")
    elif stage == "run":
        collect(
            MODELS,
            json.load(open("results/e8/tune.json")),
            range(71, 81),
            CONDITIONS,
            "e8",
        )
