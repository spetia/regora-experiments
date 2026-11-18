"""E9 data collection: 300-step LR probes, follow-through branches and an early wider-model reading. The plan is in PLAN.md next to this file.

Usage: python -m studies.e9_longer_probes.run tune
       python -m studies.e9_longer_probes.run run MODEL          (one process per model; they are independent)
       python -m studies.e9_longer_probes.run pilot NAME MODEL[,MODEL] CONDITION[,CONDITION]
"""

import json
import statistics
import sys
import time

import torch

import studies.e7_diagnosis.run as e7
import studies.e8_label_free.run as e8
import tinygpt as T

MODELS = {"squat": (3, 160), "broad": (4, 320), "tall": (10, 128)}  # layers, width
DEV_MODELS = {
    "tiny": (2, 96),
    "base": (4, 128),
    "mid": (6, 256),
    "deep": (8, 192),
}  # seen before; pilots only
CONDITIONS = (
    e8.CONDITIONS
)  # capacity (width / 4) runs at the narrow model's own tuned LR
TUNE_LRS = e8.TUNE_LRS + (
    1.6e-2,
)  # the first pilot's narrow models sat on the old grid's upper edge
CHUNK, BRANCH_CHUNKS, FOLLOW_CHUNKS, WIDER_CHUNKS = 100, 3, 9, 1
WIDER_LR_MULTS = (
    1.0,
)  # the pilots also ran 0.5 and 400 steps; neither separated capacity from healthy
NARROW = "/narrow"
PILOT_SEEDS, TEST_SEEDS = (11, 12, 13), range(81, 91)


def tune(models):
    """Healthy LR for each model and for its width / 4 version."""
    narrow = {
        name + NARROW: (layers, width // 4) for name, (layers, width) in models.items()
    }
    return e8.tune(models | narrow, TUNE_LRS)


def run_case(model, layers, width, tuned, fault, seed):
    cfg = CONDITIONS[fault]
    narrow = "width_div" in cfg
    lr = tuned[model + NARROW if narrow else model]["lr"] * cfg.get("lr_mult", 1.0)
    width //= cfg.get("width_div", 1)
    noisy = cfg.get("noisy_eval", False)
    eval_rng = torch.Generator().manual_seed(7000 + seed)

    def reported(state):
        if not noisy:
            return e7.loss_on(state, T.VAL_IDX)
        starts = torch.randint(
            0, len(T.VAL) - T.CTX - 1, (T.BATCH,), generator=eval_rng
        )
        return e7.loss_on(state, starts)

    def curve(state, data_seed, chunks, lr_mult=1.0):
        """The reported metric after each CHUNK steps, and the final state."""
        readings = []
        for _ in range(chunks):
            state = e8.train(state, data_seed, CHUNK, lr, lr_mult)
            readings.append(reported(state))
        return readings, state

    e7.N_LAYER = layers
    e7.set_width(width)
    metric, checkpoint = curve(
        T.new_state(seed), 3000 + seed, e8.CHECKPOINT_STEP // CHUNK
    )
    noise_reads = [reported(checkpoint) for _ in range(e8.NOISE_READS)]
    # replicate 0 continues the run's own data order and runs on to FOLLOW_CHUNKS, to score the advice;
    # the others branch onto fresh data orders
    branch_seeds = [3000 + seed] + [
        9000 + 10 * seed + k for k in range(1, e8.REPLICATES)
    ]
    branches = {
        name: [
            curve(checkpoint, s, BRANCH_CHUNKS if k else FOLLOW_CHUNKS, mult)[0]
            for k, s in enumerate(branch_seeds)
        ]
        for name, mult in (("control", 1.0), ("halve_lr", 0.5), ("double_lr", 2.0))
    }
    e7.set_width(2 * width)
    wider = {
        str(mult): curve(T.new_state(seed), 3000 + seed, WIDER_CHUNKS, mult)[0]
        for mult in WIDER_LR_MULTS
    }
    return dict(
        model=model,
        fault=fault,
        seed=seed,
        lr=lr,
        eval_batches=1 if noisy else e8.FIXED_EVAL_BATCHES,
        metric=metric,
        noise_reads=noise_reads,
        wider=wider,
        **branches,
    )


def collect(models, tuned, seeds, conditions, name):
    out = dict(
        config=dict(models=models, tuned=tuned, seeds=tuple(seeds), device=T.DEV),
        cases=[],
    )
    start = time.time()
    for model, (layers, width) in models.items():
        for seed in seeds:
            for fault in conditions:
                case = run_case(model, layers, width, tuned, fault, seed)
                out["cases"].append(case)
                lean = statistics.mean(
                    h[BRANCH_CHUNKS - 1] - d[BRANCH_CHUNKS - 1]
                    for h, d in zip(case["halve_lr"], case["double_lr"])
                )
                print(
                    f"[{time.time() - start:5.0f}s] {model:5s} {fault:10s} seed {seed}: "
                    f"at300 {case['metric'][-1]:.4f} lean {lean:+.4f}",
                    flush=True,
                )
                json.dump(out, open(f"results/e9/{name}_partial.json", "w"))
    json.dump(out, open(f"results/e9/{name}.json", "w"))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "pilot":
        name, models, conditions = (
            sys.argv[2],
            sys.argv[3].split(","),
            sys.argv[4].split(","),
        )
        models = {m: DEV_MODELS[m] for m in models}
        tuned = tune(models)
        json.dump(tuned, open(f"results/e9/{name}_tune.json", "w"), indent=1)
        collect(models, tuned, PILOT_SEEDS, conditions, name)
    elif stage == "tune":
        json.dump(tune(MODELS), open("results/e9/tune.json", "w"), indent=1)
    elif stage == "run":
        model = sys.argv[2]
        tuned = json.load(open("results/e9/tune.json"))
        collect({model: MODELS[model]}, tuned, TEST_SEEDS, CONDITIONS, f"e9_{model}")
