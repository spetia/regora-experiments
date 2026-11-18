"""E10 data collection: a run that corrects its own learning rate, against tuned schedules. Plan: PLAN.md.

Usage: python -m studies.e10_self_correcting.run run MODEL          (one process per model; they are independent)
       python -m studies.e10_self_correcting.run pilot NAME MODEL[,MODEL] SEED[,SEED]
"""

import json
import math
import sys
import time

import torch

import studies.e7_diagnosis.run as e7
import tinygpt as T

MODELS = {
    "slim": (5, 96),
    "stout": (3, 256),
    "lanky": (8, 128),
}  # layers, width; none seen before
DEV_MODELS = {
    "tiny": (2, 96),
    "base": (4, 128),
    "deep": (8, 192),
}  # seen in E7 to E9; pilot only
GRID = (5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2)
DEFAULT_START = 2e-3
STARTS = (GRID[0], DEFAULT_START, GRID[-1])
STEPS, SEGMENT = 2400, 300
ARMS = (0.5, 1.0, 2.0)  # LR multipliers tried on every segment
BUDGET = len(ARMS) * STEPS  # training steps the controller may spend in total
MARGIN = 0.0  # fixed in PLAN.md after the pilot, which also ran PILOT_MARGINS
PILOT_MARGINS = (0.0, 0.2)
TEST_SEEDS = range(91, 101)
# Schedules are chosen on T.VAL_IDX and scored on these separate windows, so picking the best of several
# noisy validation readings cannot flatter the result.
TEST_IDX = torch.randint(
    0,
    len(T.VAL) - T.CTX - 1,
    (40 * T.BATCH,),
    generator=torch.Generator().manual_seed(4242),
)


def finite(value):
    return value if math.isfinite(value) else math.inf


def scheduled(seed, lr, cosine):
    """One ordinary run at a fixed peak LR: constant after warmup, or cosine to 10%."""
    state, curve = T.run(
        T.new_state(seed),
        3000 + seed,
        STEPS,
        lr,
        eval_every=SEGMENT,
        total=STEPS if cosine else None,
    )
    return dict(
        lr=lr,
        val=[finite(v) for _, v in curve],
        test=finite(e7.loss_on(state, TEST_IDX)),
    )


def controlled(seed, start, margin):
    """Branch-and-keep: train every segment at half, the same and double the LR, and continue from the best.

    The LR only changes if that branch beats the unchanged one by more than `margin` x the unchanged branch's
    progress over the segment. If every branch diverges, the segment is retried at a quarter of the LR; the
    wasted steps still count against BUDGET, so the run can end short of STEPS.
    """
    state, lr, spent, trace = T.new_state(seed), start, 0, []
    previous = e7.loss_on(state, T.VAL_IDX)
    while state["step"] < STEPS and spent < BUDGET:
        ends = {
            arm: T.run(state, 3000 + seed, SEGMENT, lr, lr_mult=arm, eval_every=SEGMENT)
            for arm in ARMS
        }
        spent += len(ARMS) * SEGMENT
        val = {arm: finite(curve[-1][1]) for arm, (_, curve) in ends.items()}
        best = min(val, key=val.get)
        if not math.isfinite(val[best]):
            trace.append(dict(lr=lr, val=list(val.values()), kept=None))
            lr /= 4
            continue
        progress = (
            max(abs(previous - val[1.0]), 0.01 * val[1.0])
            if math.isfinite(val[1.0])
            else 0.0
        )
        keep = best if val[1.0] - val[best] > margin * progress else 1.0
        trace.append(dict(lr=lr, val=list(val.values()), kept=keep))
        state, lr, previous = ends[keep][0], lr * keep, val[keep]
    return dict(
        start=start,
        margin=margin,
        trace=trace,
        steps=state["step"],
        test=finite(e7.loss_on(state, TEST_IDX)),
    )


def run_seed(model, seed, margins):
    return dict(
        model=model,
        seed=seed,
        constant=[scheduled(seed, lr, cosine=False) for lr in GRID],
        cosine=[scheduled(seed, lr, cosine=True) for lr in GRID],
        controlled=[
            controlled(seed, start, margin) for margin in margins for start in STARTS
        ],
    )


def collect(models, seeds, margins, name):
    out = dict(
        config=dict(
            models=models, seeds=tuple(seeds), grid=GRID, steps=STEPS, device=T.DEV
        ),
        cases=[],
    )
    began = time.time()
    for model, (layers, width) in models.items():
        e7.N_LAYER = layers
        e7.set_width(width)
        for seed in seeds:
            case = run_seed(model, seed, margins)
            out["cases"].append(case)
            best = min(case["cosine"], key=lambda run: run["test"])
            ours = " ".join(f"{run['test']:.4f}" for run in case["controlled"])
            print(
                f"[{time.time() - began:6.0f}s] {model:5s} seed {seed}: best cosine {best['test']:.4f} "
                f"at {best['lr']}; controlled {ours}",
                flush=True,
            )
            json.dump(out, open(f"results/e10/{name}_partial.json", "w"))
    json.dump(out, open(f"results/e10/{name}.json", "w"))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "pilot":
        models = {m: DEV_MODELS[m] for m in sys.argv[3].split(",")}
        collect(
            models, [int(s) for s in sys.argv[4].split(",")], PILOT_MARGINS, sys.argv[2]
        )
    elif stage == "run":
        collect(
            {sys.argv[2]: MODELS[sys.argv[2]]},
            TEST_SEEDS,
            (MARGIN,),
            f"e10_{sys.argv[2]}",
        )
