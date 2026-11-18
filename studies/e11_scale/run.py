"""E11 data collection: the supported parts of Regora on larger models and a second task. The plan is in PLAN.md.

Usage: python -m studies.e11_scale.run speed TASK       (steps per second, to size the run)
       python -m studies.e11_scale.run tune TASK
       python -m studies.e11_scale.run pilot TASK
       python -m studies.e11_scale.run run TASK         (one process per task; a killed run resumes from its partial file)
       REGORA_EXACT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m studies.e11_scale.run exact TASK
"""

import json
import os
import statistics
import sys
import time

import torch

import scale

CONDITIONS = {
    "healthy": dict(),
    "lr_low": dict(lr_mult=1 / 12),
    "mild_low": dict(lr_mult=1 / 4),
    "mild_high": dict(lr_mult=3.0),
    "eval_noise": dict(noisy_eval=True),
}
# Every case stores the same fields as an E9 case, so E9's frozen rule reads it unchanged.
CHUNK, CHECKPOINT_CHUNKS, BRANCH_CHUNKS, FOLLOW_CHUNKS = 100, 10, 3, 15
REPLICATES, NOISE_READS = 3, 4
ARMS = dict(control=dict(), halve_lr=dict(lr_mult=0.5), double_lr=dict(lr_mult=2.0))
TUNE_SEEDS, PILOT_SEEDS, TEST_SEEDS = (3, 4), (11,), range(101, 111)
OUT = "results/e11"


def tune(task):
    """The healthy LR: lowest mean validation loss BRANCH_CHUNKS after the checkpoint, over TUNE_SEEDS."""
    steps = (CHECKPOINT_CHUNKS + BRANCH_CHUNKS) * CHUNK
    scores = {}
    for lr in task.lrs:
        finals = [
            scale.loss_on(
                task,
                scale.train(task, task.new_state(seed), 3000 + seed, steps, lr),
                task.eval_fixed(),
            )
            for seed in TUNE_SEEDS
        ]
        scores[lr] = statistics.mean(finals)
        print(f"tune lr {lr}: {scores[lr]:.4f}", flush=True)
    return dict(
        lr=min(scores, key=scores.get), scores={str(k): v for k, v in scores.items()}
    )


def run_case(task, name, healthy_lr, fault, seed, follow_chunks):
    cfg = CONDITIONS[fault]
    lr = healthy_lr * cfg.get("lr_mult", 1.0)
    noisy = cfg.get("noisy_eval", False)
    # faults outside the learning rate (E15): training options applied to the whole run, branches included
    fault_options = cfg.get("train", {})
    eval_rng = torch.Generator().manual_seed(7000 + seed)

    def reported(state):
        batches = task.eval_fresh(eval_rng) if noisy else task.eval_fixed()
        return scale.loss_on(task, state, batches)

    def curve(state, data_seed, chunks, **change):
        """The reported metric after each CHUNK steps, and the final state."""
        readings = []
        for _ in range(chunks):
            state = scale.train(
                task, state, data_seed, CHUNK, lr, **(fault_options | change)
            )
            readings.append(reported(state))
        return readings, state

    history, checkpoint = curve(task.new_state(seed), 3000 + seed, CHECKPOINT_CHUNKS)
    noise_reads = [reported(checkpoint) for _ in range(NOISE_READS)]
    # replicate 0 continues the run's own data order and runs on to `follow_chunks`; the others take fresh orders
    branch_seeds = [3000 + seed] + [9000 + 10 * seed + k for k in range(1, REPLICATES)]
    arms = dict(ARMS)
    if fault == "healthy":
        # a second control, to measure GPU non-determinism, and a small intervention, to measure pairing
        arms |= dict(
            repeat=dict(), weight_decay=dict(weight_decay=task.weight_decay_alt)
        )
    branches = {
        arm: [
            curve(checkpoint, s, BRANCH_CHUNKS if k else follow_chunks, **change)[0]
            for k, s in enumerate(branch_seeds)
        ]
        for arm, change in arms.items()
    }
    return dict(
        model=name,
        fault=fault,
        seed=seed,
        lr=lr,
        eval_batches=1 if noisy else task.eval_batches,
        history=history,
        metric=history[-3:],
        noise_reads=noise_reads,
        **branches,
    )


def collect(
    name, tuned, seeds, out_name, follow_chunks=FOLLOW_CHUNKS, conditions=CONDITIONS
):
    task = scale.TASKS[name]()
    partial = f"{OUT}/{out_name}_partial.json"
    out = dict(
        config=dict(
            task=name,
            tuned=tuned,
            seeds=tuple(seeds),
            device=scale.DEV,
            exact=scale.EXACT,
            gpu=torch.cuda.get_device_name() if scale.DEV == "cuda" else None,
            torch=torch.__version__,
            follow_chunks=follow_chunks,
        ),
        cases=[],
    )
    if os.path.exists(partial):
        out["cases"] = json.load(open(partial))["cases"]
    done = {(c["fault"], c["seed"]) for c in out["cases"]}
    start = time.time()
    # seed-major order, so a partial file always holds complete seeds
    for seed in seeds:
        for fault in conditions:
            if (fault, seed) in done:
                continue
            case = run_case(task, name, tuned["lr"], fault, seed, follow_chunks)
            out["cases"].append(case)
            at = BRANCH_CHUNKS - 1
            lean = statistics.mean(
                h[at] - d[at] for h, d in zip(case["halve_lr"], case["double_lr"])
            )
            print(
                f"[{time.time() - start:6.0f}s] {name} {fault:10s} seed {seed}: "
                f"at checkpoint {case['metric'][-1]:.4f} lean {lean:+.4f}",
                flush=True,
            )
            json.dump(out, open(partial, "w"))
    json.dump(out, open(f"{OUT}/{out_name}.json", "w"))


def speed(name, steps=200):
    task = scale.TASKS[name]()
    state = scale.train(task, task.new_state(0), 0, 20, task.lrs[2])  # warm up
    start = time.time()
    scale.loss_on(
        task, scale.train(task, state, 0, steps, task.lrs[2]), task.eval_fixed()
    )
    print(f"{name}: {steps / (time.time() - start):.1f} steps per second", flush=True)


if __name__ == "__main__":
    stage, name = sys.argv[1:3]
    tune_path = f"{OUT}/tune_{name}.json"
    if stage == "speed":
        speed(name)
    elif stage == "tune":
        json.dump(tune(scale.TASKS[name]()), open(tune_path, "w"), indent=1)
    elif stage == "pilot":
        collect(
            name,
            json.load(open(tune_path)),
            PILOT_SEEDS,
            f"pilot_{name}",
            BRANCH_CHUNKS,
        )
    elif stage == "exact":
        # healthy runs only, 300-step branches, under deterministic kernels: does exact replay restore pairing?
        assert scale.EXACT, "set REGORA_EXACT=1"
        collect(
            name,
            json.load(open(tune_path)),
            TEST_SEEDS,
            f"exact_{name}",
            BRANCH_CHUNKS,
            ("healthy",),
        )
    elif stage == "run":
        collect(name, json.load(open(tune_path)), TEST_SEEDS, f"e11_{name}")
