"""E7: can interventions diagnose why a run is underperforming, and does choosing them by information gain save compute?

PRE-REGISTERED before running. This is the first test of the "competing explanations" loop (spec 5.4): keep several
causes in play, run the probe with the most expected information per unit of compute, update, stop when one cause
dominates. E1b to E6 tested config selection, not this.

Data
- Five conditions, injected on purpose so the true cause is known:
    healthy     lr 4e-3, width 128, metric on the fixed 20-batch eval set
    lr_high     lr 3e-2
    lr_low      lr 3e-4
    eval_noise  healthy training, but the reported metric is one fresh random eval batch per reading
    capacity    width 32
- 10 fresh seeds 31..40. Seed s fixes init (s) and data order (3000 + s). Constant LR after a 50-step warmup, MPS.
- Each case trains 300 steps, then runs four probes from that checkpoint. Every probe yields one number, read through
  the case's own reported metric:
    halve_lr   100 more steps at 0.5x LR; metric minus the metric of a 100-step control branch      cost 100
    double_lr  same at 2x LR                                                                        cost 100
    reeval_4x  |metric on 4x as many fresh eval batches - reported metric| at the checkpoint        cost 4 * batches / 3
    wider      2x width trained from scratch for 400 steps; metric minus the control's              cost 1600
  The control branch costs 100 steps, charged once, with the first of halve_lr / double_lr / wider. Costs are in
  baseline training steps; wider is 400 steps at 4x the parameters. A non-finite reading counts as +infinity.

Evaluation (analyze.py), leave-one-seed-out over the 10 seeds, 50 held-out cases
- Fit on the 9 training seeds only: per probe, cut the 45 pooled readings at their terciles into three outcomes, then
  estimate p(outcome | condition) with add-one smoothing.
- Held-out case: uniform prior over the five conditions, Bayes update per probe outcome. A policy picks probes one at
  a time and stops when the top posterior reaches 0.9 or no probes are left. Diagnosis = top posterior.
- Policies: EIG_PER_COST (most expected information gain per unit cost), EIG (ignores cost), CHEAPEST (cheapest
  first), RANDOM (mean over all 24 orders), ALL (every probe, no early stop).

Decision rules
- H1 diagnosis works: ALL is right on at least 40 of 50 held-out cases (0.80; chance is 0.20).
- H2 selection saves compute: EIG_PER_COST costs at most 0.8x RANDOM's mean cost, with accuracy no more than 0.04
  below RANDOM's. H2 is STRONG if the same holds against CHEAPEST, otherwise WEAK (a trivial ordering does as well).
- H3 honest confidence: among EIG_PER_COST stops with top posterior >= 0.9, at least 0.85 are right.
- Reported, not decisive: a PASSIVE classifier with no probes, same naive Bayes on (reported metric at step 300,
  change from step 200). It shows how much the probes add over just watching the run.

Known limits: one 875k-parameter model and one dataset. The causes are single, injected and mutually exclusive; real
runs have mixed and unlisted causes. Likelihoods are learned from the same model family they are tested on, which is
the best case for this method. PASSIVE can lean on absolute loss level, which transfers across projects worse than
paired probe outcomes do. H2 is bounded by having only four probes.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

import tinygpt as T

torch.set_num_threads(2)
BASE = dict(lr=4e-3, width=128, eval_batches=20, fixed_eval=True)
FAULTS = {
    "healthy": BASE,
    "lr_high": BASE | dict(lr=3e-2),
    "lr_low": BASE | dict(lr=3e-4),
    "eval_noise": BASE | dict(eval_batches=1, fixed_eval=False),
    "capacity": BASE | dict(width=32),
}
CHECKPOINT_STEP, PROBE_STEPS, NO_EVAL = 300, 100, 10**9
N_LAYER = 4  # E7c overrides this and FAULTS to rerun the study on a larger model


def set_width(width):
    T.CFG = dict(n_layer=N_LAYER, n_embd=width, n_head=4)


@torch.no_grad()
def loss_on(state, starts):
    """Mean validation loss of `state` over the windows starting at `starts`."""
    model = T.GPT(**T.CFG).to(T.DEV)
    model.load_state_dict(state["model"])
    model.eval()
    losses = []
    for i in range(0, len(starts), T.BATCH):
        x, y = T.batch(T.VAL, starts[i : i + T.BATCH])
        losses.append(F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item())
    return sum(losses) / len(losses)


def run_case(fault, seed):
    cfg = FAULTS[fault]
    eval_rng = torch.Generator().manual_seed(7000 + seed)
    data_seed = 3000 + seed

    def fresh(batches):
        return torch.randint(
            0, len(T.VAL) - T.CTX - 1, (batches * T.BATCH,), generator=eval_rng
        )

    def reported(state):
        return loss_on(
            state, T.VAL_IDX if cfg["fixed_eval"] else fresh(cfg["eval_batches"])
        )

    def train(state, steps, lr_mult=1.0):
        return T.run(
            state, data_seed, steps, cfg["lr"], lr_mult=lr_mult, eval_every=NO_EVAL
        )[0]

    set_width(cfg["width"])
    at_200 = train(T.new_state(seed), CHECKPOINT_STEP - 100)
    metric_200 = reported(at_200)
    checkpoint = train(at_200, 100)
    metric_300 = reported(checkpoint)
    reeval = abs(loss_on(checkpoint, fresh(4 * cfg["eval_batches"])) - metric_300)
    control = reported(train(checkpoint, PROBE_STEPS))
    halve = reported(train(checkpoint, PROBE_STEPS, 0.5)) - control
    double = reported(train(checkpoint, PROBE_STEPS, 2.0)) - control
    set_width(2 * cfg["width"])
    wider = reported(train(T.new_state(seed), CHECKPOINT_STEP + PROBE_STEPS)) - control
    return dict(
        fault=fault,
        seed=seed,
        eval_batches=cfg["eval_batches"],
        metric_200=metric_200,
        metric_300=metric_300,
        control=control,
        probes=dict(halve_lr=halve, double_lr=double, reeval_4x=reeval, wider=wider),
    )


def main(seeds, name):
    """Run every condition for each seed and write results/e7/<name>.json."""
    out = dict(
        config=dict(
            faults=FAULTS, seeds=tuple(seeds), device=T.DEV, torch=torch.__version__
        ),
        cases=[],
    )
    start = time.time()
    # seed-major order, so a partial file always holds complete seeds
    for seed in seeds:
        for fault in FAULTS:
            case = run_case(fault, seed)
            out["cases"].append(case)
            readings = " ".join(f"{k} {v:+.4f}" for k, v in case["probes"].items())
            print(
                f"[{time.time() - start:5.0f}s] {fault:10s} seed {seed}: at300 {case['metric_300']:.4f} {readings}",
                flush=True,
            )
            json.dump(out, open(f"results/e7/{name}_partial.json", "w"))
    json.dump(out, open(f"results/e7/{name}.json", "w"))


if __name__ == "__main__":
    if sys.argv[1:] == ["pilot"]:
        main((1,), "pilot")
    else:
        main(range(31, 41), "e7")
