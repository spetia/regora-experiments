"""E4b: what does Regora telemetry cost a real training loop?

PRE-REGISTERED before running. E4 measured only the per-call cost (about 90 ns). This measures throughput.

Arms, each training TinyGPT for 200 steps from the same initial state and data order:
- none   : no per-step callback. The loss is never read on the host.
- sync   : a callback that receives loss.item() and does nothing. Isolates the cost of the device-to-host sync.
- record : the same callback, also recording two signals per step (loss, learning rate) through the PyO3 binding.

Design: 7 rounds. Each round runs the three arms in a rotated order to spread thermal and scheduling drift. The first
round is discarded as warm-up. Wall time is taken with time.perf_counter around the whole run; evaluation is disabled.

Reported: median ms/step per arm, and for each of rounds 2..7 the paired differences
  telemetry overhead = (record - sync) / sync     (cost of Regora itself)
  sync overhead      = (sync - none) / none       (cost of reading the loss every step)
with mean and a 95% t interval over the 6 rounds. Dropped and written record counts are reported.

Decision rule: telemetry overhead is LOW if the upper end of its 95% interval is below 1%; otherwise NOT SHOWN LOW.
The sync overhead is reported without a threshold: it is the adapter's choice, not Regora's cost.

Known limits: one small model on a shared laptop; memory overhead is not measured; CUDA is not measured.
Device comes from REGORA_DEV (cpu, mps).
"""

import json
import os
import statistics
import tempfile
import time

import regora
import torch

import tinygpt as T

torch.set_num_threads(2)
STEPS, ROUNDS, BASE_LR = 200, 7, 1e-3
ARMS = ("none", "sync", "record")


def timed_run(arm, segment_path):
    callback = None
    if arm == "sync":
        callback = lambda step, loss: None
    elif arm == "record":
        regora.attach(segment_path, 1)

        def callback(step, loss):
            regora.record(0, step, loss)
            regora.record(1, step, T.lr_at(step, BASE_LR))

    start = time.perf_counter()
    T.run(T.new_state(0), 0, STEPS, BASE_LR, eval_every=10**9, on_step=callback)
    elapsed = time.perf_counter() - start
    stats = regora.detach() if arm == "record" else None
    return 1000 * elapsed / STEPS, stats


def interval(values):
    mean = statistics.mean(values)
    half = (
        2.571 * statistics.stdev(values) / len(values) ** 0.5
    )  # two-sided 95% t, df = 5
    return mean, mean - half, mean + half


rounds, record_stats = [], []
workdir = tempfile.mkdtemp()
for r in range(ROUNDS):
    order = ARMS[r % 3 :] + ARMS[: r % 3]
    ms = {}
    for arm in order:
        ms[arm], stats = timed_run(arm, os.path.join(workdir, f"round{r}.seg"))
        if stats:
            record_stats.append(stats)
    rounds.append(ms)
    print(
        f"round {r} order {order}: "
        + "  ".join(f"{a} {ms[a]:.2f} ms/step" for a in ARMS),
        flush=True,
    )

kept = rounds[1:]
assert len(kept) == 6
telemetry = [(m["record"] - m["sync"]) / m["sync"] for m in kept]
sync = [(m["sync"] - m["none"]) / m["none"] for m in kept]
summary = dict(
    device=T.DEV,
    torch=torch.__version__,
    steps=STEPS,
    rounds=rounds,
    median_ms_per_step={a: statistics.median(m[a] for m in kept) for a in ARMS},
    telemetry_overhead=interval(telemetry),
    sync_overhead=interval(sync),
    records=[
        dict(recorded=s[0], dropped=s[1], write_errors=s[2]) for s in record_stats
    ],
)
json.dump(summary, open(f"results/e4b/e4b_{T.DEV}.json", "w"), indent=1)

print(
    "median ms/step:",
    {a: round(v, 3) for a, v in summary["median_ms_per_step"].items()},
)
for name in ("telemetry_overhead", "sync_overhead"):
    mean, low, high = summary[name]
    print(f"{name}: {100 * mean:+.2f}%  95% CI [{100 * low:+.2f}%, {100 * high:+.2f}%]")
print("records per run:", summary["records"][-1])
verdict = "LOW" if summary["telemetry_overhead"][2] < 0.01 else "NOT SHOWN LOW"
print(f"telemetry overhead: {verdict} by the pre-registered rule")
