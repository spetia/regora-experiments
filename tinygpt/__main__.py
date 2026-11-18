"""Self-check: `python -m tinygpt` trains 100 steps twice and reports determinism."""

from tinygpt import *

import time

print(
    "corpus",
    len(TRAIN),
    len(VAL),
    "params",
    sum(p.numel() for p in GPT(**CFG).parameters()),
)
s = new_state(0)
t = time.time()
_, c1 = run(s, 0, 100, 3e-3)
dt = time.time() - t
_, c2 = run(s, 0, 100, 3e-3)
print(
    f"{dt/100*1000:.0f} ms/step; bitwise deterministic across repeats: {c1 == c2}; curve {c1}"
)
