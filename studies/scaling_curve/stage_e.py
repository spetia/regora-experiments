"""Stage E of the scaling study: does the saving last, and does pairing decide better, on the largest model?

One cell: the 25.5M model, branched from a later checkpoint and followed for 1600 steps, with the kind of changes
a team actually tries. The plan and decision rules are in PLAN_E.md. `study.py` and `stage_d.py` are reused
unchanged: this file adds one arm, one arm set and one cell to them.

    python stage_e.py list
    python stage_e.py run        # the cell (resumable), then commit the results
    python stage_e.py analyze    # the table and the verdict of each rule
    python stage_e.py smoke      # mechanics check on the CPU with a tiny cell; reads no real outcome
"""

import json
import math
import os
import subprocess
import sys

SMOKE = sys.argv[1:2] == ["smoke"]
if SMOKE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # leave the GPU to the running study

import numpy as np

import analyze
import replay
import stage_d
import study

NAME = "E_s4_long"
NEW_ARMS = {"lr0.9": (0.9, 0.0, 0.0)}
DECIDE = ("control", "lr0.9", "lr1.1", "lr1.3", "wd0.01", "wd0.1", "swap1")
SMALL = ("lr0.9", "lr1.1", "wd0.01", "wd0.1")  # the arms E1 pools
H_LAST, H_EARLY = 1600, 200
CELL = study.cell(
    size="s4",
    lr_fixed=5e-4,
    arms="decide",
    ckpt_step=2000,
    horizons=(200, 400, 800, 1600),
    ckpts=3,
    orders=6,
)


def run(cell=CELL):
    """Register the cell with stage D's machinery (fixed LR, extra arms) and run it."""
    stage_d.CELLS = {**stage_d.CELLS, NAME: cell}
    stage_d.ARMS = {**stage_d.ARMS, **NEW_ARMS}
    study.ARM_SETS = {**study.ARM_SETS, "decide": DECIDE}
    stage_d.run(NAME)


def commit():
    subprocess.run(["git", "add", "-A", "results"], cwd=study.HERE)
    subprocess.run(["git", "commit", "-q", "-m", "Stage E raw results"], cwd=study.HERE)


def verdicts(h_last=H_LAST, h_early=H_EARLY, late=(800, 1600)):
    R = analyze.cell_rows(NAME)
    if R is None:
        print("not run")
        return
    analyze.show(NAME, R)

    print(
        f"\n\n##### E1: the saving lasts (geometric mean of seeds saved over {', '.join(SMALL)} at h={h_last})"
    )
    saved = [analyze.saved_at(R, a, h_last) for a in SMALL]
    if any(s is None for s in saved):
        print("  an arm diverged -> not decided")
    else:
        g = math.exp(np.mean(np.log(saved)))
        verdict = "SUPPORTED" if g >= 3 else "REJECTED" if g < 2 else "INCONCLUSIVE"
        print(
            "  "
            + ", ".join(f"{a} {s:.1f}" for a, s in zip(SMALL, saved))
            + f"  geometric mean {g:.2f}"
            f"  (rule: >= 3 supported, < 2 rejected) -> {verdict}"
        )

    print(
        f"\n##### E2: with two training runs, a pair points the wrong way at most half as often (h in {late})"
    )
    rows = [r for r in replay.rows(NAME) if r["h"] in late]
    for r in rows:
        print(
            f"  {r['arm']:8} h={r['h']:5d} saved {r['saved']:6.1f}  one run each {replay.pct(r['u1'])}"
            f"  one pair {replay.pct(r['p1'])}  {'' if r['strict'] else '(established only by pairing)'}"
        )
    if len(rows) < 3:
        print(
            f"  {len(rows)} rows with an established direction (rule needs 3) -> UNINFORMATIVE"
        )
    else:
        u, p = np.mean([r["u1"] for r in rows]), np.mean([r["p1"] for r in rows])
        verdict = (
            "UNINFORMATIVE (unpaired is already right)"
            if u < 0.04
            else "PASS" if p <= u / 2 else "FAIL"
        )
        print(
            f"  mean over {len(rows)} rows: one run each {replay.pct(u)}, one pair {replay.pct(p)} -> {verdict}"
        )

    print(
        f"\n##### Reported: does a branch's paired difference at h={h_early} have the sign it has at h={h_last}?"
    )
    hs = R["hs"]
    for a, keep in R["sign_keep"].items():
        print(f"  {a:8} {keep[hs.index(h_early)]:.2f}")


def smoke():
    """Tiny cell on the CPU in a temporary directory: the wiring only."""
    import shutil
    import tempfile

    assert study.DEV == "cpu"
    tmp = tempfile.mkdtemp(prefix="smoke_e_")
    study.OUT_DIR = analyze.OUT = tmp
    run(
        {
            **CELL,
            "size": "s0",
            "ckpt_step": 30,
            "horizons": (5, 10),
            "ckpts": 2,
            "orders": 3,
        }
    )
    d = json.load(open(os.path.join(tmp, NAME + ".json")))
    b = d["branches"]
    assert d["done"] and d["lr"] == CELL["lr_fixed"]
    assert {k.split("|")[2] for k in b} == set(DECIDE) | {"twin"}
    assert len(b) == 2 * (3 * len(DECIDE) + 1)
    assert b["0|0|twin"] == b["0|0|control"], "identical twin differs"
    assert all(
        b[f"0|0|{a}"] != b["0|0|control"] for a in DECIDE[1:]
    ), "an arm equals the control"
    assert study.FULL_ARMS["lr0.9"] == (0.9, 0.0, 0.0) and study.FULL_ARMS[
        "control"
    ] == (1.0, 0.0, 0.0)
    verdicts(h_last=10, h_early=5, late=(5, 10))
    shutil.rmtree(tmp)
    print("\nSMOKE OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print(NAME, CELL)
    elif cmd == "run":
        run()
        commit()
    elif cmd == "analyze":
        verdicts()
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit(__doc__)
