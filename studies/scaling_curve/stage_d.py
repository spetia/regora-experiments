"""Stage D of the scaling study: what sets how long a paired comparison lasts?

Added after stages A and B; the plan and decision rules are in PLAN_D.md. `study.py` is reused unchanged: this
file registers two new arms and the stage-D cells, fixes each cell's learning rate, and calls `study.run_cell`.

    python stage_d.py list
    python stage_d.py run <cell>   # one cell (resumable), results in results/<cell>.json
    python stage_d.py all          # every cell in order, one process each, then commit the results
"""

import os
import subprocess
import sys

import study

# Two changes far smaller than anything in stages A to C: the learning rate up by one part in a million and by
# one part in ten thousand. They measure how fast an almost-identical run drifts away from its control.
NEW_ARMS = {"lr1+1e-6": (1 + 1e-6, 0.0, 0.0), "lr1+1e-4": (1 + 1e-4, 0.0, 0.0)}
PROBE = ("control", "lr1+1e-6", "lr1+1e-4", "lr1.01", "wd0.1", "swap1")
HORIZONS = (25, 50, 100, 200, 400, 800)


def cell(size, lr, **kw):
    """A stage-D cell. `lr_fixed` replaces tuning. `branch_lr_mult` (optional) scales the LR of every branch,
    the control included, while the checkpoint is still trained at `lr_fixed`."""
    return study.cell(
        size=size, lr_fixed=lr, arms="probe", orders=6, horizons=HORIZONS, **kw
    )


# In running order: the cells the rules need most come first.
CELLS = {
    "D_s2_lr2e-3": cell("s2", 2e-3),
    "D_s2_lr2e-3_fp32": cell("s2", 2e-3, amp=False),
    "D_s2_lr5e-4": cell("s2", 5e-4),
    "D_s2_ckpt2e-3_branch5e-4": cell("s2", 2e-3, branch_lr_mult=0.25),
    "D_s2_lr1e-3": cell("s2", 1e-3),
    "D_s1_lr1e-3": cell("s1", 1e-3),
}


ARMS = {
    **study.FULL_ARMS,
    **NEW_ARMS,
}  # taken once, so `register` can be called for several cells in a process


def register(name):
    """Point `study.run_cell(name)` at a stage-D cell: its arms, and its fixed LR as a one-value tuning grid (so
    the result file still records a loss for that LR on the tuning seeds)."""
    c = CELLS[name]
    scale = c.get("branch_lr_mult", 1.0)
    study.FULL_ARMS = {
        a: (mult * scale, wd, swap) for a, (mult, wd, swap) in ARMS.items()
    }
    study.ARM_SETS = {**study.ARM_SETS, "probe": PROBE}
    study.LR_GRIDS = {**study.LR_GRIDS, c["size"]: (c["lr_fixed"],)}
    study.CELLS = {**study.CELLS, name: c}


def run(name):
    train, val, sha = study.load_corpus()
    print(
        f"corpus: {len(train):,} train bytes, {len(val):,} val bytes, sha256 {sha[:16]}",
        flush=True,
    )
    register(name)
    study.run_cell(name, train, val, sha)


def run_all():
    """One process per cell. A failed cell is listed and the driver exits 1; the others still run."""
    here = study.HERE
    failed = [
        n
        for n in CELLS
        if subprocess.run(
            [sys.executable, os.path.abspath(__file__), "run", n], cwd=here
        ).returncode
    ]
    subprocess.run(["git", "add", "-A", "results"], cwd=here)
    subprocess.run(["git", "commit", "-q", "-m", "Stage D raw results"], cwd=here)
    if failed:
        sys.exit("FAILED CELLS: " + " ".join(failed))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for n, c in CELLS.items():
            print(n, c)
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "all":
        run_all()
    else:
        sys.exit(__doc__)
