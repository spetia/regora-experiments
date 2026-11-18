"""Driver: stage A, freeze and commit the stage-B predictions, stage B, stage C. No human between stages.

A cell whose process exits non-zero is listed as failed and the driver exits 1; the remaining cells still run,
since a cell dropped for time or memory is reported as not run (PLAN.md)."""

import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
A = ["A_s0", "A_s1", "A_s2"]
B = ["B_s3", "B_s4"]
C = [
    "C_default_kernels",
    "C_fp32",
    "C_data100k",
    "C_data1m",
    "C_age150",
    "C_age2400",
    "C_batch8",
    "C_long",
    "C_batch128",
]
failed = []


def sh(*a):
    print("+", " ".join(a), flush=True)
    return subprocess.run(a, cwd=HERE).returncode


def run_cells(cells):
    for c in cells:
        if sh(PY, "study.py", "run", c) != 0:
            failed.append(c)


def git(msg):
    sh("git", "add", "-A", "results")
    sh("git", "commit", "-q", "-m", msg)


stages = sys.argv[1:] or ["A", "B", "C"]
if "A" in stages:
    run_cells(A)
    if not os.path.exists(os.path.join(HERE, "results", "prediction.json")):
        assert sh(PY, "analyze.py", "predict") == 0
    git("Stage A raw results; freeze stage-B predictions")
if "B" in stages:
    assert os.path.exists(
        os.path.join(HERE, "results", "prediction.json")
    ), "predictions not frozen"
    run_cells(B)
    git("Stage B raw results")
if "C" in stages:
    run_cells(C)
    git("Stage C raw results")
if failed:
    sys.exit("FAILED CELLS: " + " ".join(failed))
print("ALL DONE", flush=True)
