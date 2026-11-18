"""Mechanics check for stage D on the CPU, with tiny cells in a temporary directory. Reads no effect or variance
from a real cell; it only asserts that the stage-D wiring does what PLAN_D.md says.

    python smoke_d.py
"""

import json
import os
import shutil
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = (
    "-1"  # leave the GPU to the running study (an empty value is ignored on Windows)
)

import torch

import analyze
import analyze_d
import stage_d
import study

TINY = dict(size="s0", ckpt_step=30, horizons=(5, 10), ckpts=1, orders=2)


def main():
    torch.set_num_threads(4)
    assert study.DEV == "cpu"
    tmp = tempfile.mkdtemp(prefix="smoke_d_")
    study.OUT_DIR = analyze.OUT = tmp
    stage_d.CELLS = {n: {**c, **TINY} for n, c in stage_d.CELLS.items()}
    train, val, sha = study.load_corpus()
    out = {}
    for name, c in stage_d.CELLS.items():
        stage_d.register(name)
        study.run_cell(name, train, val, sha)
        d = out[name] = json.load(open(os.path.join(tmp, name + ".json")))
        assert (
            d["done"]
            and d["lr"] == c["lr_fixed"]
            and list(d["tune"]) == [str(c["lr_fixed"])]
        ), name
        arms = {k.split("|")[2] for k in d["branches"]}
        assert arms == set(stage_d.PROBE) | {"twin"}, arms
        assert len(d["branches"]) == 2 * len(stage_d.PROBE) + 1, len(d["branches"])
        b = d["branches"]
        assert b["0|0|twin"] == b["0|0|control"], "identical twin differs"
        for a in stage_d.PROBE[1:]:
            assert any(
                b[f"0|{s}|{a}"] != b[f"0|{s}|control"] for s in range(2)
            ), f"{name}: {a} equals control"
        print(
            name,
            "ok;",
            "butterfly - control at the last horizon:",
            [b[f"0|{s}|lr1+1e-6"][-1] - b[f"0|{s}|control"][-1] for s in range(2)],
            flush=True,
        )

    # A cell with `branch_lr_mult` shares its checkpoints with the plain cell and differs after the branch.
    plain, scaled = out["D_s2_lr2e-3"], out["D_s2_ckpt2e-3_branch5e-4"]
    assert plain["ckpt_loss"] == scaled["ckpt_loss"], "checkpoints differ"
    assert (
        plain["branches"]["0|0|control"] != scaled["branches"]["0|0|control"]
    ), "branch LR was not scaled"
    # Registering a scaled cell must not leak into the next one.
    stage_d.register("D_s2_lr2e-3")
    assert (
        study.FULL_ARMS["control"] == (1.0, 0.0, 0.0)
        and study.FULL_ARMS["lr1.01"][0] == 1.01
    )
    # Stage D must not have altered what the stage A to C cells are.
    assert study.ARM_SETS["full"] == tuple(
        a for a in stage_d.ARMS if a not in stage_d.NEW_ARMS
    )

    # The analysis runs end to end on these files (its numbers mean nothing here).
    for ref in analyze_d.REFERENCE:
        shutil.copy(os.path.join(study.HERE, "results", ref + ".json"), tmp)
    analyze_d.H = analyze_d.H_LAST = 10
    analyze_d.main()
    shutil.rmtree(tmp)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
