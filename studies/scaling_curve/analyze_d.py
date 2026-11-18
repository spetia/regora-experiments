"""Analysis for stage D. Prints each cell's table (via `analyze.show`) and the verdict of each rule in PLAN_D.md.

    python analyze_d.py

For stage-D cells the "tune" line holds the single fixed LR, so the grid-edge flag `analyze.show` prints means
nothing there.
"""

import numpy as np

import analyze

ARM, H = "lr1.01", 200
BUTTERFLY, H_LAST = "lr1+1e-6", 800
D_CELLS = (
    "D_s2_lr2e-3",
    "D_s2_lr2e-3_fp32",
    "D_s2_lr5e-4",
    "D_s2_ckpt2e-3_branch5e-4",
    "D_s2_lr1e-3",
    "D_s1_lr1e-3",
)
REFERENCE = ("A_s2", "B_s3")  # earlier cells the rules compare against
SHELF = 10  # "shelf life": the first horizon at which seeds saved is below this
SHELF_ARMS = ("lr1+1e-6", "lr1+1e-4", "lr1.01", "wd0.1", "swap1")


def fmt(x):
    return "n/a" if x is None else f"{x:.1f}"


def ratio_verdict(num, den, at_least=4.0, same=2.0):
    """The three-way rule every comparison uses: supported at >= 4x, rejected inside 2x either way."""
    if num is None or den is None:
        return None, "not run or diverged"
    r = num / den
    if r >= at_least:
        return r, "SUPPORTED"
    return r, "REJECTED" if 1 / same <= r <= same else "INCONCLUSIVE"


def compare(R, label, top, bottom, middle=None):
    a, b = (analyze.saved_at(R[n], ARM, H) if n in R else None for n in (top, bottom))
    r, verdict = ratio_verdict(a, b)
    shown = "" if r is None else f"  ratio {r:.2f}"
    print(f"  {label}: {top} {fmt(a)} vs {bottom} {fmt(b)}{shown}  -> {verdict}")
    if middle and middle in R:
        print(f"    reported: {middle} {fmt(analyze.saved_at(R[middle], ARM, H))}")


def butterfly(R, name, decides):
    """D4 for one cell. Only the bfloat16 cell decides the rule; the fp32 cell is reported."""
    if name not in R:
        print(f"  {name}: not run")
        return
    d = analyze.load(name)
    gap = float(np.abs(analyze.table(d, BUTTERFLY) - analyze.table(d, "control")).max())
    s = analyze.saved_at(R[name], BUTTERFLY, H_LAST)
    verdict = ""
    if decides:
        if gap == 0:
            verdict = "  -> UNINFORMATIVE (arm equals control)"
        elif s is None:
            verdict = "  -> diverged"
        else:
            verdict = f"  -> {'PASS' if s < SHELF else 'FAIL'}"
    print(
        f"  {name}: seeds saved {fmt(s)}, max |arm - control| over all readings {gap:.3g}{verdict}"
    )


def replication():
    """D5: `D_s2_lr2e-3` repeats `A_s2` (same seeds, LR and kernels) on the arms and horizons they share."""
    new, old = analyze.load("D_s2_lr2e-3"), analyze.load("A_s2")
    if new is None or old is None:
        print("  not run")
        return
    if new["lr"] != old["lr"]:
        print(f"  LRs differ ({new['lr']:g} vs {old['lr']:g}); not a replication")
        return
    hs_new, hs_old = list(new["config"]["horizons"]), list(old["config"]["horizons"])
    shared_h = [h for h in hs_new if h in hs_old]
    diffs = [
        abs(v[hs_new.index(h)] - old["branches"][k][hs_old.index(h)])
        for k, v in new["branches"].items()
        if k in old["branches"]
        for h in shared_h
    ]
    if not diffs:
        print("  no shared readings")
        return
    print(
        f"  {len(diffs)} shared readings, max |difference| {max(diffs):.3g} "
        f"-> {'bitwise equal' if max(diffs) == 0 else 'NOT bitwise equal'}"
    )


def shelf_life(R, arm):
    """First horizon with seeds saved below SHELF, as text; '>last' if it never drops below it in the cell."""
    for h in R["hs"]:
        s = analyze.saved_at(R, arm, h)
        if s is not None and s < SHELF:
            return str(h)
    return f">{R['hs'][-1]}"


def main():
    R = {n: analyze.cell_rows(n) for n in D_CELLS + REFERENCE}
    R = {n: v for n, v in R.items() if v}
    for n in D_CELLS:
        if n in R:
            analyze.show(n, R[n])

    print(
        f"\n\n##### Rules compare seeds saved for {ARM} at h={H}: >= 4x supported, within 2x rejected"
    )
    print(
        "\n##### D1: at a fixed size (4.9M), a lower learning rate keeps the saving longer"
    )
    compare(R, "LR 0.0005 vs 0.002", "D_s2_lr5e-4", "D_s2_lr2e-3", middle="D_s2_lr1e-3")
    print(
        "\n##### D1b: it is the learning rate during the branch, not the checkpoint it started from"
    )
    compare(
        R,
        "branch LR 0.0005 vs 0.002, same checkpoints",
        "D_s2_ckpt2e-3_branch5e-4",
        "D_s2_lr2e-3",
    )
    print(
        "\n##### D2: at a fixed learning rate (0.001), a smaller model keeps the saving longer"
    )
    compare(R, "0.88M vs 14.4M", "D_s1_lr1e-3", "B_s3", middle="D_s2_lr1e-3")
    print(
        "\n##### D3: bfloat16 rounding is what amplifies small changes (4.9M, LR 0.002)"
    )
    compare(R, "fp32 vs bfloat16", "D_s2_lr2e-3_fp32", "D_s2_lr2e-3")

    print(
        f"\n##### D4: a one-in-a-million change loses its pairing too (seeds saved < {SHELF} at h={H_LAST})"
    )
    butterfly(R, "D_s2_lr2e-3", decides=True)
    butterfly(R, "D_s2_lr2e-3_fp32", decides=False)

    print("\n##### D5: the 4.9M cell at LR 0.002 reproduces stage A's readings")
    replication()

    print(f"\n##### Reported: shelf life, the first horizon with seeds saved < {SHELF}")
    for n, v in R.items():
        print(
            f"  {n:26} lr {v['lr']:g}: "
            + "  ".join(f"{a} {shelf_life(v, a)}" for a in SHELF_ARMS if a in v["arms"])
        )


if __name__ == "__main__":
    main()
