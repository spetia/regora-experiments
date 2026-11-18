import json, itertools, numpy as np

d = json.load(open("results/e0/e0.json"))
cfg = d["config"]
sc = np.array([[v for _, v in c] for c in d["scratch"]])
steps_sc = [s for s, _ in d["scratch"][0]]
print("determinism (repeat identical):", d["determinism_repeat_identical"])
print("\nfrom-scratch SD of val loss (n=%d seeds)" % len(sc))
for s, m, sd in zip(steps_sc, sc.mean(0), sc.std(0, ddof=1)):
    print(f"  step {s:4d}: mean {m:.4f} sd {sd:.4f}")
B = d["branch"]
nds = 1 + max(int(k.split("|")[0]) for k in B)
hs = [s for s, _ in B["0|1.0"]]
V = {
    m: np.array([[v for _, v in B[f"{ds}|{m}"]] for ds in range(nds)])
    for m in (1.0, 0.5, 2.0)
}
print(f"\nbranch from shared ckpt (n={nds} data seeds); horizon h = steps after branch")
print(
    "  h  | sd(ctl) | "
    + " | ".join(
        f"mult {m}: dmean  sd_pair  rho  n80_pair n80_unpair" for m in (0.5, 2.0)
    )
)
for j, s in enumerate(hs):
    h = s - cfg["base_steps"]
    sdc = V[1.0][:, j].std(ddof=1)
    row = f"{h:4d} | {sdc:.4f} |"
    for m in (0.5, 2.0):
        dl = V[m][:, j] - V[1.0][:, j]
        mu, sp = dl.mean(), dl.std(ddof=1)
        sm = V[m][:, j].std(ddof=1)
        su = np.hypot(sdc, sm)
        rho = 1 - sp**2 / (
            sdc**2 + sm**2
        )  # Var(paired)=s1^2+sm^2-2*rho*s1*sm approx; use pooled form
        n_pair = 7.85 * (sp / mu) ** 2 if mu else float("inf")
        n_un = 7.85 * (su / mu) ** 2 if mu else float("inf")
        row += f" {mu:+.4f} {sp:.4f} {rho:+.2f} {n_pair:8.1f} {n_un:8.1f} |"
    print(row)
print(
    "\nrank agreement (E1-lite): fraction of data seeds where ordering of {0.5,1,2} at horizon h equals ordering at final"
)
fin = np.argsort(np.stack([V[m][:, -1] for m in (1.0, 0.5, 2.0)], 1), 1)
for j, s in enumerate(hs[:-1]):
    cur = np.argsort(np.stack([V[m][:, j] for m in (1.0, 0.5, 2.0)], 1), 1)
    print(f"  h={s-cfg['base_steps']:4d}: {np.mean((cur == fin).all(1)):.2f}")
print(
    "\nfinal-horizon mean val loss by mult:",
    {m: round(float(V[m][:, -1].mean()), 4) for m in V},
)
