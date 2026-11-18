import json, numpy as np

d = json.load(open("results/e0b/e0b.json"))
B = d["branch"]
print("determinism:", d["det"])
V = {
    (ck, m): np.array([[v for _, v in B[f"{ck}|{ds}|{m}"]] for ds in range(8)])
    for ck in range(3)
    for m in (1.0, 0.5, 2.0)
}
ep = lambda a: a[:, 1:].mean(1)
for m in (0.5, 2.0):
    effs, sds = [], []
    for ck in range(3):
        dl = ep(V[ck, m]) - ep(V[ck, 1.0])
        effs.append(dl.mean())
        sds.append(dl.std(ddof=1))
        print(
            f"mult {m} ckpt {ck}: effect {dl.mean():+.4f} sd_pair {dl.std(ddof=1):.4f} ctl sd {ep(V[ck,1.0]).std(ddof=1):.4f} effect/sd {dl.mean()/dl.std(ddof=1):+.2f}"
        )
    e, s = np.mean(effs), np.sqrt(np.mean(np.square(sds)))
    print(
        f"mult {m} POOLED: effect {e:+.4f} (across-ckpt sd {np.std(effs, ddof=1):.4f}) sd_pair {s:.4f} effect/sd {e/s:+.2f} n80_pair {7.85*(s/e)**2:.1f}"
    )
print(
    "final-horizon mean val loss:",
    {
        m: round(float(np.mean([V[ck, m][:, -1].mean() for ck in range(3)])), 4)
        for m in (1.0, 0.5, 2.0)
    },
)
fin = {
    ck: np.argsort(np.stack([V[ck, m][:, -1] for m in (1.0, 0.5, 2.0)], 1), 1)
    for ck in range(3)
}
for j, h in enumerate((50, 100, 150)):
    print(
        f"rank agreement h={h}:",
        round(
            float(
                np.mean(
                    [
                        (
                            np.argsort(
                                np.stack([V[ck, m][:, j] for m in (1.0, 0.5, 2.0)], 1),
                                1,
                            )
                            == fin[ck]
                        )
                        .all(1)
                        .mean()
                        for ck in range(3)
                    ]
                )
            ),
            2,
        ),
        "(chance 0.17)",
    )
