import json, numpy as np

c = json.load(open("results/e0c/e0c.json"))
b = json.load(open("results/e0b/e0b.json"))["branch"]
ep = lambda curve: np.mean([v for _, v in curve][1:])
for arm in c["config"]["arms"]:
    effs, sds = [], []
    for ck in range(3):
        dl = np.array(
            [
                ep(c["branch"][f"{ck}|{ds}|{arm}"]) - ep(b[f"{ck}|{ds}|1.0"])
                for ds in range(8)
            ]
        )
        effs.append(dl.mean())
        sds.append(dl.std(ddof=1))
    e, s = np.mean(effs), np.sqrt(np.mean(np.square(sds)))
    mde = 2.8 * s / np.sqrt(8)
    det = abs(e / s) >= 1 and sum(np.sign(x) == np.sign(e) for x in effs) >= 2
    print(
        f"{arm:9s} effect {e:+.4f} sd_pair {s:.4f} effect/sd {e/s:+.2f} per-ckpt {[round(float(x),4) for x in effs]} MDE(n=8) {mde:.4f} detectable: {det}"
    )
