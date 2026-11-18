"""TinyGPT adapter for the Regora harness. One process = one run; everything comes from REGORA_* env.

Base config: lr 1e-3 (constant after warmup), wd 0, clip 1.0. Interventions override it. The seed fixes the data order,
and the init too when there is no base checkpoint, so runs with equal seeds are paired.
Make a checkpoint:  python -m tinygpt.trainer --make-checkpoint ckpt.pt [steps]
"""

import json, os, sys, torch, tinygpt as T

if len(sys.argv) > 2 and sys.argv[1] == "--make-checkpoint":
    st, _ = T.run(
        T.new_state(0),
        0,
        int(sys.argv[3]) if len(sys.argv) > 3 else 200,
        1e-3,
        eval_every=10**9,
    )
    torch.save(st, sys.argv[2])
    sys.exit(0)

import regora

seed, horizon, every = (
    int(os.environ[k])
    for k in ("REGORA_SEED", "REGORA_HORIZON_STEPS", "REGORA_EVAL_EVERY")
)
p = dict(lr=1e-3, wd=0.0, clip=1.0)
for i in json.loads(os.environ["REGORA_CONDITION"]):
    k = i["kind"]
    if k == "SetLearningRate":
        p["lr"] = i["value"]
    elif k == "SetWeightDecay":
        p["wd"] = i["value"]
    elif k == "SetGradClip":
        p["clip"] = i["max_norm"]
    else:
        sys.exit(
            f"undeclared intervention {k}"
        )  # never silently ignore: caps.json must match this list
ck = os.environ.get("REGORA_BASE_CHECKPOINT")
state = torch.load(ck, weights_only=False) if ck else T.new_state(seed)
regora.attach(os.environ["REGORA_SEGMENT"], int(os.environ["REGORA_RUN_ID"]))
_, curve = T.run(
    state, seed, horizon, p["lr"], eval_every=every, wd=p["wd"], clip=p["clip"]
)
for step, v in curve:
    regora.record(0, step, v)  # signal 0 = val_loss (see caps.json)
regora.detach()
