"""Self-check on tiny models and random data: `python -m scale`. Runs on CPU, where training is deterministic."""

import torch

import scale


def check(task):
    start = task.new_state(0)
    fixed = task.eval_fixed()

    def after(*legs, data_seed=1, **kwargs):
        state = start
        for steps in legs:
            state = scale.train(task, state, data_seed, steps, task.lrs[2], **kwargs)
        return scale.loss_on(task, state, fixed)

    base = after(4)
    assert after(4) == base, "a branch from the same state must replay exactly"
    assert after(2, 2) == base, "resuming must match an uninterrupted run"
    assert after(4, data_seed=2) != base, "another data seed must give other batches"
    assert after(4, lr_mult=2.0) != base, "the LR multiplier must take effect"
    assert after(4, weight_decay=task.weight_decay_alt) != base
    assert scale.loss_on(task, start, fixed) != base, "the checkpoint must not move"
    fresh = task.eval_fresh(torch.Generator().manual_seed(0))
    assert len(fresh) == 1 and len(fixed) == task.eval_batches


if __name__ == "__main__":
    rng = torch.Generator().manual_seed(0)
    text = torch.randint(0, 256, (4000,), generator=rng, dtype=torch.uint8)
    check(
        scale.ByteLM(
            text, text, layers=1, width=32, heads=2, ctx=16, batch=4, eval_batches=2
        )
    )
    images = torch.randint(0, 256, (48, 3, 32, 32), generator=rng, dtype=torch.uint8)
    labels = torch.arange(48) % 5
    check(
        scale.Images(
            images,
            labels,
            images[:16],
            labels[:16],
            widths=(8,) * 4,
            batch=8,
            eval_batch=8,
        )
    )
    print("scale ok")
