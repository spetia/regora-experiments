"""E12 data collection: fine-tunes of one pretrained model, scored item by item. The plan is in PLAN.md.

A run fine-tunes MODEL on a fixed training subset and records, for every test item, whether it was answered
correctly and its loss. The seed sets the order of the training data and nothing else, so two arms on one seed are
a paired comparison and two runs of one arm on different seeds measure training noise.

Usage (all with REGORA_DEV=cuda REGORA_EXACT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8):
       python -m studies.e12_paired_eval.run speed TASK   (seconds per run and an exact-replay check)
       python -m studies.e12_paired_eval.run tune TASK
       python -m studies.e12_paired_eval.run pilot TASK
       python -m studies.e12_paired_eval.run run TASK     (a killed run resumes from its partial file)
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

import scale

MODEL = "HuggingFaceTB/SmolLM2-360M"
N_TRAIN, N_TEST, N_DEV = 4000, 2000, 1000
BATCH, STEPS, WARMUP, CLIP, MAX_TEXT = 32, 250, 25, 1.0, 96
LRS = (5e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4, 6.4e-4, 1.28e-3, 2.56e-3)
ARMS = {
    "control": dict(),
    "half_lr": dict(lr_mult=0.5),
    "prompt": dict(prompt=1),
    "label_smoothing": dict(smoothing=0.1),
    "more_steps": dict(steps=300),
}
TUNE_SEEDS, PILOT_SEEDS, TEST_SEEDS = (3, 4), (11,), range(101, 121)
OUT = "results/e12"

# name -> (dataset path, config, train split, test split, text of one row, the words that name the labels).
# The text is (body, question); the body is truncated, the question never is. PROMPTS[name][1] is the reworded question.
TASKS = {
    "topic": (
        "fancyzhx/ag_news",
        None,
        "train",
        "test",
        lambda r: (r["text"], ""),
        (" World", " Sports", " Business", " Science"),
    ),
    "nli": (
        "nyu-mll/glue",
        "mnli",
        "train",
        "validation_matched",
        lambda r: (r["premise"], f"\nHypothesis: {r['hypothesis']}"),
        (" Yes", " Maybe", " No"),
    ),
}

PROMPTS = {
    "topic": ("\nTopic:", "\nThis news story is about:"),
    "nli": (
        "\nDoes the first text imply the hypothesis?",
        "\nGiven the first text, is the hypothesis true?",
    ),
}


class Task:
    """Tokenised train, test and dev sets on the device, plus the model and its pretrained weights."""

    def __init__(self, name, model=MODEL, sizes=(N_TRAIN, N_TEST, N_DEV)):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path, config, train_split, test_split, self.text, words = TASKS[name]
        self.tok = AutoTokenizer.from_pretrained(model)
        self.pad = self.tok.pad_token_id or self.tok.eos_token_id
        first = [self.tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
        assert len(set(first)) == len(
            words
        ), "label words must start with distinct tokens"
        self.labels = torch.tensor(first, device=scale.DEV)

        data = load_dataset(path, config)
        pick = lambda split, seed: torch.randperm(
            len(data[split]), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        train, held = pick(train_split, 0), pick(test_split, 1)
        n_train, n_test, n_dev = sizes
        # each set is encoded once per wording of the question: set[0] is the control's, set[1] the reworded one
        both = lambda split, rows: [
            self.encode(data[split], rows, q) for q in PROMPTS[name]
        ]
        self.train = both(train_split, train[:n_train])
        self.test = both(test_split, held[:n_test])
        self.dev = both(test_split, held[n_test : n_test + n_dev])

        self.model = AutoModelForCausalLM.from_pretrained(model).to(scale.DEV)
        self.base = {k: v.clone() for k, v in self.model.state_dict().items()}

    def encode(self, split, rows, question):
        """Left-padded ids, so the last position always holds the end of the prompt."""
        ids = []
        for i in rows:
            body, suffix = self.text(split[i])
            suffix += question
            ids.append(
                self.tok(body, truncation=True, max_length=MAX_TEXT)["input_ids"]
                + self.tok(suffix, add_special_tokens=False)["input_ids"]
            )
        width = max(map(len, ids))
        x = torch.tensor([[self.pad] * (width - len(r)) + r for r in ids])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in ids])
        y = torch.tensor([split[i]["label"] for i in rows])
        return tuple(t.to(scale.DEV) for t in (x, mask, y))

    def logits(self, x, mask):
        """Scores of the label words as the next token."""
        with scale.autocast():
            out = self.model(input_ids=x, attention_mask=mask).logits[:, -1]
        return out[:, self.labels].float()


def finetune(task, seed, lr, lr_mult=1.0, smoothing=0.0, steps=STEPS, prompt=0):
    """Fine-tune from the pretrained weights. The seed sets the data order; a longer run sees the same order first."""
    model = task.model
    model.load_state_dict(task.base)
    model.train()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    x, mask, y = task.train[prompt]
    epochs = -(-steps * BATCH // len(y))
    order = torch.cat([torch.randperm(len(y), generator=rng) for _ in range(epochs)])
    opt = torch.optim.AdamW(model.parameters(), lr=lr * lr_mult, weight_decay=0.0)
    for step in range(steps):
        # linear warmup, then linear decay to zero at the last step
        ramp = min((step + 1) / WARMUP, (steps - step) / (steps - WARMUP))
        for group in opt.param_groups:
            group["lr"] = lr * lr_mult * ramp
        idx = order[step * BATCH : (step + 1) * BATCH].to(scale.DEV)
        loss = F.cross_entropy(
            task.logits(x[idx], mask[idx]), y[idx], label_smoothing=smoothing
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()


@torch.no_grad()
def score(task, items):
    """Per item: 1 if the top label word is the right one, and the loss."""
    task.model.eval()
    x, mask, y = items
    correct, losses = [], []
    for at in range(0, len(y), 250):
        cut = slice(at, at + 250)
        out = task.logits(x[cut], mask[cut])
        correct += (out.argmax(1) == y[cut]).int().tolist()
        losses += [
            round(v, 4) for v in F.cross_entropy(out, y[cut], reduction="none").tolist()
        ]
    return correct, losses


def run_one(task, lr, arm, seed, split="test"):
    """Fine-tune one arm and score it on `split`, asked with the wording the arm was trained on."""
    finetune(task, seed, lr, **ARMS[arm])
    correct, losses = score(task, getattr(task, split)[ARMS[arm].get("prompt", 0)])
    return dict(arm=arm, seed=seed, correct="".join(map(str, correct)), loss=losses)


def accuracy(run):
    return run["correct"].count("1") / len(run["correct"])


def config(name, **extra):
    return dict(
        task=name,
        model=MODEL,
        device=scale.DEV,
        exact=scale.EXACT,
        gpu=torch.cuda.get_device_name() if scale.DEV == "cuda" else None,
        torch=torch.__version__,
        **extra,
    )


def tune(task, known):
    """The control LR: highest mean dev accuracy over TUNE_SEEDS. The test items are not read.

    `known` holds scores from an earlier, narrower grid; runs replay exactly, so they are kept and not repeated.
    """
    scores = {float(lr): score for lr, score in known.items()}
    for lr in LRS:
        if lr not in scores:
            runs = [run_one(task, lr, "control", seed, "dev") for seed in TUNE_SEEDS]
            scores[lr] = statistics.mean(map(accuracy, runs))
        print(f"tune lr {lr}: {scores[lr]:.4f}", flush=True)
    return dict(
        lr=max(scores, key=scores.get), scores={str(k): v for k, v in scores.items()}
    )


def collect(name, tuned, seeds, out_name):
    task = Task(name)
    partial = f"{OUT}/{out_name}_partial.json"
    out = dict(config=config(name, tuned=tuned, seeds=tuple(seeds)), runs=[])
    if os.path.exists(partial):
        out["runs"] = json.load(open(partial))["runs"]
    done = {(r["arm"], r["seed"]) for r in out["runs"]}
    start = time.time()
    # seed-major order, so a partial file always holds complete seeds
    for seed in seeds:
        for arm in ARMS:
            if (arm, seed) in done:
                continue
            out["runs"].append(run_one(task, tuned["lr"], arm, seed))
            print(
                f"[{time.time() - start:6.0f}s] {name} {arm:15s} seed {seed}: "
                f"accuracy {accuracy(out['runs'][-1]):.4f}",
                flush=True,
            )
            json.dump(out, open(partial, "w"))
    json.dump(out, open(f"{OUT}/{out_name}.json", "w"))


if __name__ == "__main__":
    stage, name = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    tuned_path = f"{OUT}/tune_{name}.json"
    if stage == "speed":
        task = Task(name)
        start = time.time()
        first = run_one(task, LRS[2], "control", 1)
        took = time.time() - start
        again = run_one(task, LRS[2], "control", 1)
        other = run_one(task, LRS[2], "control", 2)
        print(
            f"{name}: {took:.1f}s per run, width {task.train[0][0].shape[1]} tokens, accuracy {accuracy(first):.4f}; "
            f"same seed replays exactly: {first == again}; another seed differs: {first['loss'] != other['loss']}"
        )
    elif stage == "tune":
        known = (
            json.load(open(tuned_path))["scores"] if os.path.exists(tuned_path) else {}
        )
        json.dump(
            dict(config=config(name), **tune(Task(name), known)), open(tuned_path, "w")
        )
    else:
        assert scale.EXACT or scale.DEV != "cuda", "run with REGORA_EXACT=1"
        seeds = dict(pilot=PILOT_SEEDS, run=TEST_SEEDS)[stage]
        prefix = dict(pilot="pilot", run="e12")[stage]
        collect(name, json.load(open(tuned_path)), seeds, f"{prefix}_{name}")
