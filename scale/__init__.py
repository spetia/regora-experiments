"""Larger models on real datasets: a byte-level GPT on enwik8 and a ResNet-18 on CIFAR-100.

Same contract as `tinygpt`: the batch at a step is a function of (data_seed, step), so branches from one checkpoint
share their data, and a state is a plain dict that can be branched any number of times. On CUDA the forward pass runs
in bfloat16 and nothing forces deterministic kernels, which is how such models are normally trained. Setting
REGORA_EXACT=1 (with CUBLAS_WORKSPACE_CONFIG=:4096:8) forces deterministic kernels instead.
"""

import hashlib
import math
import os
import pickle
import tarfile
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = os.environ.get("REGORA_DEV", "cpu")  # cpu | cuda
DATA_DIR = os.environ.get("REGORA_DATA", "data")
EXACT = os.environ.get("REGORA_EXACT") == "1"
torch.use_deterministic_algorithms(EXACT)
WARMUP = 100  # steps of linear LR warmup; the LR is constant afterwards
EVAL_SEED = 12345
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR_MEAN, CIFAR_STD = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)


def clone(obj):
    """Deep copy of a state dict, so a checkpoint is never aliased by the run that continues it."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone(v) for v in obj]
    return obj


class Task:
    """A model with its data and optimiser settings. Subclasses set the attributes below and the four methods."""

    lrs: tuple  # the grid the healthy LR is tuned over
    weight_decay: float
    weight_decay_alt: float  # a small intervention, for measuring how well pairing resolves small effects
    clip: float | None
    eval_batches: int  # batches in the fixed evaluation set

    def build(self) -> nn.Module:
        raise NotImplementedError

    def optimizer(self, model, lr) -> torch.optim.Optimizer:
        raise NotImplementedError

    def train_batch(self, data_seed, step):
        raise NotImplementedError

    def eval_fixed(self):
        """The fixed evaluation set, as a list of (inputs, targets) batches."""
        raise NotImplementedError

    def eval_fresh(self, rng):
        """One freshly drawn evaluation batch, as a one-element list."""
        raise NotImplementedError

    def loss(self, model, x, y):
        return F.cross_entropy(model(x).flatten(0, -2).float(), y.flatten())

    def new_state(self, seed):
        torch.manual_seed(seed)
        self.model = self.build().to(DEV)
        return dict(model=clone(self.model.state_dict()), opt=None, step=0)


def autocast():
    cuda = DEV == "cuda"
    return torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda)


def train(task, state, data_seed, steps, lr, lr_mult=1.0, weight_decay=None):
    """Continue `state` for `steps` steps and return the new state. `state` itself is left untouched."""
    model = task.model
    model.load_state_dict(state["model"])
    model.train()
    opt = task.optimizer(model, lr)
    if state["opt"] is not None:
        opt.load_state_dict(clone(state["opt"]))
    decay = task.weight_decay if weight_decay is None else weight_decay
    for step in range(state["step"], state["step"] + steps):
        for group in opt.param_groups:
            group["lr"] = lr * lr_mult * min(1.0, (step + 1) / WARMUP)
            group["weight_decay"] = decay  # after load, which restores the saved value
        x, y = task.train_batch(data_seed, step)
        with autocast():
            loss = task.loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if task.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), task.clip)
        opt.step()
    return dict(
        model=clone(model.state_dict()),
        opt=clone(opt.state_dict()),
        step=state["step"] + steps,
    )


@torch.no_grad()
def loss_on(task, state, batches):
    """Mean loss of `state` over `batches`. A diverged model reads as +infinity."""
    model = task.model
    model.load_state_dict(state["model"])
    model.eval()
    with autocast():
        total = sum(task.loss(model, x, y).item() for x, y in batches)
    mean = total / len(batches)
    return mean if math.isfinite(mean) else math.inf


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = (
            self.qkv(self.ln1(x))
            .view(b, t, 3, self.heads, w // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(b, t, w))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, layers, width, heads, ctx):
        super().__init__()
        self.tokens, self.positions = nn.Embedding(256, width), nn.Embedding(ctx, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        h = self.tokens(x) + self.positions(torch.arange(x.shape[1], device=x.device))
        return self.head(self.ln(self.blocks(h)))


class ByteLM(Task):
    """Next-byte prediction with AdamW. The defaults give a 25M-parameter model."""

    lrs = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3)
    weight_decay, weight_decay_alt, clip = 0.0, 0.1, 1.0

    def __init__(
        self,
        train,
        val,
        layers=8,
        width=512,
        heads=8,
        ctx=256,
        batch=64,
        eval_batches=40,
    ):
        self.train, self.val = train.to(DEV), val.to(DEV)
        self.shape = dict(layers=layers, width=width, heads=heads, ctx=ctx)
        self.ctx, self.batch, self.eval_batches = ctx, batch, eval_batches
        rng = torch.Generator().manual_seed(EVAL_SEED)
        self.fixed = [self.eval_fresh(rng)[0] for _ in range(eval_batches)]

    def build(self):
        return GPT(**self.shape)

    def optimizer(self, model, lr):
        return torch.optim.AdamW(model.parameters(), lr=lr)

    def windows(self, data, rng):
        starts = torch.randint(
            0, len(data) - self.ctx - 1, (self.batch,), generator=rng
        )
        offsets = torch.arange(self.ctx + 1, device=DEV)
        tokens = data[starts.to(DEV)[:, None] + offsets].long()
        return tokens[:, :-1], tokens[:, 1:]

    def train_batch(self, data_seed, step):
        rng = torch.Generator().manual_seed(data_seed * 1_000_003 + step)
        return self.windows(self.train, rng)

    def eval_fixed(self):
        return self.fixed

    def eval_fresh(self, rng):
        return [self.windows(self.val, rng)]


def conv_bn(c_in, c_out, stride=1):
    return [
        nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(c_out),
    ]


class Residual(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(c_in, c_out, stride), nn.ReLU(), *conv_bn(c_out, c_out)
        )
        self.skip = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class ResNet18(nn.Module):
    """The CIFAR form of ResNet-18: a 3x3 stem and no initial pooling."""

    def __init__(self, widths, classes):
        super().__init__()
        layers, c_in = [*conv_bn(3, widths[0]), nn.ReLU()], widths[0]
        for i, width in enumerate(widths):
            layers += [
                Residual(c_in, width, 1 if i == 0 else 2),
                Residual(width, width, 1),
            ]
            c_in = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, classes)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


class Images(Task):
    """Image classification with SGD and momentum. The defaults give an 11M-parameter ResNet-18."""

    lrs = (0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05)
    weight_decay, weight_decay_alt, clip = 5e-4, 1e-3, None
    PAD = 4

    def __init__(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
        widths=(64, 128, 256, 512),
        batch=128,
        eval_batch=500,
    ):
        self.train_x, self.train_y = train_x.to(DEV), train_y.to(DEV)
        self.val_x, self.val_y = val_x.to(DEV), val_y.to(DEV)
        self.widths, self.batch = widths, batch
        self.classes = int(train_y.max()) + 1
        self.mean = torch.tensor(CIFAR_MEAN, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=DEV).view(1, 3, 1, 1)
        self.fixed = [
            (
                self.normalize(self.val_x[i : i + eval_batch]),
                self.val_y[i : i + eval_batch],
            )
            for i in range(0, len(val_x), eval_batch)
        ]
        self.eval_batches = len(self.fixed)

    def build(self):
        model = ResNet18(self.widths, self.classes)
        return model.to(memory_format=torch.channels_last)

    def optimizer(self, model, lr):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    def normalize(self, images):
        images = (images.float() / 255 - self.mean) / self.std
        return images.contiguous(memory_format=torch.channels_last)

    def train_batch(self, data_seed, step):
        """A batch with the standard augmentation: a random crop from the 4-pixel padded image and a random flip."""
        rng = torch.Generator().manual_seed(data_seed * 1_000_003 + step)
        size = self.train_x.shape[-1]
        picks = torch.randint(0, len(self.train_x), (self.batch,), generator=rng).to(
            DEV
        )
        shift = torch.randint(0, 2 * self.PAD + 1, (2, self.batch), generator=rng).to(
            DEV
        )
        flip = (torch.rand(self.batch, generator=rng) < 0.5).to(DEV)
        padded = F.pad(self.train_x[picks], (self.PAD,) * 4)
        span = torch.arange(size, device=DEV)
        rows = (shift[0][:, None] + span)[:, None, :, None]
        cols = (shift[1][:, None] + span)[:, None, None, :]
        cols = torch.where(flip[:, None, None, None], cols.flip(3), cols)
        batch = torch.arange(self.batch, device=DEV)[:, None, None, None]
        channel = torch.arange(3, device=DEV)[None, :, None, None]
        return self.normalize(padded[batch, channel, rows, cols]), self.train_y[picks]

    def eval_fixed(self):
        return self.fixed

    def eval_fresh(self, rng):
        picks = torch.randint(0, len(self.val_x), (self.batch,), generator=rng).to(DEV)
        return [(self.normalize(self.val_x[picks]), self.val_y[picks])]


def download(url, md5=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(
            path + ".part", path
        )  # a partial download never sits under the final name
    if md5 is not None:
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        if digest != md5:
            raise ValueError(f"{path}: md5 {digest}, expected {md5}")
    return path


def load_enwik8():
    """The usual split of the first 100M bytes of Wikipedia: 90M train, 5M validation. The last 5M are not used."""
    with zipfile.ZipFile(download(ENWIK8_URL)) as archive:
        data = torch.frombuffer(bytearray(archive.read("enwik8")), dtype=torch.uint8)
    return data[:90_000_000], data[90_000_000:95_000_000]


def load_cifar100():
    """45,000 training images and 5,000 held out for validation. The official test set is not used."""
    with tarfile.open(download(CIFAR100_URL, CIFAR100_MD5)) as archive:
        # the archive is checked against the published md5 above before it is unpickled
        raw = pickle.load(
            archive.extractfile("cifar-100-python/train"), encoding="bytes"
        )
    images = torch.tensor(raw[b"data"]).view(-1, 3, 32, 32)
    labels = torch.tensor(raw[b"fine_labels"])
    order = torch.randperm(
        len(images), generator=torch.Generator().manual_seed(EVAL_SEED)
    )
    train, val = order[:45_000], order[45_000:]
    return images[train], labels[train], images[val], labels[val]


TASKS = {
    "lm": lambda: ByteLM(*load_enwik8()),
    "vision": lambda: Images(*load_cifar100()),
}
