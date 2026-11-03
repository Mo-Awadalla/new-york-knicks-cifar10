"""Train and evaluate a single CIFAR-10 wide residual network."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.torch_version import TorchVersion

import knicks_methods
from knicks_methods import OPTIMIZERS, POLICIES, Distillation, Rngs, Teacher, TrainingMethod, augment, method_record, normalize


ARCHS = {"wrn16_4": (16, 4, 2_748_890), "wrn22_4": (22, 4, 4_298_970), "wrn28_10": (28, 10, 36_479_194)}
# Teacher-only architectures exceed the submission cap; they may only supervise distillation.
TEACHER_ARCHS = frozenset({"wrn28_10"})
MAX_PARAMS = 5_000_000
DEFAULT_DEADLINE = 7 * 60 * 60
SOURCE_MODULES = ("knicks_cifar.py", "knicks_methods.py")
SWA_LR = 0.01


@dataclass(frozen=True)
class Manifest:
    ids: tuple[int, ...]
    files: tuple[str, ...]
    labels: tuple[int, ...] | None


@dataclass(frozen=True)
class Split:
    train_rows: tuple[int, ...]
    val_rows: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class RunConfig:
    arch: str = "wrn16_4"
    aug: str = "basic"
    split: str = "val"
    epochs: int = 200
    seed: int = 0
    batch: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    wd: float = 0.0005
    warmup: int = 5
    strategy: Literal["cosine", "swa"] = "cosine"
    optimizer: Literal["sgd", "sam"] = "sgd"
    teachers: tuple[tuple[str, str], ...] = ()
    kd_alpha: float = 0.9
    kd_temperature: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "teachers", tuple(tuple(item) for item in self.teachers))

    def validate(self) -> None:
        if self.arch not in ARCHS or self.aug not in POLICIES:
            raise ValueError("unknown architecture or augmentation")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError("unknown optimizer")
        if sum((self.aug in {"augmix_jsd", "cutmix"}, self.optimizer == "sam", self.strategy == "swa")) > 1:
            raise ValueError("portfolio treatments cannot be combined")
        if self.split not in {"val", "all"} or self.epochs < 1 or self.batch < 1:
            raise ValueError("invalid split, epoch count, or batch size")
        if self.strategy not in {"cosine", "swa"} or (self.strategy == "swa" and self.epochs < 2):
            raise ValueError("invalid training strategy or SWA epoch count")
        if not 0 <= self.seed < 2**32 or self.lr <= 0 or not 0 < self.momentum < 1 or self.wd < 0 or self.warmup < 0:
            raise ValueError("invalid optimization configuration")
        if not all(math.isfinite(x) for x in (self.lr, self.momentum, self.wd)):
            raise ValueError("optimizer values must be finite")
        if self.teachers:
            if self.optimizer != "sgd" or self.strategy != "cosine" or self.aug == "augmix_jsd":
                raise ValueError("distillation requires SGD, the cosine schedule, and a single-view augmentation")
            if not all(math.isfinite(x) for x in (self.kd_alpha, self.kd_temperature)) or not 0 < self.kd_alpha <= 1 or self.kd_temperature <= 0:
                raise ValueError("invalid distillation weight or temperature")
            if any(len(item) != 2 or not re.fullmatch(r"[A-Za-z0-9._-]+\.pt", item[0]) or not re.fullmatch(r"[0-9a-f]{64}", item[1]) for item in self.teachers):
                raise ValueError("teachers must be [checkpoint file name, sha256] pairs")
            if len({name for name, _ in self.teachers}) != len(self.teachers):
                raise ValueError("teacher file names must be unique")
        elif (self.kd_alpha, self.kd_temperature) != (0.9, 4.0):
            raise ValueError("distillation settings require teachers")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def find_data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "train_labels.csv").is_file():
            raise ValueError(f"no train_labels.csv in {root}")
        return root
    roots = sorted(p.parent for p in Path("/kaggle/input").rglob("train_labels.csv"))
    if len(roots) != 1:
        raise ValueError(f"expected one competition root under /kaggle/input, found {len(roots)}")
    return roots[0]


def read_rows(path: Path, header: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            raise ValueError(f"{path}: expected header {header}, got {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def parse_id(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"invalid integer id: {value!r}")
    return int(value)


def read_manifest(path: Path, labeled: bool | None = None) -> Manifest:
    if labeled is None:
        labeled = path.name == "train_labels.csv"
    header = ["id", "filename"] + (["label"] if labeled else [])
    rows = read_rows(path, header)
    ids = tuple(parse_id(row["id"]) for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    files = tuple(row["filename"] for row in rows)
    for name in files:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or not name.lower().endswith(".png"):
            raise ValueError(f"{path}: filename must be a PNG basename: {name!r}")
    if len(set(files)) != len(files):
        raise ValueError(f"{path}: duplicate filenames")
    labels = None
    if labeled:
        labels = tuple(parse_id(row["label"]) for row in rows)
        if any(label > 9 for label in labels):
            raise ValueError(f"{path}: label outside 0..9")
    return Manifest(ids, files, labels)


def sample_ids(path: Path) -> tuple[int, ...]:
    rows = read_rows(path, ["id", "label"])
    ids = tuple(parse_id(row["id"]) for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    return ids


def official_manifests(root: Path) -> tuple[Manifest, Manifest, tuple[int, ...]]:
    train = read_manifest(root / "train_labels.csv", True)
    test = read_manifest(root / "test.csv", False)
    sample = sample_ids(root / "sample_submission.csv")
    if len(train.ids) != 50_000 or train.labels is None or any(train.labels.count(c) != 5_000 for c in range(10)):
        raise ValueError("training manifest must contain 50,000 rows with 5,000 per class")
    if len(test.ids) != 2_000 or len(sample) != 2_000 or set(test.ids) != set(sample):
        raise ValueError("test and sample must contain the same 2,000 unique IDs")
    if set(train.files) & set(test.files):
        raise ValueError("train and test filenames overlap")
    return train, test, sample


def stratified_split(manifest: Manifest, split: str = "val") -> Split:
    if manifest.labels is None or len(manifest.ids) != 50_000:
        raise ValueError("split requires the official 50,000 labeled rows")
    if split not in {"val", "all"}:
        raise ValueError("split must be val or all")
    selected = []
    for label in range(10):
        rows = [i for i, value in enumerate(manifest.labels) if value == label]
        if len(rows) != 5_000:
            raise ValueError("split requires 5,000 rows per class")
        rows.sort(key=lambda i: (hashlib.sha256(f"knicks-v1:{manifest.ids[i]}".encode()).digest(), manifest.ids[i]))
        selected.extend(rows[:500])
    val_set = set(selected)
    fingerprint = hashlib.sha256("\n".join(str(x) for x in sorted(manifest.ids[i] for i in selected)).encode()).hexdigest()
    if split == "all":
        return Split(tuple(range(len(manifest.ids))), (), fingerprint)
    return Split(tuple(i for i in range(len(manifest.ids)) if i not in val_set), tuple(sorted(selected)), fingerprint)


def image_index(root: Path) -> dict[str, Path]:
    index = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"image outside data root or symlink: {path}")
            if path.name in index:
                raise ValueError(f"duplicate PNG basename: {path.name}")
            index[path.name] = path
    return index


def decode(manifest: Manifest, rows: tuple[int, ...], index: dict[str, Path], device: torch.device) -> torch.Tensor:
    images = torch.empty((len(rows), 3, 32, 32), dtype=torch.uint8)
    for dest, row in enumerate(rows):
        name = manifest.files[row]
        if name not in index:
            raise ValueError(f"missing image: {name}")
        with Image.open(index[name]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(f"expected 32x32 RGB PNG: {index[name]}")
            images[dest] = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)
    return images.to(device)


def channel_stats(images: torch.Tensor, rows: tuple[int, ...], batch: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(total)
    for start in range(0, len(rows), batch):
        idx = torch.tensor(rows[start:start + batch], device=images.device)
        x = images[idx].to(torch.float32).div_(255)
        total += x.sum(dim=(0, 2, 3)).cpu().double()
        squares += (x * x).sum(dim=(0, 2, 3)).cpu().double()
    count = len(rows) * 32 * 32
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    if (std <= 0).any():
        raise ValueError("training images contain a constant channel")
    return mean.to(device=images.device, dtype=torch.float32), std.to(device=images.device, dtype=torch.float32)


class Block(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(incoming, outgoing, 1, stride, bias=False) if incoming != outgoing or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.bn1(x))
        shortcut = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(activated)
        return shortcut + self.conv2(F.relu(self.bn2(self.conv1(activated))))


class WideResNet(nn.Module):
    def __init__(self, depth: int, width: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        layers = []
        incoming = 16
        for group, outgoing in enumerate((16 * width, 32 * width, 64 * width)):
            for block in range((depth - 4) // 6):
                stride = 2 if group > 0 and block == 0 else 1
                layers.append(Block(incoming, outgoing, stride))
                incoming = outgoing
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(incoming)
        self.head = nn.Linear(incoming, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.blocks(self.stem(x))))
        return self.head(x.mean(dim=(2, 3)))


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(arch: str) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture: {arch}")
    depth, width, expected = ARCHS[arch]
    model = WideResNet(depth, width)
    count = trainable_params(model)
    if count != expected or (count > MAX_PARAMS and arch not in TEACHER_ARCHS):
        raise ValueError(f"invalid trainable parameter count {count}, expected {expected}, cap {MAX_PARAMS}")
    return model


def device_for(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_model(path: Path, device: torch.device) -> tuple[dict, nn.Module, torch.Tensor, torch.Tensor]:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    checkpoint = load_checkpoint(path, device)
    model = build(checkpoint["config"]["arch"]).to(device)
    if trainable_params(model) != checkpoint["params"]:
        raise ValueError("checkpoint parameter count mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model, torch.tensor(checkpoint["mean"], device=device), torch.tensor(checkpoint["std"], device=device)


def load_distillation(cfg: RunConfig, split_sha256: str, teacher_dir: Path | None, device: torch.device) -> tuple[Distillation | None, list[dict]]:
    if not cfg.teachers:
        return None, []
    search = teacher_dir.resolve() if teacher_dir is not None else Path("/kaggle/input")
    teachers, records = [], []
    for name, expected in cfg.teachers:
        matches = sorted(search.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"expected one teacher {name} under {search}, found {len(matches)}")
        if sha256_file(matches[0]) != expected:
            raise ValueError(f"teacher {name} sha256 mismatch")
        checkpoint, model, mean, std = checkpoint_model(matches[0], device)
        teacher_cfg = checkpoint["config"]
        if checkpoint["completed_epochs"] != teacher_cfg["epochs"]:
            raise ValueError(f"teacher {name} did not complete its configured epochs")
        # A teacher that saw validation rows would leak their labels into the student.
        if cfg.split == "val" and (teacher_cfg["split"] != "val" or checkpoint["split_sha256"] != split_sha256):
            raise ValueError(f"teacher {name} was not trained on the student's training rows only")
        model.requires_grad_(False)
        teachers.append(Teacher(model, mean, std))
        records.append({"file": name, "sha256": expected, "arch": teacher_cfg["arch"], "aug": teacher_cfg["aug"], "optimizer": teacher_cfg.get("optimizer", "sgd"), "split": teacher_cfg["split"], "epochs": teacher_cfg["epochs"], "params": checkpoint["params"]})
    return Distillation(tuple(teachers), cfg.kd_alpha, cfg.kd_temperature), records


def distillation_record(cfg: RunConfig, records: list[dict], train_loss_meaning: str) -> dict:
    if not records:
        return {}
    return {"distillation_teachers": records, "kd_alpha": cfg.kd_alpha, "kd_temperature": cfg.kd_temperature, "train_loss_meaning": f"kd_weighted_{train_loss_meaning}"}


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor | None, rows: tuple[int, ...], mean: torch.Tensor, std: torch.Tensor, batch: int, deadline: float | None = None) -> tuple[list[int], float | None, int | None]:
    predictions = []
    total_loss = 0.0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("deadline expired during evaluation")
            idx = torch.tensor(rows[start:start + batch], device=images.device)
            logits = model(normalize(images[idx], mean, std))
            pred = logits.argmax(1)
            predictions.extend(pred.cpu().tolist())
            if labels is not None:
                y = labels[idx]
                total_loss += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (pred == y).sum().item()
    return predictions, total_loss / len(rows) if labels is not None else None, correct if labels is not None else None


def input_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in ("train_labels.csv", "test.csv", "sample_submission.csv")}


def source_hashes() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    if Path(knicks_methods.__file__).resolve() != source_dir / "knicks_methods.py":
        raise RuntimeError("knicks_methods must be loaded from the trainer's source directory")
    return {name: sha256_file(source_dir / name) for name in SOURCE_MODULES}


def source_digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_sha256() -> str:
    return source_digest(source_hashes())


def provenance(root: Path, device: torch.device) -> dict:
    hashes = source_hashes()
    digest = source_digest(hashes)
    return {"csv_sha256": input_hashes(root), "source_sha256": digest, "source_sha256_by_file": hashes, "numpy": np.__version__, "pillow": PIL.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def write_split(path: Path, split: Split, manifest: Manifest) -> None:
    json_write(path, {"train_ids": [manifest.ids[i] for i in split.train_rows], "val_ids": [manifest.ids[i] for i in split.val_rows], "validation_sha256": split.sha256})


def config_from(path: Path) -> RunConfig:
    cfg = RunConfig(**json.loads(path.read_text()))
    cfg.validate()
    return cfg


def stream_seed(cfg: RunConfig, offset: int) -> int:
    return cfg.seed * 16 + offset


def swa_start(cfg: RunConfig) -> int | None:
    return 3 * cfg.epochs // 4 if cfg.strategy == "swa" else None


def learning_rate(cfg: RunConfig, step: int, steps_per_epoch: int) -> float:
    start = swa_start(cfg)
    if start is not None and step >= start * steps_per_epoch:
        return SWA_LR
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = min(cfg.warmup * steps_per_epoch, total_steps)
    if warmup_steps and step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def refresh_batchnorm(model: nn.Module, images: torch.Tensor, rows: tuple[int, ...], cfg: RunConfig, mean: torch.Tensor, std: torch.Tensor, deadline: float) -> int:
    order_rng = torch.Generator().manual_seed(stream_seed(cfg, 4))
    aug_rng = torch.Generator().manual_seed(stream_seed(cfg, 5))
    row_tensor = torch.tensor(rows, dtype=torch.long, device=images.device)
    batches = 0
    momenta = {module: module.momentum for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}

    def inputs():
        nonlocal batches
        permutation = torch.randperm(len(rows), generator=order_rng).to(images.device)
        for start in range(0, len(rows), cfg.batch):
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline expired during batchnorm refresh")
            idx = row_tensor[permutation[start:start + cfg.batch]]
            batches += 1
            yield augment(images[idx], cfg.aug, mean, std, aug_rng, aug_rng)
        synchronize(images.device)
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline expired during batchnorm refresh")

    devices = [images.device.index if images.device.index is not None else torch.cuda.current_device()] if images.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(stream_seed(cfg, 5))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(stream_seed(cfg, 5))
            update_bn(inputs(), model)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.eval()
    return batches


def make_checkpoint(cfg: RunConfig, model: nn.Module, mean: torch.Tensor, std: torch.Tensor, split: Split, info: dict, kind: str, epoch: int) -> dict:
    return {"config": asdict(cfg), "state_dict": model.state_dict(), "mean": mean.cpu().tolist(), "std": std.cpu().tolist(), "params": trainable_params(model), "split_sha256": split.sha256, "provenance": info, "kind": kind, "completed_epochs": epoch}


def configure_training_determinism(cfg: RunConfig, device: torch.device) -> None:
    if device.type == "cuda":
        torch.use_deterministic_algorithms(cfg.aug != "randaugment_cutout")


def train(cfg: RunConfig, root: Path, out: Path, device: torch.device, max_seconds: float, kind: str = "train", teacher_dir: Path | None = None) -> None:
    cfg.validate()
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")
    if device.type == "cuda" and (not re.match(r"^\d+\.\d+\.\d+", str(torch.__version__)) or TorchVersion(str(torch.__version__)) < "2.11.0"):
        raise RuntimeError(f"CUDA training requires PyTorch 2.11 or newer for deterministic algorithms; found {torch.__version__}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + max_seconds
    status = {"status": "incomplete", "kind": kind, "candidate_eligible": False, "completed_epochs": 0, "processed_batches": 0, "processed_images": 0, "model_views": 0, "optimizer_updates": 0, "amp_skips": 0, "sam_ascent_skips": 0, "schedule_unit": "processed_batches", "current_epoch_rows": 0, "max_seconds": max_seconds}
    start_epoch = swa_start(cfg)
    status.update({"strategy": cfg.strategy, "swa_start_epoch_index": start_epoch, "swa_lr": SWA_LR if start_epoch is not None else None, "swa_snapshot_first_epoch_index": None, "swa_snapshot_last_epoch_index": None, "averaged_snapshots": 0, "bn_refresh_policy": "train_rows_augmented" if start_epoch is not None else None, "bn_refresh_batches": 0, "bn_refresh_seconds": 0.0, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "final_val_correct": None, "final_val_loss": None, "final_val_accuracy": None})
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    json_write(out / "config.json", asdict(cfg))
    json_write(out / "manifest.json", status)
    try:
        manifest, test, _ = official_manifests(root)
        split = stratified_split(manifest, cfg.split)
        write_split(out / "split.json", split, manifest)
        info = provenance(root, device)
        info.update(method_record(cfg.aug, cfg.optimizer))
        status.update(info)
        status.update({"train_rows": len(split.train_rows), "val_rows": len(split.val_rows), "test_rows": len(test.ids), "split_sha256": split.sha256})
        index = image_index(root)
        images = decode(manifest, tuple(range(len(manifest.ids))), index, device)
        mean, std = channel_stats(images, split.train_rows)
        labels = torch.tensor(manifest.labels, dtype=torch.long, device=device)
        configure_training_determinism(cfg, device)
        status["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        # Teachers are built before seeding so the student's initialization stream is unchanged.
        distillation, teacher_records = load_distillation(cfg, split.sha256, teacher_dir, device)
        info.update(distillation_record(cfg, teacher_records, info["train_loss_meaning"]))
        status.update(info)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = build(cfg.arch).to(device)
        average = AveragedModel(model, use_buffers=False) if start_epoch is not None else None
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd, nesterov=True)
        order_rng = torch.Generator().manual_seed(stream_seed(cfg, 1))
        rngs = Rngs(crop=torch.Generator().manual_seed(stream_seed(cfg, 2)), cutout=torch.Generator().manual_seed(stream_seed(cfg, 3)), method=np.random.default_rng(stream_seed(cfg, 6)))
        method = TrainingMethod(cfg.aug, cfg.optimizer, model=model, optimizer=optimizer, mean=mean, std=std, rngs=rngs, amp=device.type == "cuda", deadline=deadline, distillation=distillation)
        rows = torch.tensor(split.train_rows, dtype=torch.long, device=device)
        steps_per_epoch = math.ceil(len(rows) / cfg.batch)
        step = 0
        with (out / "metrics.jsonl").open("w") as metrics:
            for epoch in range(cfg.epochs):
                synchronize(device)
                epoch_start = time.monotonic()
                model.train()
                permutation = torch.randperm(len(rows), generator=order_rng).to(device)
                running_loss = 0.0
                correct = 0
                seen = 0
                status["current_epoch"] = epoch + 1
                status["current_epoch_rows"] = 0
                for start in range(0, len(rows), cfg.batch):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deadline expired during training")
                    idx = rows[permutation[start:start + cfg.batch]]
                    lr = learning_rate(cfg, step, steps_per_epoch)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    outcome = method.step(images[idx], labels[idx])
                    status["processed_images"] += outcome.rows
                    status["model_views"] += outcome.forward_images
                    status["amp_skips"] += int(outcome.update != "applied")
                    status["sam_ascent_skips"] += int(outcome.update == "ascent_skip")
                    status["optimizer_updates"] += int(outcome.update == "applied")
                    running_loss += outcome.loss * outcome.rows
                    correct += outcome.correct
                    seen += outcome.rows
                    step += 1
                    status["processed_batches"] = step
                    status["current_epoch_rows"] = seen
                if average is not None and epoch >= start_epoch:
                    average.update_parameters(model)
                    status["averaged_snapshots"] = int(average.n_averaged.item())
                    if status["swa_snapshot_first_epoch_index"] is None:
                        status["swa_snapshot_first_epoch_index"] = epoch
                    status["swa_snapshot_last_epoch_index"] = epoch
                predictions, val_loss, val_correct = ([], None, None)
                if split.val_rows:
                    predictions, val_loss, val_correct = evaluate(model, images, labels, split.val_rows, mean, std, cfg.batch, deadline)
                synchronize(device)
                record = {"epoch": epoch + 1, "lr": lr, "train_loss": running_loss / seen, "train_correct": correct, "train_rows": seen, "train_accuracy": correct / seen, "val_loss": val_loss, "val_correct": val_correct, "val_rows": len(split.val_rows), "val_accuracy": val_correct / len(split.val_rows) if val_correct is not None else None, "seconds": time.monotonic() - epoch_start}
                if cfg.aug == "cutmix":
                    record["train_primary_target_correct"] = record.pop("train_correct")
                    record["train_primary_target_accuracy"] = record.pop("train_accuracy")
                record.update({key: status[key] for key in ("processed_batches", "processed_images", "model_views", "optimizer_updates", "amp_skips", "sam_ascent_skips", "schedule_unit", "averaged_snapshots")})
                record["train_accuracy_meaning"] = info["train_accuracy_meaning"]
                record["train_loss_meaning"] = info["train_loss_meaning"]
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
                status["completed_epochs"] = epoch + 1
        final_model = model
        if average is not None:
            if status["averaged_snapshots"] != cfg.epochs - start_epoch:
                raise ValueError("SWA snapshot count does not match the completed tail")
            final_model = average.module
            synchronize(device)
            refresh_start = time.monotonic()
            try:
                status["bn_refresh_batches"] = refresh_batchnorm(final_model, images, split.train_rows, cfg, mean, std, deadline)
            finally:
                synchronize(device)
                status["bn_refresh_seconds"] = time.monotonic() - refresh_start
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline expired before checkpoint")
        info.update({key: status[key] for key in ("strategy", "swa_start_epoch_index", "swa_lr", "swa_snapshot_first_epoch_index", "swa_snapshot_last_epoch_index", "averaged_snapshots", "bn_refresh_policy", "bn_refresh_batches", "bn_refresh_seconds", "deterministic_algorithms")})
        checkpoint_path = out / "checkpoint.pt"
        torch.save(make_checkpoint(cfg, final_model, mean, std, split, info, kind, cfg.epochs), checkpoint_path)
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline expired after checkpoint")
        _, restored, saved_mean, saved_std = checkpoint_model(checkpoint_path, device)
        if split.val_rows:
            predictions, val_loss, val_correct = evaluate(restored, images, labels, split.val_rows, saved_mean, saved_std, cfg.batch, deadline)
            np.save(out / "val_pred.npy", np.asarray(predictions, dtype=np.int8))
            status.update({"final_val_correct": val_correct, "final_val_loss": val_loss, "final_val_accuracy": val_correct / len(split.val_rows)})
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline expired after checkpoint validation")
        status.update({"status": "complete", "checkpoint_sha256": sha256_file(checkpoint_path), "params": trainable_params(final_model), "mean": mean.cpu().tolist(), "std": std.cpu().tolist(), "candidate_eligible": kind == "train" and cfg.arch not in TEACHER_ARCHS})
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if device.type == "cuda":
            status["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
            torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
        status["elapsed_seconds"] = time.monotonic() - started
        json_write(out / "manifest.json", status)


def write_predictions(path: Path, ids: tuple[int, ...], predictions: list[int]) -> None:
    if len(ids) != len(predictions):
        raise ValueError("prediction count mismatch")
    with path.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, predictions))


def check_submission(path: Path, sample: Path) -> None:
    rows = read_rows(path, ["id", "label"])
    ids = tuple(parse_id(row["id"]) for row in rows)
    labels = [parse_id(row["label"]) for row in rows]
    if ids != sample_ids(sample) or any(label > 9 for label in labels):
        raise ValueError("submission IDs/order or labels do not match the sample")


def predict(checkpoint_path: Path, root: Path, out: Path, device: torch.device) -> None:
    checkpoint, model, mean, std = checkpoint_model(checkpoint_path, device)
    cfg = RunConfig(**checkpoint["config"])
    cfg.validate()
    run_manifest_path = checkpoint_path.parent / "manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError("checkpoint requires its completed run manifest")
    run_manifest = json.loads(run_manifest_path.read_text())
    if cfg.arch in TEACHER_ARCHS or checkpoint["params"] > MAX_PARAMS:
        raise ValueError("teacher-only architectures exceed the parameter cap and cannot create a submission")
    if checkpoint["kind"] != "train" or checkpoint["completed_epochs"] != cfg.epochs or run_manifest.get("status") != "complete" or run_manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("only a completed production training checkpoint can create a submission")
    train_manifest, test, sample = official_manifests(root)
    if checkpoint["provenance"]["csv_sha256"] != input_hashes(root) or checkpoint["split_sha256"] != stratified_split(train_manifest, cfg.split).sha256:
        raise ValueError("dataset manifests or split differ from the checkpoint")
    images = decode(test, tuple(range(len(test.ids))), image_index(root), device)
    predictions, _, _ = evaluate(model, images, None, tuple(range(len(test.ids))), mean, std, cfg.batch)
    by_id = dict(zip(test.ids, predictions))
    write_predictions(out, sample, [by_id[i] for i in sample])
    check_submission(out, root / "sample_submission.csv")


def smoke(root: Path, out: Path, device: torch.device, cfg: RunConfig, teacher_dir: Path | None = None) -> None:
    cfg.validate()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "kind": "smoke", "candidate_eligible": False}
    json_write(out / "manifest.json", status)
    try:
        train_manifest, test_manifest, sample = official_manifests(root)
        index = image_index(root)
        train_rows = tuple(i for i, name in enumerate(train_manifest.files) if name in index)
        test_rows = tuple(i for i, name in enumerate(test_manifest.files) if name in index)
        if len(train_rows) < 2 or len(test_rows) < 1:
            raise ValueError("smoke requires at least two available training images and one test image")
        train_rows = train_rows[:2]
        test_rows = test_rows[:2]
        images = decode(train_manifest, train_rows, index, device)
        test_images = decode(test_manifest, test_rows, index, device)
        mean, std = channel_stats(images, tuple(range(len(train_rows))))
        labels = torch.tensor([train_manifest.labels[i] for i in train_rows], device=device)
        distillation, teacher_records = load_distillation(cfg, stratified_split(train_manifest, cfg.split).sha256, teacher_dir, device)
        torch.manual_seed(cfg.seed)
        model = build(cfg.arch).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        x = normalize(images, mean, std)
        model.train()
        with torch.no_grad():
            initial_loss = F.cross_entropy(model(x), labels).item()
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_loss = F.cross_entropy(model(x), labels).item()
        if not final_loss < initial_loss:
            raise ValueError(f"smoke overfit did not reduce training loss: {initial_loss} -> {final_loss}")
        method_optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd, nesterov=True)
        rngs = Rngs(crop=torch.Generator().manual_seed(stream_seed(cfg, 2)), cutout=torch.Generator().manual_seed(stream_seed(cfg, 3)), method=np.random.default_rng(stream_seed(cfg, 6)))
        method = TrainingMethod(cfg.aug, cfg.optimizer, model=model, optimizer=method_optimizer, mean=mean, std=std, rngs=rngs, amp=device.type == "cuda", distillation=distillation)
        outcome = method.step(images, labels)
        model.eval()
        with torch.inference_mode():
            logits = model(x)
        split = stratified_split(train_manifest, "val")
        info = provenance(root, device)
        info.update(method_record(cfg.aug, cfg.optimizer))
        info.update(distillation_record(cfg, teacher_records, info["train_loss_meaning"]))
        json_write(out / "config.json", asdict(cfg))
        write_split(out / "split.json", split, train_manifest)
        checkpoint_path = out / "checkpoint.pt"
        torch.save(make_checkpoint(cfg, model, mean, std, split, info, "smoke", 0), checkpoint_path)
        _, restored, saved_mean, saved_std = checkpoint_model(checkpoint_path, device)
        with torch.inference_mode():
            restored_logits = restored(normalize(images, saved_mean, saved_std))
        if not torch.equal(logits, restored_logits):
            raise ValueError("smoke checkpoint reload changed evaluation logits")
        predictions, _, _ = evaluate(restored, test_images, None, tuple(range(len(test_rows))), saved_mean, saved_std, cfg.batch)
        by_id = {test_manifest.ids[row]: pred for row, pred in zip(test_rows, predictions)}
        partial_ids = tuple(i for i in sample if i in by_id)
        write_predictions(out / "smoke_predictions.csv", partial_ids, [by_id[i] for i in partial_ids])
        status.update(info)
        status.update({"status": "complete", "train_rows": len(train_rows), "test_rows": len(test_rows), "initial_loss": initial_loss, "final_loss": final_loss, "steps": 12, "method_steps": 1, "method_loss": outcome.loss, "method_update": outcome.update, "method_forward_images": outcome.forward_images, "reload_logits_exact": True, "params": trainable_params(model), "split_sha256": split.sha256, "checkpoint_sha256": sha256_file(checkpoint_path)})
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        synchronize(device)
        status["elapsed_seconds"] = time.monotonic() - started
        json_write(out / "manifest.json", status)


def stage(config_path: Path, run_id: str, out: Path, mode: str, seed: int | None = None, teachers_dataset: str | None = None) -> None:
    cfg = config_from(config_path)
    if seed is not None:
        cfg = replace(cfg, seed=seed)
        cfg.validate()
    if mode not in {"train", "smoke", "calibrate", "compare"}:
        raise ValueError("invalid staging mode")
    if mode == "compare" and cfg.split != "val":
        raise ValueError("comparison staging requires the validation split")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", run_id) or len(run_id) > 42:
        raise ValueError("run-id must be lowercase alphanumeric words separated by hyphens, at most 42 characters")
    if bool(cfg.teachers) != (teachers_dataset is not None) or (teachers_dataset is not None and not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", teachers_dataset)):
        raise ValueError("distillation configs require exactly one owner/slug teacher dataset; other configs take none")
    out.mkdir(parents=True, exist_ok=False)
    slug = f"knicks-{run_id}"
    source_hashes()
    source_dir = Path(__file__).resolve().parent
    sources = {name: (source_dir / name).read_bytes() for name in SOURCE_MODULES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    digest = source_digest(hashes)
    spec = {"slug": slug, "mode": mode, "config": asdict(cfg), "source_sha256": digest}
    staged = (
        "import sys\n"
        "import tempfile\n"
        "from pathlib import Path\n\n"
        f"SOURCES = {sources!r}\n"
        f"SPEC = {spec!r}\n\n"
        "with tempfile.TemporaryDirectory(prefix='knicks-src-') as source_dir:\n"
        "    for name, data in SOURCES.items():\n"
        "        (Path(source_dir) / name).write_bytes(data)\n"
        "    sys.path.insert(0, source_dir)\n"
        "    import knicks_cifar\n"
        "    knicks_cifar.run_kernel(SPEC, sys.argv[1:])\n"
    )
    (out / "kernel.py").write_text(staged)
    json_write(out / "kernel-metadata.json", {"id": f"moawadalla/{slug}", "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "dataset_sources": [teachers_dataset] if teachers_dataset else [], "competition_sources": ["cifar-10-res-net-architecture-challenge-dl"], "kernel_sources": []})


def run_kernel(spec: dict, argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run a staged CIFAR experiment")
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--teachers", type=Path, help="directory searched for teacher checkpoints; defaults to /kaggle/input")
    args = parser.parse_args(argv)
    if source_sha256() != spec["source_sha256"]:
        raise RuntimeError("staged source hash mismatch")
    args.working.mkdir(parents=True, exist_ok=True)
    config_path = args.working / f"{spec['slug']}-config.json"
    output_path = args.working / f"{spec['slug']}-output"
    with config_path.open("x") as config_file:
        json.dump(spec["config"], config_file, sort_keys=True)
        config_file.write("\n")
    mode = spec["mode"]
    command = ["smoke" if mode == "smoke" else "train", "--config", str(config_path), "--out", str(output_path), "--device", args.device]
    if args.data is not None:
        command += ["--data", str(args.data)]
    if args.teachers is not None:
        command += ["--teachers", str(args.teachers)]
    if mode == "calibrate":
        command += ["--epochs", "3", "--calibrate"]
    elif mode == "compare":
        command += ["--calibrate"]
    main(command)
    if mode == "train":
        prediction = ["predict", "--checkpoint", str(output_path / "checkpoint.pt"), "--out", str(output_path / "submission.csv"), "--device", args.device]
        if args.data is not None:
            prediction += ["--data", str(args.data)]
        main(prediction)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="train on the official full-size dataset")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--data", type=Path)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--split", choices=("val", "all"))
    train_parser.add_argument("--max-seconds", type=float, default=DEFAULT_DEADLINE)
    train_parser.add_argument("--teachers", type=Path, help="directory searched for teacher checkpoints; defaults to /kaggle/input")
    train_parser.add_argument("--calibrate", action="store_true", help="mark this run ineligible for submissions")
    smoke_parser = sub.add_parser("smoke", help="exercise available official images without making a submission")
    smoke_parser.add_argument("--config", type=Path)
    smoke_parser.add_argument("--data", type=Path)
    smoke_parser.add_argument("--out", type=Path, required=True)
    smoke_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    smoke_parser.add_argument("--teachers", type=Path, help="directory searched for teacher checkpoints; defaults to /kaggle/input")
    predict_parser = sub.add_parser("predict", help="create a submission from a completed checkpoint")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--data", type=Path)
    predict_parser.add_argument("--out", type=Path, required=True)
    predict_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    check_parser = sub.add_parser("check", help="validate submission IDs, order, and labels")
    check_parser.add_argument("--submission", type=Path, required=True)
    check_parser.add_argument("--sample", type=Path, required=True)
    stage_parser = sub.add_parser("stage", help="write an offline Kaggle script without launching it")
    stage_parser.add_argument("--config", type=Path, required=True)
    stage_parser.add_argument("--run-id", required=True)
    stage_parser.add_argument("--out", type=Path, required=True)
    stage_parser.add_argument("--mode", choices=("train", "smoke", "calibrate", "compare"), default="train")
    stage_parser.add_argument("--seed", type=int)
    stage_parser.add_argument("--teachers-dataset", help="owner/slug of the private Kaggle dataset holding teacher checkpoints")
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.config, args.run_id, args.out, args.mode, args.seed, args.teachers_dataset)
    elif args.command == "check":
        check_submission(args.submission, args.sample)
        print("submission matches sample IDs and order")
    elif args.command == "smoke":
        smoke(find_data_root(args.data), args.out, device_for(args.device), config_from(args.config) if args.config else RunConfig(), args.teachers)
    elif args.command == "predict":
        predict(args.checkpoint, find_data_root(args.data), args.out, device_for(args.device))
    else:
        cfg = config_from(args.config)
        cfg = replace(cfg, epochs=args.epochs if args.epochs is not None else cfg.epochs, split=args.split if args.split else cfg.split)
        train(cfg, find_data_root(args.data), args.out, device_for(args.device), args.max_seconds, "calibrate" if args.calibrate else "train", args.teachers)


if __name__ == "__main__":
    main()
