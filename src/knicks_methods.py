"""Minibatch preparation and optimizer updates for the CIFAR training recipes."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import RandAugment

Policy = Literal["basic", "cutout", "randaugment_cutout", "augmix_jsd", "cutmix"]
OptimizerKind = Literal["sgd", "sam"]
POLICIES = get_args(Policy)
OPTIMIZERS = get_args(OptimizerKind)
RANDAUGMENT = RandAugment(num_ops=2, magnitude=9)
AUGMIX_WIDTH = 3
AUGMIX_SEVERITY = 3
JSD_WEIGHT = 12.0
CUTMIX_PROB = 0.5
SAM_RHO = 0.05


@dataclass(frozen=True)
class Rngs:
    crop: torch.Generator
    cutout: torch.Generator
    method: np.random.Generator


@dataclass(frozen=True)
class HardBatch:
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class MixedBatch:
    x: torch.Tensor
    y_a: torch.Tensor
    y_b: torch.Tensor
    lam: float


@dataclass(frozen=True)
class AugMixBatch:
    x: torch.Tensor
    y: torch.Tensor


TrainBatch = HardBatch | MixedBatch | AugMixBatch


@dataclass(frozen=True)
class StepOutcome:
    loss: float
    correct: int
    rows: int
    forward_images: int
    update: Literal["applied", "amp_skip", "ascent_skip"]


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.float().div_(255) - mean[None, :, None, None]) / std[None, :, None, None]


def _crop_flip(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    n = x.shape[0]
    device = x.device
    offsets = torch.randint(0, 9, (n, 2), generator=generator).to(device)
    flip = torch.rand(n, generator=generator).to(device) < 0.5
    padded = F.pad(x.float(), (4, 4, 4, 4), mode="reflect")
    yy = offsets[:, 0, None, None] + torch.arange(32, device=device)[None, :, None]
    xx = offsets[:, 1, None, None] + torch.arange(32, device=device)[None, None, :]
    cropped = padded[torch.arange(n, device=device)[:, None, None], :, yy, xx].permute(0, 3, 1, 2)
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)


def augment(x: torch.Tensor, aug: str, mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator, cutout_generator: torch.Generator) -> torch.Tensor:
    if aug not in {"basic", "cutout", "randaugment_cutout"}:
        raise ValueError(f"not a single-view augmentation: {aug}")
    n = x.shape[0]
    device = x.device
    cropped = _crop_flip(x, generator)
    if aug == "randaugment_cutout":
        cropped = torch.stack([RANDAUGMENT(image) for image in cropped.to(torch.uint8)])
    result = normalize(cropped, mean, std)
    if aug in {"cutout", "randaugment_cutout"}:
        corners = torch.randint(0, 32, (n, 2), generator=cutout_generator).to(device) - 8
        y = torch.arange(32, device=device)[None, :, None]
        xgrid = torch.arange(32, device=device)[None, None, :]
        mask = (y >= corners[:, 0, None, None]) & (y < corners[:, 0, None, None] + 16)
        mask = mask & (xgrid >= corners[:, 1, None, None]) & (xgrid < corners[:, 1, None, None] + 16)
        result = result.masked_fill(mask[:, None], 0)
    return result


# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Adapted from google-research/augmix augmentations.py and cifar.py. The operators
# take an explicit generator and the image mixing uses the training-set statistics.

def _autocontrast(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return ImageOps.autocontrast(image)


def _equalize(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return ImageOps.equalize(image)


def _posterize(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return ImageOps.posterize(image, 4 - int(rng.uniform(0.1, AUGMIX_SEVERITY) * 4 / 10))


def _rotate(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    degrees = int(rng.uniform(0.1, AUGMIX_SEVERITY) * 30 / 10)
    if rng.uniform() > 0.5:
        degrees = -degrees
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR)


def _solarize(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return ImageOps.solarize(image, 256 - int(rng.uniform(0.1, AUGMIX_SEVERITY) * 256 / 10))


def _shear_x(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    level = rng.uniform(0.1, AUGMIX_SEVERITY) * 0.3 / 10
    if rng.uniform() > 0.5:
        level = -level
    return image.transform((32, 32), Image.Transform.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.Resampling.BILINEAR)


def _shear_y(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    level = rng.uniform(0.1, AUGMIX_SEVERITY) * 0.3 / 10
    if rng.uniform() > 0.5:
        level = -level
    return image.transform((32, 32), Image.Transform.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.Resampling.BILINEAR)


def _translate_x(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    level = int(rng.uniform(0.1, AUGMIX_SEVERITY) * (32 / 3) / 10)
    if rng.random() > 0.5:
        level = -level
    return image.transform((32, 32), Image.Transform.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.Resampling.BILINEAR)


def _translate_y(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    level = int(rng.uniform(0.1, AUGMIX_SEVERITY) * (32 / 3) / 10)
    if rng.random() > 0.5:
        level = -level
    return image.transform((32, 32), Image.Transform.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.Resampling.BILINEAR)


AUGMIX_OPS = (_autocontrast, _equalize, _posterize, _rotate, _solarize, _shear_x, _shear_y, _translate_x, _translate_y)


def _augmix_view(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet([1.0] * AUGMIX_WIDTH).astype(np.float32)
    mixing = np.float32(rng.beta(1.0, 1.0))
    pil = Image.fromarray(image)
    mixture = np.zeros(image.shape, dtype=np.float32)
    for weight in weights:
        chain = pil.copy()
        for _ in range(int(rng.integers(1, 4))):
            chain = AUGMIX_OPS[int(rng.integers(len(AUGMIX_OPS)))](chain, rng)
        mixture += weight * np.asarray(chain, dtype=np.float32)
    return (1 - mixing) * image.astype(np.float32) + mixing * mixture


def training_batch(policy: Policy, images_u8: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, rngs: Rngs) -> TrainBatch:
    if policy in {"basic", "cutout", "randaugment_cutout"}:
        return HardBatch(augment(images_u8, policy, mean, std, rngs.crop, rngs.cutout), labels)
    if policy not in {"augmix_jsd", "cutmix"}:
        raise ValueError(f"unknown augmentation: {policy}")
    cropped = _crop_flip(images_u8, rngs.crop)
    if policy == "augmix_jsd":
        arrays = cropped.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        views = [[], []]
        for image in arrays:
            for view in views:
                view.append(_augmix_view(image, rngs.method))
        mixed = [torch.from_numpy(np.stack(view)).permute(0, 3, 1, 2).to(images_u8.device) for view in views]
        return AugMixBatch(torch.cat([normalize(cropped, mean, std)] + [normalize(view, mean, std) for view in mixed]), labels)
    x = normalize(cropped, mean, std)
    if rngs.method.random() >= CUTMIX_PROB:
        return HardBatch(x, labels)
    # Adapted from clovaai/CutMix-PyTorch train.py. The probability is 0.5,
    # and the label weight uses the clipped area, not the sampled beta value.
    # Copyright (c) 2019-present NAVER Corp.
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    lam = rngs.method.beta(1.0, 1.0)
    permutation = torch.as_tensor(rngs.method.permutation(len(labels)), device=labels.device, dtype=torch.long)
    size = int(32 * math.sqrt(1.0 - lam))
    cx, cy = (int(v) for v in rngs.method.integers(0, 32, size=2))
    x1, x2 = max(0, cx - size // 2), min(32, cx + size // 2)
    y1, y2 = max(0, cy - size // 2), min(32, cy + size // 2)
    x[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    return MixedBatch(x, labels, labels[permutation], 1.0 - (x2 - x1) * (y2 - y1) / 1024.0)


def objective(batch: TrainBatch, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, AugMixBatch):
        clean, view1, view2 = logits.split(len(batch.y))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            clean, view1, view2 = clean.float(), view1.float(), view2.float()
            probabilities = [F.softmax(view, dim=1) for view in (clean, view1, view2)]
            log_mean = (sum(probabilities) / 3).clamp(1e-7, 1).log()
            divergence = sum((p * (F.log_softmax(view, dim=1) - log_mean)).sum(dim=1).mean() for p, view in zip(probabilities, (clean, view1, view2))) / 3
            loss = F.cross_entropy(clean, batch.y) + JSD_WEIGHT * divergence
        return loss, (clean.argmax(1) == batch.y).sum()
    if isinstance(batch, MixedBatch):
        loss = batch.lam * F.cross_entropy(logits, batch.y_a) + (1 - batch.lam) * F.cross_entropy(logits, batch.y_b)
        return loss, (logits.argmax(1) == batch.y_a).sum()
    if isinstance(batch, HardBatch):
        return F.cross_entropy(logits, batch.y), (logits.argmax(1) == batch.y).sum()
    raise TypeError(f"unknown batch type: {type(batch).__name__}")


def method_record(policy: Policy, optimizer_kind: OptimizerKind) -> dict:
    record = {"augmentation": policy, "optimizer": optimizer_kind, "train_accuracy_meaning": "primary_target_diagnostic" if policy == "cutmix" else "clean_target_accuracy", "train_loss_meaning": "clean_cross_entropy_plus_jsd" if policy == "augmix_jsd" else ("mixed_target_cross_entropy" if policy == "cutmix" else "cross_entropy")}
    if policy == "augmix_jsd":
        record.update({"augmix_width": 3, "augmix_depth": [1, 3], "augmix_severity": 3, "augmix_alpha": 1.0, "jsd_weight": JSD_WEIGHT})
    if policy == "cutmix":
        record.update({"cutmix_probability": CUTMIX_PROB, "cutmix_alpha": 1.0})
    if optimizer_kind == "sam":
        record.update({"sam_rho": SAM_RHO, "sam_adaptive": False})
    return record


class TrainingMethod:
    def __init__(self, policy: Policy, optimizer_kind: OptimizerKind, *, model: nn.Module, optimizer: torch.optim.Optimizer, mean: torch.Tensor, std: torch.Tensor, rngs: Rngs, amp: bool, deadline: float | None = None):
        if policy not in POLICIES or optimizer_kind not in OPTIMIZERS or (optimizer_kind == "sam" and policy in {"augmix_jsd", "cutmix"}):
            raise ValueError("unsupported training method")
        self.policy, self.optimizer_kind = policy, optimizer_kind
        self.model, self.optimizer = model, optimizer
        self.mean, self.std, self.rngs = mean, std, rngs
        self.amp, self.deadline = amp, deadline
        device_type = mean.device.type
        self.descent_scaler = torch.amp.GradScaler(device_type, enabled=amp)
        self.ascent_scaler = torch.amp.GradScaler(device_type, enabled=amp) if optimizer_kind == "sam" else None

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("deadline expired during training method")

    def _forward(self, batch: TrainBatch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=batch.x.device.type, enabled=self.amp):
            loss, correct = objective(batch, self.model(batch.x))
        if not math.isfinite(loss.item()):
            raise ValueError("non-finite training loss")
        return loss, correct

    def step(self, images_u8: torch.Tensor, labels: torch.Tensor) -> StepOutcome:
        self._check_deadline()
        batch = training_batch(self.policy, images_u8, labels, self.mean, self.std, self.rngs)
        self._check_deadline()
        self.optimizer.zero_grad(set_to_none=True)
        loss, correct = self._forward(batch)
        loss_value, correct_value = loss.item(), int(correct.item())
        self._check_deadline()
        if self.optimizer_kind == "sgd":
            self.descent_scaler.scale(loss).backward()
            self._check_deadline()
            previous_scale = self.descent_scaler.get_scale()
            self.descent_scaler.step(self.optimizer)
            self.descent_scaler.update()
            update = "amp_skip" if self.descent_scaler.get_scale() < previous_scale else "applied"
            return StepOutcome(loss_value, correct_value, len(labels), len(batch.x), update)

        # Foret et al., https://github.com/google-research/sam. The perturbation
        # uses the loss gradient before SGD applies weight decay.
        self.ascent_scaler.scale(loss).backward()
        self.ascent_scaler.unscale_(self.optimizer)
        params = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])) if grads else torch.zeros((), device=images_u8.device)
        self.ascent_scaler.update()
        self._check_deadline()
        if not torch.isfinite(norm).item():
            return StepOutcome(loss_value, correct_value, len(labels), len(labels), "ascent_skip")
        saved_params = [p.detach().clone() for p in params]
        bn_buffers = [(buffer, buffer.detach().clone()) for module in self.model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm) for buffer in (module.running_mean, module.running_var, module.num_batches_tracked) if buffer is not None]
        perturbation_scale = SAM_RHO / (norm.item() + 1e-12)
        try:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=perturbation_scale)
            self.optimizer.zero_grad(set_to_none=True)
            self._check_deadline()
            second_loss, _ = self._forward(batch)
            self._check_deadline()
            self.descent_scaler.scale(second_loss).backward()
        finally:
            with torch.no_grad():
                for p, original in zip(params, saved_params):
                    p.copy_(original)
                for buffer, original in bn_buffers:
                    buffer.copy_(original)
        self._check_deadline()
        if not self.amp:
            finite = [torch.isfinite(p.grad).all() for p in params if p.grad is not None]
            if finite and not torch.stack(finite).all().item():
                raise ValueError("non-finite SAM descent gradients")
        previous_scale = self.descent_scaler.get_scale()
        self.descent_scaler.step(self.optimizer)
        self.descent_scaler.update()
        update = "amp_skip" if self.descent_scaler.get_scale() < previous_scale else "applied"
        return StepOutcome(loss_value, correct_value, len(labels), 2 * len(labels), update)
