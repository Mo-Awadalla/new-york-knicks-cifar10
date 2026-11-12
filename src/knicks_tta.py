"""Frozen, per-ID AugMix probability averaging for the existing classifier.

The augmentation operations and mixture arithmetic are adapted from MEMO's
cifar-10-exps/utils/third_party.py (https://github.com/zhangmarvin/memo),
which credits Google Research's AugMix. The local per-ID random streams and
checkpoint normalization are deliberate reproducibility adaptations.

MIT License

Copyright (c) 2023 Marvin Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import hashlib
from numbers import Integral

import numpy as np
import torch
from PIL import Image, ImageOps


POLICY_NAME = "augmix32"
VIEWS = 32
SEED = 20261105


def policy_record() -> dict:
    """Return a fresh, JSON-compatible description of the frozen policy."""
    return {
        "name": POLICY_NAME,
        "views": VIEWS,
        "seed": SEED,
        "version": "knicks-augmix32-memo-v1",
        "reference": "https://github.com/zhangmarvin/memo/blob/main/cifar-10-exps/utils/third_party.py",
        "rng": {
            "per_id": "SHA256 UTF-8 knicks-augmix32-memo-v1:20261105:<id>; independent of input order and batching",
            "torch": "CPU generator seeded with little-endian digest bytes 0:8 masked to 63 bits",
            "numpy": "RandomState MT19937 seeded with little-endian digest bytes 8:12",
            "streams": "reuse both local generators over all 32 views; no global random draws",
        },
        "transform": {
            "preaugment": "RGB zero-pad 4; independently draw top and left in [0,8] with torch.randint, crop 32, then horizontal flip when torch.rand < 0.5",
            "mixture": "float32 Dirichlet(1,1,1), then float32 Beta(1,1); three independent chains, each depth randint(1,4), uniform operations with replacement; m*normalized clean + (1-m)*sum(w*normalized chain)",
            "operations": ["autocontrast", "equalize", "rotate", "solarize", "shear_x", "shear_y", "translate_x", "translate_y", "posterize"],
            "severity": "level uniform(0.1,1.0); rotation int(level*30/10), solarize 256-int(level*256/10), shear level*0.3/10, translation int(level*(32/3)/10), posterize 4-int(level*4/10); signed operations negate if a separate uniform draw exceeds 0.5",
            "resampling": "bilinear rotation and affine with literal fillcolor=128",
            "normalization": "each clean and chain image converted to FP32 /255 and normalized with checkpoint mean and std before mixing",
            "prediction": "FP32 softmax of each view's logits, then FP32 mean of exactly 32 probabilities; no extra clean view",
        },
    }


def _streams(image_id):
    digest = hashlib.sha256(f"knicks-augmix32-memo-v1:{SEED}:{image_id}".encode("utf-8")).digest()
    torch_seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    numpy_seed = int.from_bytes(digest[8:12], "little")
    return torch.Generator(device="cpu").manual_seed(torch_seed), np.random.RandomState(numpy_seed)


def _operation(image, operation, rng):
    if operation == 0:
        return ImageOps.autocontrast(image)
    if operation == 1:
        return ImageOps.equalize(image)
    level = rng.uniform(0.1, 1.0)
    if operation == 2:
        degrees = int(level * 30 / 10)
        if rng.uniform() > 0.5:
            degrees = -degrees
        return image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=128)
    if operation == 3:
        return ImageOps.solarize(image, 256 - int(level * 256 / 10))
    if operation in (4, 5):
        amount = float(level) * 0.3 / 10.0
    elif operation in (6, 7):
        amount = int(level * (32 / 3) / 10)
    else:
        return ImageOps.posterize(image, 4 - int(level * 4 / 10))
    if rng.uniform() > 0.5:
        amount = -amount
    matrix = {
        4: (1, amount, 0, 0, 1, 0),
        5: (1, 0, 0, amount, 1, 0),
        6: (1, 0, amount, 0, 1, 0),
        7: (1, 0, 0, 0, 1, amount),
    }[operation]
    return image.transform((32, 32), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR, fillcolor=128)


def _normalize(image, mean, std):
    pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).to(torch.float32).div_(255)
    return (pixels - mean) / std


def _view(image, torch_rng, numpy_rng, mean, std):
    padded = ImageOps.expand(image, border=4, fill=0)
    top = int(torch.randint(0, 9, (1,), generator=torch_rng).item())
    left = int(torch.randint(0, 9, (1,), generator=torch_rng).item())
    clean = padded.crop((left, top, left + 32, top + 32))
    if torch.rand(1, generator=torch_rng).item() < 0.5:
        clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    normalized = _normalize(clean, mean, std)
    weights = np.float32(numpy_rng.dirichlet([1.0, 1.0, 1.0]))
    mixing = np.float32(numpy_rng.beta(1.0, 1.0))
    mixture = torch.zeros_like(normalized)
    for weight in weights:
        branch = clean.copy()
        for _ in range(numpy_rng.randint(1, 4)):
            branch = _operation(branch, numpy_rng.choice(9), numpy_rng)
        mixture += float(weight) * _normalize(branch, mean, std)
    return float(mixing) * normalized + float(1 - mixing) * mixture


def _snapshot(model):
    return {
        (kind, name): value.detach().clone()
        for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
        for name, value in values
    }


def _same_bits(left, right):
    # Compare bytes so unchanged NaNs and signed zero are handled exactly.
    left = left.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    right = right.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left, right)


def _check_snapshot(model, before):
    after = {
        (kind, name): value.detach()
        for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
        for name, value in values
    }
    if before.keys() != after.keys() or any(
        old.dtype != after[key].dtype or old.device != after[key].device
        or old.shape != after[key].shape or not _same_bits(old, after[key])
        for key, old in before.items()
    ):
        raise RuntimeError("model parameters or buffers changed during AugMix inference")


def augmix32_probabilities(model, images_u8, ids, mean, std, *, batch_size=128):
    """Average 32 fixed per-ID augmented predictions, without adapting the model."""
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if not isinstance(images_u8, torch.Tensor) or images_u8.dtype != torch.uint8 or images_u8.ndim != 4 or tuple(images_u8.shape[1:]) != (3, 32, 32) or images_u8.shape[0] == 0:
        raise ValueError("images_u8 must be a nonempty uint8 tensor of shape [N,3,32,32]")
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence aligned with images")
    try:
        ids = tuple(ids)
    except TypeError as exc:
        raise ValueError("ids must be a sequence aligned with images") from exc
    if len(ids) != len(images_u8) or any(isinstance(x, bool) or not isinstance(x, (str, Integral)) or isinstance(x, str) and not x for x in ids):
        raise ValueError("ids must contain one nonempty string or integer per image")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        mean = torch.as_tensor(mean, dtype=torch.float32, device="cpu").detach().reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32, device="cpu").detach().reshape(-1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("mean and std must each have three finite values") from exc
    if mean.numel() != 3 or std.numel() != 3 or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("mean and std must each have three finite values, with positive std")
    mean, std = mean.reshape(3, 1, 1), std.reshape(3, 1, 1)
    before = _snapshot(model)
    devices = {value.device for value in before.values()}
    if len(devices) > 1:
        raise ValueError("model parameters and buffers must be on one device")
    device = next(iter(devices), torch.device("cpu"))
    images = images_u8.detach().to(device="cpu")
    pending, owners = [], []
    sums = None

    def flush():
        nonlocal sums
        if not pending:
            return
        logits = model(torch.stack(pending).to(device=device, dtype=torch.float32))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != len(pending) or logits.shape[1] == 0:
            raise ValueError("model must return logits of shape [batch, classes]")
        probs = torch.softmax(logits.float(), dim=1).cpu()
        if not torch.isfinite(probs).all().item():
            raise ValueError("model produced non-finite probabilities")
        if sums is None:
            sums = torch.zeros((len(ids), probs.shape[1]), dtype=torch.float32)
        elif sums.shape[1] != probs.shape[1]:
            raise ValueError("model changed its number of classes between batches")
        sums.index_add_(0, torch.tensor(owners, dtype=torch.long), probs)
        pending.clear()
        owners.clear()

    try:
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for index, image_id in enumerate(ids):
                image = Image.fromarray(images[index].permute(1, 2, 0).numpy())
                torch_rng, numpy_rng = _streams(image_id)
                for _ in range(VIEWS):
                    pending.append(_view(image, torch_rng, numpy_rng, mean, std))
                    owners.append(index)
                    if len(pending) == batch_size:
                        flush()
            flush()
        return sums / VIEWS
    finally:
        _check_snapshot(model, before)
