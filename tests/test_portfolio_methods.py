import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import knicks_methods as km


MEAN = torch.tensor([0.2, 0.4, 0.6])
STD = torch.tensor([0.3, 0.25, 0.2])


def streams(seed=0):
    return km.Rngs(torch.Generator().manual_seed(seed + 2),
                   torch.Generator().manual_seed(seed + 3),
                   np.random.default_rng(seed + 6))


def images(count=4):
    return torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(41))


def legacy_augment(x, policy, crop_rng, cutout_rng):
    count = len(x)
    offsets = torch.randint(0, 9, (count, 2), generator=crop_rng).tolist()
    flips = (torch.rand(count, generator=crop_rng) < 0.5).tolist()
    transform = None
    if policy == "randaugment_cutout":
        from torchvision.transforms import RandAugment
        transform = RandAugment(num_ops=2, magnitude=9)
    prepared = []
    for image, (top, left), flip in zip(x, offsets, flips):
        padded = F.pad(image.float().unsqueeze(0), (4, 4, 4, 4), mode="reflect")[0]
        crop = padded[:, top:top + 32, left:left + 32]
        if flip:
            crop = crop.flip(-1)
        if transform is not None:
            crop = transform(crop.to(torch.uint8))
        prepared.append((crop.float() / 255 - MEAN[:, None, None]) / STD[:, None, None])
    if policy in {"cutout", "randaugment_cutout"}:
        corners = (torch.randint(0, 32, (count, 2), generator=cutout_rng) - 8).tolist()
        for image, (top, left) in zip(prepared, corners):
            image[:, max(0, top):min(32, top + 16), max(0, left):min(32, left + 16)] = 0
    return torch.stack(prepared)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 3)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.head(x.mean((2, 3)))


def make_method(model, policy="basic", optimizer_kind="sgd", amp=False, seed=0, **sgd):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                                weight_decay=sgd.get("weight_decay", 0.0))
    method = km.TrainingMethod(policy, optimizer_kind, model=model, optimizer=optimizer,
                               mean=MEAN, std=STD, rngs=streams(seed), amp=amp)
    return method, optimizer


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class LegacyStepTest(unittest.TestCase):
    def test_legacy_cutout_sgd_and_random_streams(self):
        torch.manual_seed(7)
        actual = TinyModel()
        expected = copy.deepcopy(actual)
        rng = streams(12)
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.1, momentum=0.9,
                                    weight_decay=0.0005, nesterov=True)
        method = km.TrainingMethod("cutout", "sgd", model=actual, optimizer=optimizer,
                                   mean=MEAN, std=STD, rngs=rng, amp=False)
        old_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1, momentum=0.9,
                                        weight_decay=0.0005, nesterov=True)
        old_rng = streams(12)
        x, y = images(), torch.tensor([0, 1, 2, 1])
        prepared = legacy_augment(x, "cutout", old_rng.crop, old_rng.cutout)
        old_optimizer.zero_grad(set_to_none=True)
        logits = expected(prepared)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        old_optimizer.step()
        outcome = method.step(x, y)
        self.assertEqual(outcome.update, "applied")
        self.assertEqual((outcome.rows, outcome.forward_images), (4, 4))
        self.assertEqual(outcome.loss, loss.item())
        self.assertEqual(outcome.correct, int((logits.argmax(1) == y).sum()))
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
        self.assertTrue(torch.equal(rng.crop.get_state(), old_rng.crop.get_state()))
        self.assertTrue(torch.equal(rng.cutout.get_state(), old_rng.cutout.get_state()))

    def test_randaugment_legacy_global_and_explicit_streams(self):

        x = images()
        rng = streams(17)
        actual_rng = streams(17)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            expected = legacy_augment(x, "randaugment_cutout", rng.crop, rng.cutout)
            state = torch.get_rng_state()
            torch.manual_seed(31)
            actual = km.augment(x, "randaugment_cutout", MEAN, STD, actual_rng.crop, actual_rng.cutout)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertTrue(torch.equal(actual_rng.crop.get_state(), rng.crop.get_state()))
        self.assertTrue(torch.equal(actual_rng.cutout.get_state(), rng.cutout.get_state()))

    def test_legacy_basic_and_cutout_preparation(self):
        x = images()
        for policy in ("basic", "cutout"):
            with self.subTest(policy=policy):
                actual_rng, old_rng = streams(22), streams(22)
                got = km.augment(x, policy, MEAN, STD, actual_rng.crop, actual_rng.cutout)
                expected = legacy_augment(x, policy, old_rng.crop, old_rng.cutout)
                self.assertTrue(torch.equal(got, expected))
                self.assertTrue(torch.equal(actual_rng.crop.get_state(), old_rng.crop.get_state()))
                self.assertTrue(torch.equal(actual_rng.cutout.get_state(), old_rng.cutout.get_state()))


class AugMixTest(unittest.TestCase):
    def test_operations_keep_rgb_dimensions(self):
        self.assertEqual(len(km.AUGMIX_OPS), 9)
        self.assertEqual({op.__name__.lstrip("_") for op in km.AUGMIX_OPS},
                         {"autocontrast", "equalize", "posterize", "rotate", "solarize",
                          "shear_x", "shear_y", "translate_x", "translate_y"})
        rgb = np.random.default_rng(5).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        for index, op in enumerate(km.AUGMIX_OPS):
            with self.subTest(operation=index):
                output = op(Image.fromarray(rgb), np.random.default_rng(12))
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.size, (32, 32))

    def test_views_seeded_independent_and_clean_matches_control(self):
        x, y = images(), torch.tensor([0, 1, 2, 1])
        first = km.training_batch("augmix_jsd", x, y, MEAN, STD, streams(3))
        repeat = km.training_batch("augmix_jsd", x, y, MEAN, STD, streams(3))
        changed = km.training_batch("augmix_jsd", x, y, MEAN, STD, streams(4))
        self.assertIsInstance(first, km.AugMixBatch)
        self.assertEqual(tuple(first.x.shape), (12, 3, 32, 32))
        self.assertTrue(torch.equal(first.x, repeat.x))
        self.assertFalse(torch.equal(first.x[4:8], first.x[8:12]))
        self.assertFalse(torch.equal(first.x, changed.x))
        pixels = (first.x[4:] * STD[None, :, None, None] + MEAN[None, :, None, None]) * 255
        self.assertTrue(bool(((pixels - pixels.round()).abs() > 1e-3).any()))
        rng = streams(3)
        expected = legacy_augment(x, "basic", rng.crop, rng.cutout)
        self.assertTrue(torch.equal(first.x[:4], expected))
        self.assertTrue(torch.equal(first.y, y))

    def test_jsd_matches_reference_and_fp16_extremes_are_finite(self):
        x, y = images(2), torch.tensor([0, 2])
        batch = km.training_batch("augmix_jsd", x, y, MEAN, STD, streams(3))
        base = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.5, -0.7]])
        loss, correct = km.objective(batch, base.repeat(3, 1))
        self.assertTrue(torch.allclose(loss, F.cross_entropy(base, y), atol=1e-6))
        self.assertEqual(int(correct), int((base.argmax(1) == y).sum()))
        logits = torch.tensor([[50., -50., 0.], [-50., 50., 0.],
                               [-50., 50., 0.], [50., -50., 0.],
                               [0., -50., 50.], [0., 50., -50.]], dtype=torch.float16)
        loss, _ = km.objective(batch, logits)
        parts = logits.float().chunk(3)
        probs = [part.softmax(1) for part in parts]
        log_mean = torch.stack(probs).mean(0).clamp(1e-7, 1).log()
        expected = F.cross_entropy(parts[0], y) + 4 * sum(F.kl_div(log_mean, p, reduction="batchmean") for p in probs)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss.float(), expected, atol=1e-5))


class CutMixTest(unittest.TestCase):
    def test_border_box_pixel_sources_actual_area_and_loss(self):
        x = images()
        y = torch.tensor([0, 1, 2, 0])
        for seed in range(10000):
            rng = np.random.default_rng(seed + 6)
            if rng.random() >= 0.5:
                continue
            lam = rng.beta(1, 1)
            perm = rng.permutation(4)
            cx, cy = int(rng.integers(32)), int(rng.integers(32))
            half = int(32 * math.sqrt(1 - lam)) // 2
            x1, x2 = max(0, cx - half), min(32, cx + half)
            y1, y2 = max(0, cy - half), min(32, cy + half)
            if x2 > x1 and y2 > y1 and (x1 == 0 or y1 == 0 or x2 == 32 or y2 == 32) and not np.array_equal(perm, np.arange(4)):
                break
        else:
            self.fail("no deterministic border case found")
        batch = km.training_batch("cutmix", x, y, MEAN, STD, streams(seed))
        self.assertIsInstance(batch, km.MixedBatch)
        rng = streams(seed)
        clean = legacy_augment(x, "basic", rng.crop, rng.cutout)
        expected = clean.clone()
        expected[:, :, y1:y2, x1:x2] = clean[torch.as_tensor(perm), :, y1:y2, x1:x2]
        self.assertTrue(torch.equal(batch.x, expected))
        self.assertTrue(torch.equal(batch.y_a, y))
        self.assertTrue(torch.equal(batch.y_b, y[torch.as_tensor(perm)]))
        actual_lam = 1 - (x2 - x1) * (y2 - y1) / 1024
        self.assertEqual(batch.lam, actual_lam)
        logits = torch.tensor([[2., 1., -1.], [1., 2., -1.], [-1., 0., 1.], [0., 1., 2.]])
        loss, correct = km.objective(batch, logits)
        expected_loss = actual_lam * F.cross_entropy(logits, y) + (1 - actual_lam) * F.cross_entropy(logits, y[torch.as_tensor(perm)])
        self.assertTrue(torch.allclose(loss, expected_loss))
        self.assertEqual(int(correct), int((logits.argmax(1) == y).sum()))

    def test_no_mix_equals_shared_basic_crop(self):
        seed = next(s for s in range(100) if np.random.default_rng(s + 6).random() >= 0.5)
        x, y = images(), torch.tensor([0, 1, 2, 1])
        batch = km.training_batch("cutmix", x, y, MEAN, STD, streams(seed))
        self.assertIsInstance(batch, km.HardBatch)
        rng = streams(seed)
        self.assertTrue(torch.equal(batch.x, legacy_augment(x, "basic", rng.crop, rng.cutout)))
        self.assertTrue(torch.equal(batch.y, y))


class BatchNormModel(nn.Module):
    def __init__(self, fail_second=False, nonfinite_second=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Linear(3, 3)
        self.calls = 0
        self.inputs = []
        self.fail_second = fail_second
        self.nonfinite_second = nonfinite_second
        self.first_bn_buffers = None

    def forward(self, x):
        self.calls += 1
        self.inputs.append(x.detach().clone())
        x = self.bn(x).mean((2, 3))
        if self.calls == 1:
            self.first_bn_buffers = {
                name: getattr(self.bn, name).detach().clone()
                for name in ("running_mean", "running_var", "num_batches_tracked")
            }
        if self.calls == 2 and self.fail_second:
            raise RuntimeError("second pass failed")
        logits = self.head(x)
        if self.calls == 2 and self.nonfinite_second:
            return logits * float("inf")
        return logits


class SAMTest(unittest.TestCase):
    def test_matches_manual_two_gradient_reference_with_weight_decay(self):
        torch.manual_seed(9)
        model = TinyModel()
        reference = copy.deepcopy(model)
        x, y = images(), torch.tensor([0, 1, 2, 1])
        method, _ = make_method(model, policy="cutout", optimizer_kind="sam", seed=8, weight_decay=0.07)
        rng = streams(8)
        prepared = legacy_augment(x, "cutout", rng.crop, rng.cutout)
        initial = [p.detach().clone() for p in reference.parameters()]
        first = F.cross_entropy(reference(prepared), y)
        grads = torch.autograd.grad(first, tuple(reference.parameters()))
        norm = torch.sqrt(sum((g * g).sum() for g in grads))
        with torch.no_grad():
            for param, grad in zip(reference.parameters(), grads):
                param.add_(grad * (0.05 / (norm + 1e-12)))
        second = F.cross_entropy(reference(prepared), y)
        descent = torch.autograd.grad(second, tuple(reference.parameters()))
        # First Nesterov SGD step uses (1 + momentum) times the gradient plus decay.
        expected = [p - 0.1 * 1.9 * (g + 0.07 * p) for p, g in zip(initial, descent)]
        outcome = method.step(x, y)
        self.assertEqual((outcome.update, outcome.rows, outcome.forward_images), ("applied", 4, 8))
        self.assertAlmostEqual(outcome.loss, first.item(), places=6)
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(torch.equal(model.inputs[0], prepared))
        self.assertTrue(torch.equal(model.inputs[0], model.inputs[1]))
        for param, value in zip(model.parameters(), expected):
            self.assertTrue(torch.allclose(param, value, atol=2e-7, rtol=1e-6))

    def test_batchnorm_keeps_only_first_pass_updates(self):
        torch.manual_seed(10)
        model = BatchNormModel()
        x, y = images(), torch.tensor([0, 1, 2, 1])
        method, _ = make_method(model, policy="cutout", optimizer_kind="sam", seed=5)
        outcome = method.step(x, y)
        self.assertEqual(outcome.update, "applied")
        self.assertTrue(torch.equal(model.inputs[0], model.inputs[1]))
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)
        self.assertEqual(int(model.bn.num_batches_tracked), 1)

    def test_zero_gradient_leaves_parameters_unchanged(self):
        class ZeroGradient(TinyModel):
            def forward(self, x):
                return super().forward(x) * 0

        model = ZeroGradient()
        before = snapshot(model)
        method, _ = make_method(model, optimizer_kind="sam")
        outcome = method.step(images(), torch.tensor([0, 1, 2, 1]))
        self.assertEqual((outcome.update, outcome.forward_images), ("applied", 8))
        for name, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)

    def test_second_pass_failure_restores_parameters_and_first_bn_buffers(self):
        for nonfinite in (False, True):
            with self.subTest(nonfinite=nonfinite):
                torch.manual_seed(10)
                model = BatchNormModel(fail_second=not nonfinite, nonfinite_second=nonfinite)
                before = snapshot(model)
                method, optimizer = make_method(model, optimizer_kind="sam")
                with self.assertRaises(ValueError if nonfinite else RuntimeError):
                    method.step(images(), torch.tensor([0, 1, 2, 1]))
                for name, value in model.named_parameters():
                    self.assertTrue(torch.equal(value, before[name]), name)
                for name in ("running_mean", "running_var", "num_batches_tracked"):
                    self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)
                self.assertFalse(optimizer.state)

    def test_nonfinite_second_pass_batchnorm_does_not_poison_buffers(self):
        torch.manual_seed(10)
        model = BatchNormModel()
        before = snapshot(model)
        method, optimizer = make_method(model, optimizer_kind="sam")

        def poison_second(module, args):
            if model.calls == 2:
                return (torch.full_like(args[0], float("inf")),)
            return args

        hook = model.bn.register_forward_pre_hook(poison_second)
        try:
            with self.assertRaises(ValueError):
                method.step(images(), torch.tensor([0, 1, 2, 1]))
        finally:
            hook.remove()
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), name)
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)
        self.assertFalse(optimizer.state)

    def test_deadline_after_perturbed_forward_rolls_back(self):
        torch.manual_seed(10)
        model = BatchNormModel()
        before = snapshot(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        method = km.TrainingMethod("basic", "sam", model=model, optimizer=optimizer,
                                   mean=MEAN, std=STD, rngs=streams(), amp=False, deadline=1.0)
        with patch.object(km.time, "monotonic", side_effect=lambda: 2.0 if model.calls == 2 else 0.0):
            with self.assertRaises(TimeoutError):
                method.step(images(), torch.tensor([0, 1, 2, 1]))
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), name)
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)
        self.assertFalse(optimizer.state)

    def test_unscaled_nonfinite_descent_gradient_cannot_update_sam(self):
        torch.manual_seed(10)
        model = BatchNormModel()
        before = snapshot(model)
        method, optimizer = make_method(model, optimizer_kind="sam", amp=False)
        for param in model.parameters():
            optimizer.state[param]["momentum_buffer"] = torch.full_like(param, 0.375)
        old_state = {param: state["momentum_buffer"].clone() for param, state in optimizer.state.items()}
        calls = 0

        def inject(grad):
            nonlocal calls
            calls += 1
            return torch.full_like(grad, float("inf")) if calls == 2 else grad

        handle = model.head.weight.register_hook(inject)
        try:
            with self.assertRaisesRegex(ValueError, "non-finite SAM descent gradients"):
                method.step(images(), torch.tensor([0, 1, 2, 1]))
        finally:
            handle.remove()
        self.assertEqual(calls, 2)
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, before[name]), name)
        for param, value in old_state.items():
            self.assertTrue(torch.equal(optimizer.state[param]["momentum_buffer"], value))
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)

    def test_real_cpu_gradscaler_skips_each_overflow_phase(self):
        for overflow_pass, outcome_name, expected_views in ((1, "ascent_skip", 4), (2, "amp_skip", 8)):
            with self.subTest(overflow_pass=overflow_pass):
                torch.manual_seed(10)
                model = BatchNormModel()
                before = snapshot(model)
                method, optimizer = make_method(model, optimizer_kind="sam", amp=True)
                for param in model.parameters():
                    optimizer.state[param]["momentum_buffer"] = torch.full_like(param, 0.375)
                old_state = {param: value["momentum_buffer"].clone() for param, value in optimizer.state.items()}
                calls = [0]

                def inject(grad):
                    calls[0] += 1
                    return torch.full_like(grad, float("inf")) if calls[0] == overflow_pass else grad

                handle = model.head.weight.register_hook(inject)
                ascent_scale = method.ascent_scaler.get_scale()
                descent_scale = method.descent_scaler.get_scale()
                try:
                    result = method.step(images(), torch.tensor([0, 1, 2, 1]))
                finally:
                    handle.remove()
                self.assertEqual((result.update, result.rows, result.forward_images), (outcome_name, 4, expected_views))
                self.assertEqual(calls[0], overflow_pass)
                self.assertEqual(method.ascent_scaler.get_scale(), ascent_scale / 2 if overflow_pass == 1 else ascent_scale)
                self.assertEqual(method.descent_scaler.get_scale(), descent_scale / 2 if overflow_pass == 2 else descent_scale)
                for name, param in model.named_parameters():
                    self.assertTrue(torch.equal(param, before[name]), name)
                for param, value in old_state.items():
                    self.assertTrue(torch.equal(optimizer.state[param]["momentum_buffer"], value))
                for name in ("running_mean", "running_var", "num_batches_tracked"):
                    self.assertTrue(torch.equal(getattr(model.bn, name), model.first_bn_buffers[name]), name)
