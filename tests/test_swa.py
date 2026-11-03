import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import knicks_cifar as kc


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Linear(3, 10)

    def forward(self, x):
        return self.head(self.bn(x).mean(dim=(2, 3)))


class SWATest(unittest.TestCase):
    def setUp(self):
        self.cfg = kc.config_from(Path(__file__).resolve().parents[1] / "configs/cutout.json")
        self.mean = torch.zeros(3)
        self.std = torch.ones(3)
        self.images = torch.randint(0, 256, (9, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(9))
        self.rows = (0, 2, 3, 5, 7)

    def test_cosine_saved_run_schedule_and_swa_boundary(self):
        for step, expected in [(351, 0.02), (1759, 0.1), (52799, 0.01536420552286627), (53151, 0.01478775568368792), (70399, 0.0)]:
            self.assertAlmostEqual(kc.learning_rate(self.cfg, step, 352), expected, places=14)
        swa = replace(self.cfg, strategy="swa")
        self.assertEqual(kc.swa_start(swa), 150)
        self.assertEqual(kc.swa_start(replace(swa, epochs=3)), 2)
        self.assertIsNone(kc.swa_start(self.cfg))
        for step in range(52800):
            self.assertEqual(kc.learning_rate(swa, step, 352), kc.learning_rate(self.cfg, step, 352))
        for step in range(52800, 70400):
            self.assertEqual(kc.learning_rate(swa, step, 352), 0.01)

    def test_seeded_streams_are_distinct_across_runs_and_match_between_arms(self):
        domains = set()
        for seed in range(4):
            cosine = replace(self.cfg, seed=seed)
            swa = replace(cosine, strategy="swa")
            for offset in range(1, 6):
                stream = kc.stream_seed(cosine, offset)
                self.assertEqual(stream, seed * 16 + offset)
                self.assertEqual(stream, kc.stream_seed(swa, offset))
                self.assertNotIn(stream, domains)
                domains.add(stream)

    def test_strategy_validation_and_legacy_default(self):
        self.assertEqual(kc.RunConfig().strategy, "cosine")
        with self.assertRaises(ValueError):
            replace(self.cfg, strategy="invalid").validate()
        with self.assertRaises(ValueError):
            replace(self.cfg, strategy="swa", epochs=1).validate()
        replace(self.cfg, strategy="swa", split="all").validate()

    def test_cuda_determinism_policy_restores_randaugment_support(self):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            kc.configure_training_determinism(self.cfg, torch.device("cuda"))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            kc.configure_training_determinism(replace(self.cfg, aug="randaugment_cutout"), torch.device("cuda"))
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)

    def test_comparison_staging_rejects_full_data_config(self):
        config = Path(__file__).resolve().parents[1] / "configs/full-data-cutout.json"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage"
            with self.assertRaises(ValueError):
                kc.main(["stage", "--config", str(config), "--run-id", "compare", "--out", str(out), "--mode", "compare"])
            self.assertFalse(out.exists())

    def test_cuda_training_rejects_unsupported_torch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            for version in ("2.0.0+cu118", "2.10.0", "2.11.0rc1", "unknown"):
                with self.subTest(version=version), patch.object(torch, "__version__", version):
                    with self.assertRaisesRegex(RuntimeError, "PyTorch 2.11"):
                        kc.train(self.cfg, Path(tmp), out, torch.device("cuda"), 60)
                    self.assertFalse(out.exists())

    def test_refresh_uses_each_training_row_and_preserves_other_state(self):
        cfg = replace(self.cfg, strategy="swa", batch=2)
        model = TinyModel().eval()
        model.bn.running_mean.fill_(99)
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
        params = [p.detach().clone() for p in model.parameters()]
        grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        seen = []
        original = kc.augment

        def capture(x, *args):
            seen.extend(x[:, 0, 0, 0].tolist())
            return original(x, *args)

        with patch.object(kc, "augment", side_effect=capture):
            batches = kc.refresh_batchnorm(model, self.images, self.rows, cfg, self.mean, self.std, time.monotonic() + 60)
        self.assertEqual(batches, 3)
        self.assertCountEqual(seen, self.images[list(self.rows), 0, 0, 0].tolist())
        self.assertFalse(torch.equal(model.bn.running_mean, torch.full((3,), 99.0)))
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for param, before, grad in zip(model.parameters(), params, grads):
            self.assertTrue(torch.equal(param, before))
            if grad is None:
                self.assertIsNone(param.grad)
            else:
                self.assertTrue(torch.equal(param.grad, grad))
        other_images = self.images.clone()
        other_images[[1, 4, 6, 8]] = 0
        other = TinyModel().eval()
        other.load_state_dict(model.state_dict())
        kc.refresh_batchnorm(other, other_images, self.rows, cfg, self.mean, self.std, time.monotonic() + 60)
        self.assertTrue(torch.equal(model.bn.running_mean, other.bn.running_mean))
        self.assertTrue(torch.equal(model.bn.running_var, other.bn.running_var))

    def test_refresh_deadline_raises_and_restores_mode(self):
        model = TinyModel().eval()
        with self.assertRaises(TimeoutError):
            kc.refresh_batchnorm(model, self.images, self.rows, replace(self.cfg, strategy="swa"), self.mean, self.std, time.monotonic() - 1)
        self.assertFalse(model.training)
        self.assertEqual(model.bn.momentum, 0.1)

    def test_refresh_deadline_is_checked_between_batches(self):
        model = TinyModel().eval()
        with patch.object(kc.time, "monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaises(TimeoutError):
                kc.refresh_batchnorm(model, self.images, self.rows, replace(self.cfg, strategy="swa", batch=2), self.mean, self.std, 1.0)
        self.assertEqual(model.bn.num_batches_tracked.item(), 1)
        self.assertFalse(model.training)
        self.assertEqual(model.bn.momentum, 0.1)

    def test_averaged_checkpoint_strictly_reloads_plain_model(self):
        cfg = replace(self.cfg, strategy="swa", batch=2)
        model = TinyModel()
        avg = AveragedModel(model, use_buffers=False)
        snapshots = []
        for amount in (1.0, 2.0, 4.0):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(amount)
            snapshots.append([p.detach().clone() for p in model.parameters()])
            avg.update_parameters(model)
        kc.refresh_batchnorm(avg.module, self.images, self.rows, cfg, self.mean, self.std, time.monotonic() + 60)
        for index, param in enumerate(avg.module.parameters()):
            self.assertTrue(torch.allclose(param, torch.stack([s[index] for s in snapshots]).mean(0), atol=1e-6))
        split = kc.Split(self.rows, (1, 4, 6, 8), "hash")
        with tempfile.TemporaryDirectory() as tmp, patch.object(kc, "build", side_effect=lambda arch: TinyModel()):
            path = Path(tmp) / "checkpoint.pt"
            torch.save(kc.make_checkpoint(cfg, avg.module, self.mean, self.std, split, {}, "train", 200), path)
            checkpoint, loaded, _, _ = kc.checkpoint_model(path, torch.device("cpu"))
            self.assertEqual(set(checkpoint["state_dict"]), set(model.state_dict()))
            with torch.no_grad():
                self.assertTrue(torch.equal(avg.module(self.images[:2].float()), loaded(self.images[:2].float())))


if __name__ == "__main__":
    unittest.main()
