import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import knicks_cifar as kc
import knicks_methods as km


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 10)

    def forward(self, x):
        return self.head(x.mean((2, 3)))


class ConfigContractTest(unittest.TestCase):
    def test_default_legacy_and_single_treatment_rules(self):
        cfg = kc.config_from(ROOT / "configs/wrn22-cutout.json")
        self.assertEqual(cfg.optimizer, "sgd")
        self.assertEqual(kc.RunConfig(**{k: v for k, v in asdict(cfg).items() if k != "optimizer"}).optimizer, "sgd")
        for change in ({"aug": "bad"}, {"optimizer": "bad"},
                       {"optimizer": "sam", "strategy": "swa"},
                       {"optimizer": "sam", "aug": "augmix_jsd"},
                       {"optimizer": "sam", "aug": "cutmix"},
                       {"aug": "augmix_jsd", "strategy": "swa"},
                       {"aug": "cutmix", "strategy": "swa"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change).validate()
        for change in ({"aug": "augmix_jsd"}, {"aug": "cutmix"},
                       {"optimizer": "sam"}, {"strategy": "swa"}):
            replace(cfg, **change).validate()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        rng = km.Rngs(torch.Generator(), torch.Generator(), np.random.default_rng(0))
        for policy, kind in (("bad", "sgd"), ("basic", "bad"), ("cutmix", "sam"), ("augmix_jsd", "sam")):
            with self.subTest(policy=policy, kind=kind), self.assertRaises(ValueError):
                km.TrainingMethod(policy, kind, model=model, optimizer=optimizer,
                                  mean=torch.zeros(3), std=torch.ones(3), rngs=rng, amp=False)

    def test_channel_statistics_ignore_validation_rows(self):
        pixels = torch.arange(32 * 32, dtype=torch.int64).reshape(32, 32).remainder(200).byte()
        images = torch.stack([torch.stack((pixels, pixels + 10, pixels + 20)),
                              torch.stack((pixels + 30, pixels + 40, pixels + 50)),
                              torch.full((3, 32, 32), 255, dtype=torch.uint8)])
        mean, std = kc.channel_stats(images, (0, 1), batch=1)
        training = images[:2].float().div(255).permute(1, 0, 2, 3).reshape(3, -1)
        self.assertTrue(torch.allclose(mean, training.mean(1), atol=1e-7))
        self.assertTrue(torch.allclose(std, training.std(1, unbiased=False), atol=1e-7))
        self.assertFalse(torch.allclose(mean, images.float().div(255).mean((0, 2, 3))))

    def test_legacy_checkpoint_loads_strictly_and_compare_cannot_predict(self):
        cfg = kc.config_from(ROOT / "configs/wrn22-cutout.json")
        model = TinyModel().eval()
        values = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(7))
        checkpoint = kc.make_checkpoint(cfg, model, torch.zeros(3), torch.ones(3),
                                        kc.Split((0,), (1,), "split"), {}, "calibrate", cfg.epochs)
        del checkpoint["config"]["optimizer"]
        with tempfile.TemporaryDirectory() as directory, patch.object(kc, "build", side_effect=lambda arch: TinyModel()):
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded_checkpoint, loaded, _, _ = kc.checkpoint_model(path, torch.device("cpu"))
            self.assertEqual(kc.RunConfig(**loaded_checkpoint["config"]).optimizer, "sgd")
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(values), loaded(values)))
            (path.parent / "manifest.json").write_text(json.dumps({"status": "complete", "candidate_eligible": False, "checkpoint_sha256": kc.sha256_file(path)}))
            submission = path.parent / "submission.csv"
            with self.assertRaises(ValueError):
                kc.predict(path, ROOT / "data", submission, torch.device("cpu"))
            self.assertFalse(submission.exists())
            del checkpoint["state_dict"]["head.bias"]
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                kc.checkpoint_model(path, torch.device("cpu"))


class StagedSmokeTest(unittest.TestCase):
    def test_isolated_bundles_run_treatments_on_official_local_images(self):
        data = ROOT / "data"
        manifests = [data / name for name in ("train_labels.csv", "test.csv", "sample_submission.csv")]
        if not all(path.is_file() for path in manifests):
            self.skipTest("official local manifests are unavailable")
        images = kc.image_index(data)
        required = {"train_00000.png", "train_00001.png", "test_0000.png", "test_0001.png"}
        if not required.issubset(images):
            self.skipTest("official local image fixture is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(tmp)})
            (tmp / "knicks_cifar.py").write_text("raise RuntimeError('unrelated shadow module imported')\n")
            (tmp / "knicks_methods.py").write_text("raise RuntimeError('unrelated shadow module imported')\n")
            for name, expected_views in (("augmix", 6), ("sam", 4), ("cutmix", 2)):
                with self.subTest(treatment=name):
                    staged, work = tmp / f"stage-{name}", tmp / f"work-{name}"
                    kc.stage(ROOT / f"configs/wrn22-{name}.json", name, staged, "smoke", seed=1)
                    metadata = json.loads((staged / "kernel-metadata.json").read_text())
                    self.assertEqual(metadata["competition_sources"], ["cifar-10-res-net-architecture-challenge-dl"])
                    self.assertTrue(metadata["is_private"])
                    self.assertTrue(metadata["enable_gpu"])
                    self.assertFalse(metadata["enable_internet"])
                    command = [sys.executable, "-I", str(staged / "kernel.py"), "--working", str(work), "--data", str(data), "--device", "cpu"]
                    result = subprocess.run(command, cwd=tmp, env=environment, capture_output=True, text=True, timeout=600)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    output = work / f"knicks-{name}-output"
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["status"], "complete")
                    self.assertFalse(manifest["candidate_eligible"])
                    self.assertEqual((manifest["train_rows"], manifest["test_rows"]), (2, 2))
                    self.assertLess(manifest["final_loss"], manifest["initial_loss"])
                    self.assertEqual(manifest["method_steps"], 1)
                    self.assertEqual(manifest["method_update"], "applied")
                    self.assertEqual(manifest["method_forward_images"], expected_views)
                    self.assertTrue(manifest["reload_logits_exact"])
                    expected_hashes = {module: hashlib.sha256((ROOT / "src" / module).read_bytes()).hexdigest()
                                       for module in ("knicks_cifar.py", "knicks_methods.py")}
                    self.assertEqual(manifest["source_sha256_by_file"], expected_hashes)
                    self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256((output / "checkpoint.pt").read_bytes()).hexdigest())
                    with (output / "smoke_predictions.csv").open(newline="") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(0 <= int(row["label"]) <= 9 for row in rows))
                    config_bytes = (work / f"knicks-{name}-config.json").read_bytes()
                    checkpoint_hash = manifest["checkpoint_sha256"]
                    retry = subprocess.run(command, cwd=tmp, env=environment, capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(retry.returncode, 0)
                    self.assertEqual((work / f"knicks-{name}-config.json").read_bytes(), config_bytes)
                    self.assertEqual(hashlib.sha256((output / "checkpoint.pt").read_bytes()).hexdigest(), checkpoint_hash)
