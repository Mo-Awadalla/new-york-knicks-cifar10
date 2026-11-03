import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import knicks_cifar as kc
import knicks_methods as km


class RecordingModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.head = nn.Linear(3, 10)
        torch.nn.init.normal_(self.head.weight, generator=torch.Generator().manual_seed(seed))
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.head(x.mean((2, 3)))


class DistillationTest(unittest.TestCase):
    def test_teachers_see_same_pixels_under_their_own_statistics(self):
        student_mean, student_std = torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.25, 0.2, 0.3])
        pixels = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(3))
        x = (pixels - student_mean[None, :, None, None]) / student_std[None, :, None, None]
        teachers = tuple(km.Teacher(RecordingModel(seed).eval(), torch.tensor(mean), torch.tensor(std))
                         for seed, mean, std in ((1, [0.1, 0.2, 0.3], [0.2, 0.2, 0.2]), (2, [0.6, 0.5, 0.4], [0.3, 0.25, 0.2])))
        kd = km.Distillation(teachers, alpha=0.9, temperature=4.0)
        targets = kd.targets(x, student_mean, student_std, amp=False)
        expected = 0
        for teacher in teachers:
            teacher_input = (pixels - teacher.mean[None, :, None, None]) / teacher.std[None, :, None, None]
            self.assertTrue(torch.allclose(teacher.model.inputs[0], teacher_input, atol=1e-5))
            expected = expected + F.softmax(teacher.model.head(teacher_input.mean((2, 3))) / 4.0, dim=1)
        self.assertTrue(torch.allclose(targets, expected / 2, atol=1e-6))
        self.assertTrue(torch.allclose(targets.sum(1), torch.ones(4)))
        # The divergence term vanishes only when the student's tempered output matches the ensemble.
        hard = torch.tensor(2.0)
        matched = kd.loss(hard, 4.0 * targets.log(), targets)
        self.assertAlmostEqual(matched.item(), 0.1 * 2.0, places=5)
        self.assertGreater(kd.loss(hard, torch.zeros(4, 10), targets).item(), matched.item())

    def test_validation_student_rejects_teacher_trained_on_all_rows(self):
        split = kc.Split((0,), (1,), "fingerprint")
        teacher_cfg = kc.RunConfig(arch="wrn16_4", aug="cutout", split="all")
        with tempfile.TemporaryDirectory() as directory, patch.object(kc, "build", side_effect=lambda arch: RecordingModel(0)):
            path = Path(directory) / "teacher.pt"
            torch.save(kc.make_checkpoint(teacher_cfg, RecordingModel(0), torch.zeros(3), torch.ones(3), split, {}, "train", teacher_cfg.epochs), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            student = kc.RunConfig(arch="wrn22_4", aug="cutmix", split="val", teachers=(("teacher.pt", digest),))
            student.validate()
            with self.assertRaises(ValueError):
                kc.load_distillation(student, "fingerprint", Path(directory), torch.device("cpu"))
            distillation, records = kc.load_distillation(replace(student, split="all"), "fingerprint", Path(directory), torch.device("cpu"))
            self.assertEqual(len(distillation.teachers), 1)
            self.assertEqual(records[0]["split"], "all")
            with self.assertRaises(ValueError):
                kc.load_distillation(replace(student, teachers=(("teacher.pt", "0" * 64),)), "fingerprint", Path(directory), torch.device("cpu"))

    def test_distillation_rejects_sam_augmix_and_orphan_settings(self):
        teachers = (("t.pt", "0" * 64),)
        for change in ({"optimizer": "sam", "aug": "cutout"}, {"aug": "augmix_jsd"}, {"strategy": "swa"}, {"kd_alpha": 0.0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                kc.RunConfig(teachers=teachers, **change).validate()
        with self.assertRaises(ValueError):
            kc.RunConfig(kd_temperature=2.0).validate()

    def test_teacher_architecture_cannot_create_submission(self):
        cfg = kc.RunConfig(arch="wrn28_10", aug="cutmix", split="all")
        cfg.validate()
        self.assertGreater(kc.ARCHS["wrn28_10"][2], kc.MAX_PARAMS)
        with tempfile.TemporaryDirectory() as directory, patch.object(kc, "build", side_effect=lambda arch: RecordingModel(0)):
            path = Path(directory) / "checkpoint.pt"
            checkpoint = kc.make_checkpoint(cfg, RecordingModel(0), torch.zeros(3), torch.ones(3), kc.Split((0,), (), "s"), {}, "train", cfg.epochs)
            torch.save(checkpoint, path)
            (path.parent / "manifest.json").write_text(json.dumps({"status": "complete", "checkpoint_sha256": kc.sha256_file(path)}))
            with self.assertRaisesRegex(ValueError, "teacher-only"):
                kc.predict(path, ROOT / "data", path.parent / "submission.csv", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
