import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import knicks_cifar as kc


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "scripts/compare_portfolio.py"


@dataclass(frozen=True)
class DataFixture:
    root: Path
    csv_hashes: dict
    train_ids: tuple
    val_ids: tuple
    fingerprint: str


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(root):
    root.mkdir()
    for name, fields, rows in (
        ("train_labels.csv", ["id", "filename", "label"], ((i, f"train_{i}.png", i % 10) for i in range(50000))),
        ("test.csv", ["id", "filename"], ((i, f"test_{i}.png") for i in range(2000))),
        ("sample_submission.csv", ["id", "label"], ((i, 0) for i in range(2000))),
    ):
        with (root / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    training = kc.read_manifest(root / "train_labels.csv", True)
    split = kc.stratified_split(training)
    hashes = {name: digest(root / name) for name in ("train_labels.csv", "test.csv", "sample_submission.csv")}
    return DataFixture(root, hashes, tuple(training.ids[i] for i in split.train_rows), tuple(training.ids[i] for i in split.val_rows), split.sha256)


def run_fixture(root, data, seed, correct, *, aug="cutout", optimizer="sgd", seconds=100, overrides=None):
    root.mkdir()
    labels = np.asarray(data.val_ids) % 10
    predicted = (labels + 1) % 10
    predicted[:correct] = labels[:correct]
    np.save(root / "val_pred.npy", predicted.astype(np.int8))
    config = {"arch": "wrn22_4", "split": "val", "epochs": 200, "seed": seed, "batch": 128, "lr": 0.1, "momentum": 0.9, "wd": 0.0005, "warmup": 5, "strategy": "cosine", "aug": aug, "optimizer": optimizer, **(overrides or {})}
    split = {"val_ids": data.val_ids, "train_ids": data.train_ids, "validation_sha256": data.fingerprint}
    source_files = {name: hashlib.sha256(name.encode()).hexdigest() for name in ("knicks_cifar.py", "knicks_methods.py")}
    provenance = {"csv_sha256": data.csv_hashes, "source_sha256": kc.source_digest(source_files), "source_sha256_by_file": source_files, "torch": "2.11.0+cu128", "cuda": "12.8", "gpu": "Tesla T4", "deterministic_algorithms": True, "numpy": "2.5.4", "pillow": "13.0.0", "torchvision": "0.26.0"}
    checkpoint = {"config": config, "kind": "calibrate", "completed_epochs": 200, "params": 4298970, "split_sha256": data.fingerprint, "provenance": provenance, "state_dict": {"fixture_marker": torch.tensor([seed, correct])}}
    torch.save(checkpoint, root / "checkpoint.pt")
    manifest = {**provenance, "kind": "calibrate", "status": "complete", "completed_epochs": 200, "train_rows": 45000, "val_rows": 5000, "params": 4298970, "candidate_eligible": False, "checkpoint_sha256": digest(root / "checkpoint.pt"), "split_sha256": data.fingerprint, "final_val_correct": correct, "final_val_accuracy": correct / 5000, "elapsed_seconds": seconds}
    for name, value in (("config.json", config), ("split.json", split), ("manifest.json", manifest)):
        (root / name).write_text(json.dumps(value))
    return root


def selection_arguments(control, candidates):
    command = ["--control", control]
    for candidate in candidates:
        command.extend(("--candidate", candidate))
    return command


class PortfolioSelectionTest(unittest.TestCase):
    def invoke(self, data, *args):
        return subprocess.run([sys.executable, str(COMMAND), "--data", str(data.root), *map(str, args)], capture_output=True, text=True, timeout=30)

    def portfolio_fixture(self, root, data, overrides=None, *, control_correct=4818, counts=(4843, 4843, 4817)):
        control = run_fixture(root / "control", data, 1, control_correct, overrides=overrides)
        candidates = [
            run_fixture(root / "augmix", data, 1, counts[0], aug="augmix_jsd", seconds=120, overrides=overrides),
            run_fixture(root / "sam", data, 1, counts[1], optimizer="sam", seconds=80, overrides=overrides),
            run_fixture(root / "cutmix", data, 1, counts[2], aug="cutmix", seconds=60, overrides=overrides),
        ]
        return control, candidates

    def test_selection_uses_both_exact_thresholds_and_elapsed_tiebreak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            result = self.invoke(data, *selection_arguments(control, candidates))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["phase"], "single_seed_selection")
            self.assertEqual(report["selected_candidate"], "sam")
            self.assertTrue(report["adopt"])
            self.assertEqual(report["incumbent_correct"], 4818)
            self.assertEqual(report["minimum_control_delta_images"], 25)
            self.assertEqual(report["minimum_incumbent_delta_images"], 25)
            self.assertEqual(report["minimum_candidate_correct"], 4843)
            rows = {row["name"]: row for row in report["results"]}
            self.assertEqual(set(rows), {"augmix", "sam", "cutmix"})
            self.assertEqual({name: row["delta_images"] for name, row in rows.items()}, {"augmix": 25, "sam": 25, "cutmix": -1})
            self.assertEqual({name: row["incumbent_delta_images"] for name, row in rows.items()}, {"augmix": 25, "sam": 25, "cutmix": -1})
            self.assertEqual(rows["augmix"]["incumbent_delta_percentage_points"], 0.5)
            self.assertEqual({name: row["qualified"] for name, row in rows.items()}, {"augmix": True, "sam": True, "cutmix": False})

    def test_highest_correct_count_beats_faster_qualifying_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data, counts=(4845, 4844, 4843))
            result = self.invoke(data, *selection_arguments(control, candidates))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["adopt"])
            self.assertEqual(report["selected_candidate"], "augmix")
            self.assertTrue(all(row["qualified"] for row in report["results"]))

    def test_each_threshold_can_independently_prevent_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            for name, control_correct, counts in (
                ("control_margin", 4819, (4843, 4842, 4819)),
                ("incumbent_floor", 4817, (4842, 4841, 4817)),
            ):
                with self.subTest(threshold=name):
                    folder = root / name
                    folder.mkdir()
                    control, candidates = self.portfolio_fixture(folder, data, control_correct=control_correct, counts=counts)
                    result = self.invoke(data, *selection_arguments(control, candidates))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertFalse(report["adopt"])
                    self.assertIsNone(report["selected_candidate"])
                    self.assertFalse(any(row["qualified"] for row in report["results"]))

    def test_requires_exactly_one_control_and_all_distinct_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            for args in (
                selection_arguments(control, candidates[:2]),
                selection_arguments(control, [candidates[0], candidates[0], candidates[2]]),
                selection_arguments(control, candidates + [candidates[0]]),
                ["--control", control],
                selection_arguments(control, candidates) + ["--control", control],
                sum((["--candidate", candidate] for candidate in candidates), []),
            ):
                with self.subTest(args=args):
                    self.assertNotEqual(self.invoke(data, *args).returncode, 0)

    def test_seed_other_than_one_is_rejected_for_control_and_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            wrong_control = run_fixture(root / "wrong_control", data, 2, 4818)
            wrong_candidate = run_fixture(root / "wrong_candidate", data, 3, 4843, aug="augmix_jsd")
            for bad_control, bad_candidates in (
                (wrong_control, candidates),
                (control, [wrong_candidate, *candidates[1:]]),
            ):
                with self.subTest(control=bad_control):
                    result = self.invoke(data, *selection_arguments(bad_control, bad_candidates))
                    self.assertEqual(result.returncode, 2)

    def test_mismatched_implementation_and_incomplete_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            before = digest(candidates[0] / "checkpoint.pt")
            path = candidates[0] / "manifest.json"
            original = json.loads(path.read_text())
            for changes in ({"source_sha256": "different implementation"}, {"status": "incomplete"}, {"candidate_eligible": True}, {"final_val_correct": 4844}):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**original, **changes}))
                    result = self.invoke(data, *selection_arguments(control, candidates))
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(digest(candidates[0] / "checkpoint.pt"), before)

    def test_checkpoint_cannot_be_relabelled_as_another_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            candidate = candidates[0]
            (candidate / "checkpoint.pt").write_bytes((control / "checkpoint.pt").read_bytes())
            path = candidate / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["checkpoint_sha256"] = digest(candidate / "checkpoint.pt")
            path.write_text(json.dumps(manifest))
            result = self.invoke(data, *selection_arguments(control, candidates))
            self.assertEqual(result.returncode, 2)

    def test_tampered_artifacts_and_ineligible_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            candidate = candidates[0]
            command = selection_arguments(control, candidates)
            prediction_path = candidate / "val_pred.npy"
            original_predictions = prediction_path.read_bytes()
            predictions = np.load(prediction_path, allow_pickle=False)
            predictions[0] = (int(predictions[0]) + 1) % 10
            np.save(prediction_path, predictions)
            self.assertEqual(self.invoke(data, *command).returncode, 2)
            prediction_path.write_bytes(original_predictions)

            csv_path = data.root / "test.csv"
            original_csv = csv_path.read_bytes()
            csv_path.write_bytes(original_csv + b"\n")
            self.assertEqual(self.invoke(data, *command).returncode, 2)
            csv_path.write_bytes(original_csv)

            (candidate / "submission.csv").write_text("id,label\n")
            self.assertEqual(self.invoke(data, *command).returncode, 2)
            (candidate / "submission.csv").unlink()

            with (candidate / "checkpoint.pt").open("ab") as checkpoint:
                checkpoint.write(b"tampered")
            self.assertEqual(self.invoke(data, *command).returncode, 2)

    def test_shared_but_unapproved_recipe_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data, overrides={"batch": 64, "lr": 0.2})
            result = self.invoke(data, *selection_arguments(control, candidates))
            self.assertEqual(result.returncode, 2)

    def test_rebalanced_noncanonical_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            path = control / "split.json"
            split = json.loads(path.read_text())
            val_id = split["val_ids"][0]
            train_index = next(i for i, identifier in enumerate(split["train_ids"]) if identifier % 10 == val_id % 10)
            split["val_ids"][0], split["train_ids"][train_index] = split["train_ids"][train_index], val_id
            path.write_text(json.dumps(split))
            result = self.invoke(data, *selection_arguments(control, candidates))
            self.assertEqual(result.returncode, 2)

    def test_matching_sidecars_cannot_hide_dependency_mismatch_between_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root / "data")
            control, candidates = self.portfolio_fixture(root, data)
            run = candidates[0]
            original_manifest = json.loads((run / "manifest.json").read_text())
            original_checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
            for key in ("numpy", "pillow", "torchvision"):
                with self.subTest(dependency=key):
                    checkpoint = {**original_checkpoint, "provenance": {**original_checkpoint["provenance"], key: "different-version"}}
                    torch.save(checkpoint, run / "checkpoint.pt")
                    manifest = {**original_manifest, key: "different-version", "checkpoint_sha256": digest(run / "checkpoint.pt")}
                    (run / "manifest.json").write_text(json.dumps(manifest))
                    result = self.invoke(data, *selection_arguments(control, candidates))
                    self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
