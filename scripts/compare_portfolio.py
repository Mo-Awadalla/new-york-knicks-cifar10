"""Compare verified portfolio checkpoints on the same held-out training images."""

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from knicks_cifar import read_manifest, stratified_split


SHARED_PROVENANCE = ("source_sha256", "source_sha256_by_file", "split_sha256", "csv_sha256", "torch", "cuda", "gpu", "deterministic_algorithms", "numpy", "pillow", "torchvision")
FIXED_CONFIG = {"arch": "wrn22_4", "split": "val", "epochs": 200, "batch": 128, "lr": 0.1, "momentum": 0.9, "wd": 0.0005, "warmup": 5, "strategy": "cosine"}
INCUMBENT_CORRECT = 4818
MINIMUM_CONTROL_DELTA = 25
MINIMUM_CANDIDATE_CORRECT = INCUMBENT_CORRECT + MINIMUM_CONTROL_DELTA


@dataclass(frozen=True)
class Run:
    path: Path
    config: dict
    manifest: dict
    validation_ids: tuple[int, ...]
    correct: np.ndarray


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_run(path, data):
    config = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    split = json.loads((path / "split.json").read_text())
    if any(config.get(key) != value for key, value in FIXED_CONFIG.items()):
        raise ValueError(f"{path}: configuration differs from the fixed portfolio recipe")
    if type(config.get("seed")) is not int or config["seed"] != 1:
        raise ValueError(f"{path}: requires protocol seed 1")
    if manifest["status"] != "complete" or manifest["completed_epochs"] != 200:
        raise ValueError(f"{path}: training is incomplete")
    if manifest["train_rows"] != 45000 or manifest["val_rows"] != 5000 or manifest["params"] != 4298970:
        raise ValueError(f"{path}: dataset counts or parameter budget do not match the protocol")
    if manifest.get("candidate_eligible") is not False or (path / "submission.csv").exists():
        raise ValueError(f"{path}: use the no-submission comparison mode")
    for name in ("train_labels.csv", "test.csv", "sample_submission.csv"):
        if manifest["csv_sha256"][name] != digest(data / name):
            raise ValueError(f"{path}: input manifest changed: {name}")
    if digest(path / "checkpoint.pt") != manifest["checkpoint_sha256"]:
        raise ValueError(f"{path}: checkpoint hash mismatch")
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["config"] != config or checkpoint["completed_epochs"] != 200:
        raise ValueError(f"{path}: checkpoint configuration or horizon differs from sidecars")
    if checkpoint["kind"] != "calibrate" or manifest["kind"] != checkpoint["kind"] or checkpoint["params"] != manifest["params"]:
        raise ValueError(f"{path}: checkpoint is not the declared comparison artifact")
    for key in SHARED_PROVENANCE:
        if key == "split_sha256":
            value = checkpoint["split_sha256"]
        else:
            value = checkpoint["provenance"].get(key)
        if value is None or value != manifest.get(key):
            raise ValueError(f"{path}: checkpoint provenance differs from sidecars: {key}")
    training = read_manifest(data / "train_labels.csv", True)
    canonical = stratified_split(training)
    ids = tuple(split["val_ids"])
    expected_ids = tuple(training.ids[index] for index in canonical.val_rows)
    expected_train = tuple(training.ids[index] for index in canonical.train_rows)
    if ids != expected_ids or tuple(split["train_ids"]) != expected_train:
        raise ValueError(f"{path}: split membership or order differs from the canonical fixed split")
    if split["validation_sha256"] != canonical.sha256 or manifest["split_sha256"] != canonical.sha256:
        raise ValueError(f"{path}: split fingerprint differs from canonical membership")
    target = np.asarray([training.labels[index] for index in canonical.val_rows])
    predicted = np.load(path / "val_pred.npy", allow_pickle=False)
    if predicted.shape != (5000,) or not np.issubdtype(predicted.dtype, np.integer) or not np.all((predicted >= 0) & (predicted <= 9)):
        raise ValueError(f"{path}: invalid validation predictions")
    correct = predicted == target
    if int(correct.sum()) != manifest.get("final_val_correct"):
        raise ValueError(f"{path}: final checkpoint metric disagrees with saved predictions")
    if not math.isclose(float(manifest["final_val_accuracy"]), float(correct.mean()), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{path}: final checkpoint accuracy disagrees with saved predictions")
    elapsed = float(manifest["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f"{path}: invalid elapsed time")
    return Run(path, config, manifest, ids, correct)


def recipe(config):
    return {key: value for key, value in config.items() if key not in {"aug", "optimizer", "seed"}}


def comparison(control, candidate):
    if control.config["aug"] != "cutout" or control.config.get("optimizer", "sgd") != "sgd" or control.config.get("strategy", "cosine") != "cosine":
        raise ValueError("control must be cosine-SGD with Cutout")
    treatment = (candidate.config["aug"], candidate.config.get("optimizer", "sgd"))
    names = {("augmix_jsd", "sgd"): "augmix", ("cutout", "sam"): "sam", ("cutmix", "sgd"): "cutmix"}
    if treatment not in names:
        raise ValueError("candidate must be one of the three preregistered portfolio arms")
    if recipe(control.config) != recipe(candidate.config) or control.config["seed"] != candidate.config["seed"]:
        raise ValueError("paired configuration or initialization seed differs outside the intervention")
    if control.validation_ids != candidate.validation_ids:
        raise ValueError("paired validation IDs or prediction ordering differ")
    for key in SHARED_PROVENANCE:
        if control.manifest.get(key) is None or control.manifest.get(key) != candidate.manifest.get(key):
            raise ValueError(f"paired runtime or provenance mismatch: {key}")
    delta = int(candidate.correct.sum()) - int(control.correct.sum())
    return {
        "name": names[treatment],
        "seed": control.config["seed"],
        "control_correct": int(control.correct.sum()),
        "candidate_correct": int(candidate.correct.sum()),
        "validation_rows": 5000,
        "delta_images": delta,
        "incumbent_delta_images": int(candidate.correct.sum()) - INCUMBENT_CORRECT,
        "incumbent_delta_percentage_points": (int(candidate.correct.sum()) - INCUMBENT_CORRECT) / 50,
        "qualified": delta >= MINIMUM_CONTROL_DELTA and int(candidate.correct.sum()) >= MINIMUM_CANDIDATE_CORRECT,
        "delta_percentage_points": delta / 50,
        "candidate_only_correct": int((candidate.correct & ~control.correct).sum()),
        "control_only_correct": int((control.correct & ~candidate.correct).sum()),
        "candidate_elapsed_seconds": float(candidate.manifest["elapsed_seconds"]),
        "control_path": str(control.path),
        "candidate_path": str(candidate.path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        if len(args.control) != 1:
            raise ValueError("selection requires exactly one control")
        control = load_run(args.control[0], args.data)
        rows = [comparison(control, load_run(path, args.data)) for path in args.candidate]
        if len(rows) != 3 or {row["name"] for row in rows} != {"augmix", "sam", "cutmix"}:
            raise ValueError("selection requires exactly the three distinct preregistered candidates")
        qualifying = [row for row in rows if row["qualified"]]
        selected = min(qualifying, key=lambda row: (-row["candidate_correct"], row["candidate_elapsed_seconds"])) if qualifying else None
        result = {
            "phase": "single_seed_selection",
            "results": rows,
            "incumbent_correct": INCUMBENT_CORRECT,
            "minimum_control_delta_images": MINIMUM_CONTROL_DELTA,
            "minimum_candidate_correct": MINIMUM_CANDIDATE_CORRECT,
            "minimum_incumbent_delta_images": MINIMUM_CANDIDATE_CORRECT - INCUMBENT_CORRECT,
            "selected_candidate": selected["name"] if selected else None,
            "adopt": selected is not None,
            "claim_limit": "Single-seed operational selection only, not a significance or leaderboard claim.",
        }
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
