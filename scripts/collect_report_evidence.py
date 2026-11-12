"""Export aggregate report evidence from the original local run artifacts."""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knicks_cifar import read_manifest, sha256_file, stratified_split

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SOURCES = [
    ("Crop and flip", "wrn16_4", "runs/basic-200-001/output/knicks-basic-200-001-output", 57883025),
    ("Cutout", "wrn16_4", "runs/cutout-200-001/output/knicks-cutout-200-001-output", 57883535),
    ("WRN-22-4 Cutout", "wrn22_4", "runs/wrn22-cutout-200-001/reauthenticated-output/knicks-wrn22-cutout-200-001-output", None),
    ("Matched Cutout control", "wrn22_4", "runs/portfolio-control22-screen-s1/recovered-output/knicks-portfolio-control22-screen-s1-output", None),
    ("SAM with Cutout", "wrn22_4", "runs/portfolio-sam22-screen-s1/reauthenticated-output/knicks-portfolio-sam22-screen-s1-output", 57955108),
    ("CutMix", "wrn22_4", "runs/portfolio-cutmix22-screen-s1/reauthenticated-output/knicks-portfolio-cutmix22-screen-s1-output", 57952636),
    ("Small-teacher distillation", "wrn22_4", "runs/kd-cutmix22-val-001/output/knicks-kd-cutmix22-val-001-output", 57959124),
    ("Large-teacher distillation", "wrn22_4", "runs/weight-edits-001/source", 57968810),
]
raw_scores = subprocess.run(
    ["kaggle", "competitions", "submissions", "cifar-10-res-net-architecture-challenge-dl", "--csv"],
    check=True, capture_output=True, text=True,
).stdout
(OUT / "kaggle-submissions.csv").write_text(raw_scores)
scores = {int(row["ref"]): row for row in csv.DictReader(io.StringIO(raw_scores))}
training = read_manifest(ROOT / "runs/weight-edits-001/data/train_labels.csv", True)
split = stratified_split(training)
labels = np.asarray(training.labels)[list(split.val_rows)]
experiments = []
curves = {}
for name, arch, relative, ref in SOURCES:
    run = ROOT / relative
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_epochs"] == config["epochs"]
    digest = sha256_file(run / "checkpoint.pt")
    assert digest == manifest["checkpoint_sha256"]
    run_split = json.loads((run / "split.json").read_text())
    assert run_split["val_ids"] == [training.ids[row] for row in split.val_rows]
    predictions = np.load(run / "val_pred.npy", allow_pickle=False)
    assert predictions.shape == labels.shape == (5000,)
    correct = int((predictions == labels).sum())
    score = scores.get(ref, {})
    experiments.append({
        "name": name, "arch": arch, "params": manifest["params"],
        "epochs": config["epochs"], "seed": config["seed"],
        "validation_rows": manifest["val_rows"], "validation_correct": correct,
        "validation_accuracy": correct / len(labels),
        "submission_ref": ref,
        "public_accuracy": float(score["publicScore"]) if score else None,
        "private_accuracy": float(score["privateScore"]) if score else None,
        "test_training_rows": 45000 if ref in (57883025, 57883535) else 50000 if ref else None,
        "source_run": relative, "validation_checkpoint_sha256": digest,
    })
    curves[name] = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    assert len(curves[name]) == config["epochs"]

validation = ROOT / "runs/weight-edits-001/source"
full_checkpoint = ROOT / "runs/kd-w28x2-e200-full/hpc/kdw28-full.pt"
full_manifest = json.loads((full_checkpoint.with_suffix(".manifest.json")).read_text())
assert sha256_file(full_checkpoint) == full_manifest["checkpoint_sha256"]
full = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
(OUT / "final-full-config.json").write_text(json.dumps(full["config"], indent=2) + "\n")
for target, source in [
    ("final-full-manifest.json", full_checkpoint.with_suffix(".manifest.json")),
    ("final-validation-manifest.json", validation / "manifest.json"),
    ("final-validation-config.json", validation / "config.json"),
]:
    (OUT / target).write_text(source.read_text())
saved_split = json.loads((validation / "split.json").read_text())
assert saved_split["val_ids"] == [training.ids[row] for row in split.val_rows]
predictions = np.load(validation / "val_pred.npy", allow_pickle=False)
assert predictions.shape == labels.shape == (5000,)
assert int((predictions == labels).sum()) == 4872
confusion = np.zeros((10, 10), dtype=np.int64)
np.add.at(confusion, (labels, predictions), 1)
(OUT / "validation-confusion.json").write_text(json.dumps({
    "rows": "true labels", "columns": "predicted labels", "class_labels": list(range(10)),
    "counts": confusion.tolist(), "checkpoint_sha256": experiments[-1]["validation_checkpoint_sha256"],
}, indent=2) + "\n")
summary = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "competition": "cifar-10-res-net-architecture-challenge-dl",
    "repository": "https://github.com/Mo-Awadalla/new-york-knicks-cifar10",
    "score_source": "Kaggle competitions submissions API; hidden labels are not available locally",
    "final_submission_ref": 57968810,
    "final_test_accuracy": float(scores[57968810]["privateScore"]),
    "final_public_accuracy": float(scores[57968810]["publicScore"]),
    "final_params": full_manifest["params"],
    "final_checkpoint_sha256": full_manifest["checkpoint_sha256"],
    "final_submission_sha256": sha256_file(ROOT / "runs/kd-w28x2-e200-full/output/submission.csv"),
    "validation_checkpoint_sha256": experiments[-1]["validation_checkpoint_sha256"],
    "validation_correct": 4872, "validation_rows": 5000,
    "experiments": experiments,
    "limitations": [
        "Private test accuracy is Kaggle-reported, not a local hidden-label evaluation.",
        "Validation and full-data scores use different checkpoints of the same recipe.",
        "Experiments used one seed per comparison; validation was reused across experiments.",
        "No staff confirmation of the larger-teacher parameter-cap interpretation was found.",
        "Published artifacts contain aggregates, not staff-provided images or per-image labels.",
    ],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "learning-curves.json").write_text(json.dumps(curves, indent=2) + "\n")
artifacts = sorted(path for path in OUT.iterdir() if path.suffix in {".json", ".csv"})
(OUT / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in artifacts))
print(json.dumps({key: summary[key] for key in ("final_submission_ref", "final_test_accuracy", "final_public_accuracy", "final_params", "validation_correct", "validation_rows")}, indent=2))
print(f"Exported {len(experiments)} hash-verified experiment records to results/.")
