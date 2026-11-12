"""Load the report's aggregate evidence without conflating validation and test models."""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class FinalModel:
    arch: str
    params: int
    train_rows: int
    checkpoint_sha256: str
    submission_ref: int
    private_accuracy: float
    public_accuracy: float


@dataclass(frozen=True)
class ValidationModel:
    arch: str
    params: int
    train_rows: int
    checkpoint_sha256: str
    split_sha256: str
    rows: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.rows


@dataclass(frozen=True)
class ProjectEvidence:
    final: FinalModel
    validation: ValidationModel
    experiments: tuple[dict, ...]
    curves: dict[str, list[dict]]
    confusion: tuple[tuple[int, ...], ...]


EVIDENCE_FILES = frozenset({
    "summary.json", "kaggle-submissions.csv", "learning-curves.json",
    "validation-confusion.json", "final-full-config.json", "final-full-manifest.json",
    "final-validation-config.json", "final-validation-manifest.json",
})


def load_evidence(directory: Path) -> ProjectEvidence:
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if name not in EVIDENCE_FILES or name in checksums:
            raise ValueError(f"Invalid evidence checksum entry {name}")
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Evidence checksum mismatch for {name}")
        checksums[name] = expected
    if checksums.keys() != EVIDENCE_FILES:
        raise ValueError("Evidence checksum list is incomplete")

    summary = json.loads((directory / "summary.json").read_text())
    full = json.loads((directory / "final-full-manifest.json").read_text())
    val = json.loads((directory / "final-validation-manifest.json").read_text())
    full_config = json.loads((directory / "final-full-config.json").read_text())
    val_config = json.loads((directory / "final-validation-config.json").read_text())
    counts = json.loads((directory / "validation-confusion.json").read_text())
    curves = json.loads((directory / "learning-curves.json").read_text())
    with (directory / "kaggle-submissions.csv").open(newline="") as stream:
        scores = {int(row["ref"]): row for row in csv.DictReader(stream)}
    score = scores[summary["final_submission_ref"]]
    private = float(score["privateScore"])
    public = float(score["publicScore"])
    if not 0 <= private <= 1 or not 0 <= public <= 1:
        raise ValueError("Kaggle accuracy outside [0, 1]")
    if private != summary["final_test_accuracy"] or public != summary["final_public_accuracy"]:
        raise ValueError("Summary and Kaggle scores disagree")
    if full["train_rows"] != 50000 or full["val_rows"] != 0 or full["final_val_correct"] is not None:
        raise ValueError("Final model must have no held-out validation score")
    if full_config["split"] != "all" or val_config["split"] != "val":
        raise ValueError("Final and validation configurations have the wrong splits")
    if full_config["arch"] != val_config["arch"] or full_config["arch"] != "wrn22_4":
        raise ValueError("Final and validation architectures disagree")
    if not full["params"] == val["params"] == summary["final_params"] <= 5000000:
        raise ValueError("Parameter counts disagree or exceed the student cap")
    if full["checkpoint_sha256"] != summary["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint provenance disagrees")
    if val["checkpoint_sha256"] != summary["validation_checkpoint_sha256"]:
        raise ValueError("Validation checkpoint provenance disagrees")
    if full["checkpoint_sha256"] == val["checkpoint_sha256"]:
        raise ValueError("Final and validation checkpoints must be distinct")
    if val["train_rows"] != 45000 or val["val_rows"] != 5000:
        raise ValueError("Unexpected validation split size")
    matrix = counts["counts"]
    if len(matrix) != 10 or any(len(row) != 10 for row in matrix):
        raise ValueError("Confusion matrix must have ten rows and columns")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(10))
    if total != summary["validation_rows"] or total != val["val_rows"]:
        raise ValueError("Confusion total and validation row counts disagree")
    if correct != summary["validation_correct"] or correct != val["final_val_correct"]:
        raise ValueError("Confusion diagonal and validation accuracy disagree")
    if counts["checkpoint_sha256"] != val["checkpoint_sha256"]:
        raise ValueError("Confusion counts belong to another checkpoint")
    if val["split_sha256"] != full["split_sha256"]:
        raise ValueError("Final and validation split fingerprints disagree")
    if any(sum(row) != 500 for row in matrix):
        raise ValueError("Validation classes must each contain 500 rows")
    for experiment in summary["experiments"]:
        history = curves[experiment["name"]]
        if [row["epoch"] for row in history] != list(range(1, experiment["epochs"] + 1)):
            raise ValueError(f"Incomplete epoch history for {experiment['name']}")
        if experiment["validation_accuracy"] != experiment["validation_correct"] / experiment["validation_rows"]:
            raise ValueError(f"Invalid validation accuracy for {experiment['name']}")
        if experiment["submission_ref"] is not None:
            recorded = scores[experiment["submission_ref"]]
            if experiment["public_accuracy"] != float(recorded["publicScore"]) or experiment["private_accuracy"] != float(recorded["privateScore"]):
                raise ValueError(f"Experiment and Kaggle scores disagree for {experiment['name']}")
    return ProjectEvidence(
        final=FinalModel(full_config["arch"], full["params"], full["train_rows"], full["checkpoint_sha256"], int(score["ref"]), private, public),
        validation=ValidationModel(val_config["arch"], val["params"], val["train_rows"], val["checkpoint_sha256"], val["split_sha256"], total, correct),
        experiments=tuple(summary["experiments"]), curves=curves,
        confusion=tuple(tuple(row) for row in matrix),
    )
