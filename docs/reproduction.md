# Reproduce the Project 1 results

## Rerun the submitted analysis

Use Python 3.12 and uv from the repository root.

```bash
uv sync --python 3.12 --extra report
uv run python scripts/build_notebook.py --out runs/results-only.ipynb
uv run python scripts/build_report.py
uv run python scripts/verify_submission.py
```

The executed notebook is `notebooks/project1_analysis.ipynb`. The report is `report/project1.pdf`. The report source is `report/project1.md`.

The portable rerun writes `runs/results-only.ipynb`. It preserves the submitted notebook's actual checkpoint-replay outputs. The verifier checks the submitted notebook, not this aggregate-only copy.

The portable analysis uses the checked-in aggregate evidence in `results/`. It constructs the real architecture and counts its trainable parameters. It regenerates learning curves and validation analysis. It prints the Kaggle-reported private test accuracy, public test accuracy, and the separate validation score. It does not pretend to evaluate unavailable hidden test labels.

## Interpret the results

| Quantity | Value | Source |
| --- | --- | --- |
| Final architecture | WRN-22-4 | Full-data checkpoint configuration |
| Trainable parameters | 4,298,970 | Model construction and checkpoint manifest |
| Private test accuracy | 85.90% | Kaggle submission 57968810, privateScore |
| Public test accuracy | 88.40% | Kaggle submission 57968810, publicScore |
| Recipe validation accuracy | 97.44%, 4,872 of 5,000 | Separate model trained on 45,000 rows |

The final model trained on all 50,000 official training images for 200 epochs. The validation model trained only on the fixed 45,000-row training split. Evaluating the full-data model on those 5,000 validation rows would be training-set evaluation, not held-out validation.

The original 2,000-row test manifest has no labels. Kaggle reports scores on its public and private portions. The final student's private score is the reported final test result. No local file contains the hidden test labels.

## Reevaluate the validation checkpoint

The notebook supports an explicit checkpoint replay. Use only the official competition data obtained through an authorized Kaggle account. Keep the original `train_labels.csv`, `test.csv`, and `sample_submission.csv` manifests. Put the PNG files under the data directory. Do not replace them with torchvision CIFAR-10.

The validation checkpoint must have SHA-256 `42fa268b43159b51eb9a936f9d00e91a5cfdd0194e7abb884bb79333658f9719`. Its original run directory also contains `manifest.json`, `config.json`, `split.json`, `metrics.jsonl`, and `val_pred.npy`.

The notebook documents its replay environment variables next to the evaluation cell. A requested replay must fail on missing files or mismatched provenance. The replay computes predictions from the actual saved model and compares them with the original predictions. It does not retrain the model or recompute the private test score.

```bash
KNICKS_DATA_ROOT=OFFICIAL_DATA_DIRECTORY \
KNICKS_VAL_CHECKPOINT=VALIDATION_RUN/checkpoint.pt \
KNICKS_FINAL_CHECKPOINT=FINAL_RUN/checkpoint.pt \
KNICKS_DEVICE=cpu uv run python scripts/build_notebook.py --out notebooks/project1_analysis.ipynb
```

Set `KNICKS_DEVICE=mps` or `cuda` for the corresponding available accelerator. Omit `KNICKS_FINAL_CHECKPOINT` if only the validation checkpoint is available. Use `--out runs/results-only.ipynb` for a portable results-only copy without replacing the submitted replay notebook.

Checkpoints and official images are excluded from Git. The repository contains model, training, inference, and analysis code, plus aggregate run evidence. Exact checkpoint replay requires the original artifacts. Source-only users can regenerate plots and inspect the executed replay output but cannot independently reproduce trained weights without training.

## Reproduce training

The selected validation and full-data student configurations are `results/final-validation-config.json` and `results/final-full-config.json`. The associated manifests pin the original source, data CSV, teacher checkpoint, and student checkpoint hashes.

The original training sources are preserved in Git commit `6cfe397`. Its `src/knicks_cifar.py` SHA-256 matches the training manifest's `7d1d610a1d0a4c25046f07ab89fc49ba763a82549cd85b9bead9021cf6ddbba5`. Later source additions include the rejected AugMix32 inference experiment. They are not evidence that the submitted model used that policy.

Train each of the two scratch WRN-28-10 teachers with its matching split before training the student. Their recipes are in `configs/teacher-wrn28-cutmix-val.json`, `configs/teacher-wrn28-samcutout-val.json`, `configs/teacher-wrn28-cutmix-full.json`, and `configs/teacher-wrn28-samcutout-full.json`.

```bash
uv run python src/knicks_cifar.py train \
  --config configs/teacher-wrn28-cutmix-val.json \
  --data data --out runs/reproduction-teacher-cutmix-val --device cuda
uv run python src/knicks_cifar.py train \
  --config configs/teacher-wrn28-samcutout-val.json \
  --data data --out runs/reproduction-teacher-sam-val --device cuda
```

Fresh teachers can have different checkpoint hashes on another software or hardware stack. Stage the new validation teachers under explicit names, then generate and use a student configuration that pins their actual hashes.

```bash
mkdir -p teachers
cp runs/reproduction-teacher-cutmix-val/checkpoint.pt teachers/repro-cutmix-45k.pt
cp runs/reproduction-teacher-sam-val/checkpoint.pt teachers/repro-sam-45k.pt
uv run python hpc/student_config.py reproduction-student-val 200 val repro-cutmix repro-sam
uv run python src/knicks_cifar.py train \
  --config configs/reproduction-student-val.json \
  --teachers teachers --data data \
  --out runs/reproduction-student-val --device cuda
```

Use fresh full-data teachers for the final student. Do not reuse full-data teachers in a validation experiment.

```bash
uv run python src/knicks_cifar.py train \
  --config configs/teacher-wrn28-cutmix-full.json \
  --data data --out runs/reproduction-teacher-cutmix-full --device cuda
uv run python src/knicks_cifar.py train \
  --config configs/teacher-wrn28-samcutout-full.json \
  --data data --out runs/reproduction-teacher-sam-full --device cuda
cp runs/reproduction-teacher-cutmix-full/checkpoint.pt teachers/repro-cutmix-full.pt
cp runs/reproduction-teacher-sam-full/checkpoint.pt teachers/repro-sam-full.pt
uv run python hpc/student_config.py reproduction-student-full 200 full repro-cutmix repro-sam
uv run python src/knicks_cifar.py train \
  --config configs/reproduction-student-full.json \
  --teachers teachers --data data \
  --out runs/reproduction-student-full --device cuda
```

The generated configurations describe new reproduction runs. Keep the historical evidence in `results/` unchanged. If the original hash-matched teacher artifacts are available, use the pinned `results/final-validation-config.json` or `results/final-full-config.json` with their original filenames instead.

The recorded GPU environment used PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3, and Pillow 13.0.0. The training CLI creates a fresh output directory, records provenance, saves the final checkpoint, and verifies checkpoint reload. Hardware and library changes can affect bitwise reproducibility.

Two training-only teachers each have 36,479,194 parameters. Only the 4,298,970-parameter student performs final inference. The course's teacher-cap exemption was not explicitly confirmed. This is a disclosed rules uncertainty, not a claim of staff approval.

## Reproduce final predictions

The final checkpoint SHA-256 is `9a5a28fa5e00dc4e929d5f0b25f7c3bd14a5123c43d6cb6409407be039460a49`. Place its matching manifest beside it as `manifest.json`.

```bash
uv run python src/knicks_cifar.py predict \
  --checkpoint FINAL_RUN/checkpoint.pt --data data \
  --out submissions/reproduction.csv --device cpu
uv run python src/knicks_cifar.py check \
  --submission submissions/reproduction.csv --sample data/sample_submission.csv
```

Final inference uses one view and one student. The rejected AugMix32 policy is not part of the submitted model. Do not submit new predictions to reproduce a historical score. Use `results/kaggle-submissions.csv` to inspect the original scores.

## Refresh aggregate evidence from local originals

```bash
uv run python scripts/collect_report_evidence.py
```

This command requires the original ignored `runs/` artifacts, official training labels, and authenticated Kaggle CLI access. It hash-checks eight checkpoints, independently scores the saved validation predictions, captures Kaggle submission scores, and exports aggregates. It never uploads images or predictions. The ordinary notebook and PDF builds do not require Kaggle credentials.
