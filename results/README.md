# Project 1 aggregate evidence

`summary.json` distinguishes the final full-data model from the separate validation model. `kaggle-submissions.csv` is the authenticated Kaggle CLI response captured by `scripts/collect_report_evidence.py`. The final student's submission reference is 57968810. Its private score is 0.85900 and its public score is 0.88400.

`learning-curves.json` contains original per-epoch records for eight completed experiments. `validation-confusion.json` contains aggregate counts from the validation model's saved predictions, independently scored against the official training labels. It has 5,000 total rows and 4,872 correct predictions.

`final-validation-config.json` and `final-validation-manifest.json` describe the 45,000-row model. `final-full-config.json` and `final-full-manifest.json` describe the submitted 50,000-row model. Each manifest records the corresponding checkpoint and teacher hashes. The submitted model has no held-out validation score.

`SHA256SUMS` covers every JSON and CSV evidence file. It detects changes since export. It is not a signature or independent proof of Kaggle's scoring.

This directory contains no official images, per-image training labels, hidden test labels, credentials, or checkpoints. Portable analysis regenerates figures from aggregates. Recomputing validation predictions requires the original checkpoint and authorized official data. Hidden test accuracy cannot be recomputed locally because the labels are not released.

The experiment table pairs each validation recipe with its associated test submission where one exists. The early crop-and-flip and Cutout submissions trained on 45,000 images. Later test submissions trained on all 50,000 images. Architecture, augmentation, teacher size, and training-row changes are not all controlled in that cross-submission table. Use the matched control rows for recipe comparisons, not a causal interpretation of every table delta.
