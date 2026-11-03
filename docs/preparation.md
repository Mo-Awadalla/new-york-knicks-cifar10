# The New York Knicks solo competition preparation

Mohamed Awadalla is competing solo as **The New York Knicks** in NYU Tandon Deep Learning F26, the CIFAR-10 ResNet architecture challenge. As of November 1, 2026, the entry has one member and no submissions.

# Deadlines and registration

- **Competition closes November 6, 2026 at 11:59 p.m. Eastern**, as displayed by Kaggle.

- **Report due November 9, 2026 before 11:59 p.m.**, according to the Fall 2026 course PDF. The PDF does not state a timezone; confirm the Brightspace assignment timezone.

- Use **The New York Knicks** consistently on Kaggle, the staff Google registration form, and the report.

- Course rules cap teams at **3 members**, despite Kaggle's team interface allowing 5. Confirm the roster and course registration before adding members. Membership changes after the registration deadline require staff approval.

- The project is worth 20% of the course grade.

# Rules that shape the work

Train a convolutional network with residual connections from scratch, with **at most 5,000,000 trainable parameters**, including the classifier. Use only staff-permitted CIFAR-10 training data. Augmentation is allowed. External datasets and pretrained weights are prohibited. Distillation is allowed only with teachers trained from scratch on permitted data.

Use a fixed validation split from training data to compare experiments. Never train on, manually label, externally look up labels for, or tune against hidden evaluation images outside the official Kaggle process. Public and private evaluation portions are 50% each; private accuracy determines final rankings. Cite referenced methods and code.

# Proposed work schedule

| Date | Work | Completion check |
| --- | --- | --- |
| Nov 1 | Confirm solo registration; download official competition files; inspect CSV schemas and filenames; establish reproducible split and baseline | One batch loads correctly, parameter budget passes, baseline training and inference run |
| Nov 2 | Finish baseline; validate preprocessing and image ID alignment; make the first meaningful submission | Validation accuracy, checkpoint, configuration, and valid CSV archived |
| Nov 3 | Compare one architectural change and one augmentation change separately | Same split and training budget; record accuracy and runtime |
| Nov 4 | Extend the strongest configuration and check seed stability if compute permits | Improvement holds on validation, rather than just the public leaderboard |
| Nov 5 | Freeze architecture and training recipe; train final model on all permitted training images using the selected epoch budget | Final checkpoint, parameter count, command, seed, and logs saved |
| Nov 6 | Generate and verify final predictions; submit with several hours of buffer; choose final candidates | Successful Kaggle result and selected final submissions verified |
| Nov 7–8 | Complete report and reproduce predictions from the repository | Report within 8 pages; clean reproduction succeeds |
| Nov 9 | Final review and individual report uploads | Every member verifies a Brightspace upload receipt |

If the GPU budget is limited, prioritize a complete reproducible baseline, one controlled improvement, and final training before broad searches.

# Proposed experiments

Start with a compact CIFAR-style residual network: a 3 by 3 stride-1 stem, no initial large-image max pool, global average pooling, and a 10-class head. A WRN-16-4 style network is a candidate; compute the exact instantiated parameter count before training. This is a proposed starting point, not a verified model or promised accuracy.

Use a stratified, fixed 45,000/5,000 train/validation split if the supplied labels contain 50,000 training examples. Keep validation transforms deterministic and separate from training augmentation. A starting recipe to test is SGD with momentum 0.9, batch size 128, learning rate 0.1, weight decay 0.0005, cosine decay with short warmup, and 150–200 epochs subject to measured GPU throughput. Begin with random crop and horizontal flip; use normalization derived from the training split or a clearly documented permitted convention.

| Experiment | Change from baseline | Decision |
| --- | --- | --- |
| Baseline | Compact residual network and basic augmentation | Establish working pipeline and validation accuracy |
| Augmentation | Add one of Cutout, MixUp, or CutMix | Keep only if validation improves under comparable training |
| Architecture | Change depth or width while retaining parameter headroom | Compare accuracy, training time, and exact parameter count |
| Final recipe | Retrain selected configuration and epoch budget on all permitted training data | Preserve full provenance and generate final predictions |

Log seed, split indices, code revision, architecture, parameter count, optimizer, schedule, batch size, epochs, augmentation, validation accuracy, training time, and checkpoint path for every run. Avoid ensembling until staff clarify how the 5M budget applies to multiple models.

# Submission checks

The Data page lists `train_images`, `test_images`, `train_labels.csv`, `test.csv`, and `sample_submission.csv`. The current sample has **2,000 evaluation rows** with IDs 0–1999. Treat the downloaded sample and test manifest as the authority; do not infer filename ordering lexicographically.

- [ ] Save the file as `submission.csv` with exactly `id,label` columns and no extra index column.

- [ ] Match every sample ID exactly once, in sample order, with no missing, duplicate, or extra IDs.

- [ ] Predict an integer class 0–9 for every evaluation image.

- [ ] Use evaluation mode and deterministic inference preprocessing consistent with training normalization.

- [ ] Count all trainable parameters and assert the total is at most 5,000,000.

- [ ] Keep CSV, checkpoint, configuration, and inference command together.

- [ ] Respect the **5 submissions per day per team** limit.

- [ ] Verify successful processing and select up to **2 final submissions** on Kaggle.

Class mapping: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.

# Report and repository

The report must be a **PDF of no more than 8 pages, including figures, tables, and citations**. Every member must upload a copy to Brightspace. Put a link to the **publicly accessible GitHub repository on page one**.

Suggested page budget: page 1 team, repository link, overview and findings; pages 2–3 architecture and design rationale; pages 4–5 data, training and controlled experiments; page 6 results and parameter count; page 7 limitations and lessons; page 8 references. Adjust to fit the actual evidence.

The repository needs model implementation, preprocessing and augmentation, training and inference commands, exact configuration, parameter count, citations, and instructions that reproduce the submitted predictions. Preserve checkpoints and logs for staff verification. The open Loss-Guided-GFN repository concerns GFlowNet experiments; it is not a CIFAR-10 ResNet starter.

# Items still to resolve

- Solo participant: Mohamed Awadalla. No teammate recruitment is needed.

- Staff Google registration form and registration deadline; neither was linked on the inspected Project 1 page.

- The verified allowance before AugMix recovery was 17.57 Kaggle GPU hours. The calibration-based recovery envelope is 7.4 hours for AugMix and 8.7 hours reserved for the worst selected final retrain, inference, and recovery. These are planning estimates, not guaranteed full-run durations.

- Brightspace report assignment timezone and upload destination.

Seven scored submissions reached 30.20%, 84.90%, 86.30%, 87.10%, 87.90%, 87.40%, and 87.50%. The 87.90% entry is WRN-22-4 CutMix and the 87.40% entry is WRN-22-4 SAM with Cutout, each trained for 200 epochs from scratch on all 50,000 permitted images with seed 1. Neither recipe cleared the locked validation gate; the user directed both retrains after the AugMix screen finished at 4,788 of 5,000. The 87.50% entry is a WRN-22-4 CutMix student distilled from those full-data SAM and CutMix models plus the full-data WRN-16-4 Cutout model. Its 45,000-row version, distilled only from 45,000-row teachers, reached 4,866 of 5,000 validation images, 29 above the matching CutMix run and 43 above the fresh control. Full-data models have no held-out validation score.

The user selected the corrected validation-led AugMix-JSD, SAM-Cutout, and CutMix-only portfolio on WRN-22-4. These methods and the single-seed selector have 43 local tests passing. All three methods completed their three-epoch Kaggle GPU calibrations with 1,056 optimizer updates each and zero skipped updates. These are calibration results, not competitive accuracy results.

The control, SAM, and CutMix completed all 200 epochs. The selector checked their actual checkpoint-bound provenance and recomputed saved validation predictions against the official labels.

| Model | Validation correct out of 5,000 | Internal validation accuracy | Gain over fresh control |
| --- | --- | --- | --- |
| Fresh Cutout control | 4,823 | 96.46% | Reference |
| SAM with Cutout | 4,846 | 96.92% | 0.46 percentage points |
| CutMix | 4,837 | 96.74% | 0.28 percentage points |

Neither completed treatment reaches the locked 25-image improvement over the fresh control. SAM misses by two validation images. No new public submission has been made.

The Colab AugMix observer stopped updating after epoch 68. Reconnecting returned an empty runtime without the training files or a final checkpoint. The exact interruption cause is unknown. The stale local observer was cancelled, and the newly allocated empty diagnostic runtime was released without deleting the saved notebook. AugMix was relaunched on Kaggle from scratch with the same seed, source, recipe, fixed split, and 200-epoch horizon. Its kernel reports `RUNNING`. `runs/portfolio-scores-and-recovery-001.json` records the evidence and bounded recovery budget.

Earlier authentication failures stopped the local Kaggle status monitors, not the control, SAM, or CutMix training. Renewed OAuth restored access. The existing completed SAM and CutMix outputs were downloaded and verified without retraining them. This monitoring recovery is separate from restarting the unrecovered AugMix attempt after Colab's files became unavailable. The exact authentication rejection cause remains unconfirmed.

Colab training readiness passed in an isolated virtual environment on a Tesla T4. Its exact PyTorch, torchvision, NumPy, Pillow, and CUDA versions match Kaggle. The private CPU-only Kaggle export contains all 50,000 official training images and no evaluation images. Archive, CSV, and individual image hashes passed after transfer.

The canonical three-epoch Colab control reproduces all 119 saved tensors, all 5,000 validation predictions, and every epoch metric except elapsed time exactly. Its checkpoint file hash also matches Kaggle's. AugMix completed three real Colab epochs with zero skipped updates and an identical checkpoint file hash to its Kaggle calibration. The downloaded calibration archive passed its SHA-256 check and is extracted under `runs/colab-calibrations-001`.

The user explicitly chose one matched seed per arm instead of three-seed confirmation. The prospective rule in `docs/portfolio-protocol.md` requires at least 4,843 correct validation predictions and at least 25 more correct predictions than the fresh control. With the verified control at 4,823 correct, the effective qualification threshold is now 4,848 of 5,000, or 96.96%. This applies the locked rule without changing it. A selected recipe is retrained from scratch on all 50,000 official images for 200 epochs with seed 1. If none qualifies, the existing 87.10% full-data Cutout model remains the fallback. Public scores do not select the recipe. One seed does not establish seed stability. Future free Colab hours are not assumed.

# Sources

- [Kaggle competition overview](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview)

- [Competition rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules)

- [Competition data](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data)

- [Team settings](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/team)

- [Submissions](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/submissions)

- [Fall 2026 course project PDF](https://brightspace.nyu.edu/content/enforced/619245-FA26_CS-GY_6953_1_I_crse/project1.pdf)

- [Colab FAQ, including GPU availability and runtime limits](https://research.google.com/colaboratory/faq.html)

Rules and course sources checked November 1, 2026. Colab access, training readiness, and compute allocation checked November 2, 2026.
