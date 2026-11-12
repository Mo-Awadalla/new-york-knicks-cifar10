# The New York Knicks CIFAR-10 project

Mohamed Awadalla's solo entry for NYU Tandon Deep Learning F26.

## Current state

- Kaggle team: **The New York Knicks**.
- Competition: `cifar-10-res-net-architecture-challenge-dl`.
- Globally available CLI: Kaggle 2.2.5, installed with `uv tool install --python 3.12 --with six kaggle`.
- `src/knicks_cifar.py` is the training, inference, validation, and staging CLI. `src/knicks_methods.py` owns training augmentation, losses, and optimizer updates; `src/knicks_tta.py` owns the fixed, non-adaptive AugMix32 inference policy. Staging bundles these canonical sources into an offline Kaggle script.
- Eight submissions have scored 30.20%, 84.90%, 86.30%, 87.10%, 87.90%, 87.40%, 87.50%, and 88.40%. The best public score, 88.40%, is a WRN-22-4 CutMix student distilled from two teacher-only WRN-28-10 models (CutMix and SAM with Cutout) trained on NYU HPC, all trained from scratch for 200 epochs on all 50,000 official images. Its 45,000-row recipe reached 4,872 of 5,000 validation images (97.44%), the best submittable validation result. Earlier entries: WRN-22-4 CutMix 87.90%, SAM with Cutout 87.40%, and a student distilled from those smaller models 87.50%. All four have 4,298,970 trainable parameters. The public split covers about 1,000 images, so differences of a few tenths of a point are a handful of images.
- The corrected portfolio implements AugMix with JSD, SAM with Cutout, and CutMix alone, plus ensemble distillation. All 54 local behavior tests pass, including isolated execution of the generated scripts, selection boundaries, teacher restrictions, and frozen-TTA probability averaging, per-ID reproducibility, partial batches, and model-state preservation. Real three-epoch control and AugMix calibrations produced identical checkpoint file hashes on Kaggle and Colab. The control's 119 saved tensors and all 5,000 validation predictions were independently checked exact. Calibration results are not competitive accuracy results.
- Full preparation plan: [docs/preparation.md](docs/preparation.md).
- Online brief: [competition preparation](https://chatgpt.com/space/page_df3cfcf7b8708191b2f30e5075fa0f89).

## Project 1 submission

The final WRN-22-4 student has **4,298,970 trainable parameters**. Kaggle submission `57968810` reports **85.90% private test accuracy** and **88.40% public test accuracy**. The separate 45,000-row validation model reached **97.44%**, or 4,872 of 5,000 images. The full-data model has no held-out validation score.

- [Report PDF](report/project1.pdf) contains the architecture, methodology, results, lessons, and citations.
- [Executed analysis notebook](notebooks/project1_analysis.ipynb) prints the parameter count and explicitly labeled test scores, with learning curves and validation analysis.
- [Reproduction instructions](docs/reproduction.md) explain portable analysis and checkpoint reevaluation.
- `results/` contains aggregate evidence and the captured Kaggle score response. It does not contain competition images or per-image labels.

```bash
uv sync --python 3.12 --extra report
uv run python scripts/build_notebook.py --out runs/results-only.ipynb
uv run python scripts/build_report.py
uv run python scripts/verify_submission.py
```

Private test labels are unavailable. The notebook reports the official Kaggle private score rather than claiming a local hidden-test evaluation. A real validation checkpoint replay is a separate, labeled operation. No staff confirmation of an exemption for the two larger training-only teachers was found.

## Run the pipeline

Install the local environment from the project root.

```bash
uv sync --python 3.12
uv run python src/knicks_cifar.py --help
kaggle quota
```

Complete **Phone verify** in [Kaggle Settings](https://www.kaggle.com/settings) before launching GPU work. CLI authorization and a displayed GPU quota do not establish accelerator access. The notebook editor's GPU option must be enabled, and a real diagnostic must report CUDA before training starts.

Training reads only the official competition mount under `/kaggle/input`. Staging attaches `cifar-10-res-net-architecture-challenge-dl` and disables internet. Do not substitute torchvision's CIFAR-10 dataset. Bulk local download returned HTTP 404; downloading all images individually is unnecessary.

For a local smoke run, download two official training images and two test images, plus the three CSV manifests.

```bash
COMP=cifar-10-res-net-architecture-challenge-dl
kaggle competitions download "$COMP" -f train_labels.csv -p data
kaggle competitions download "$COMP" -f test.csv -p data
kaggle competitions download "$COMP" -f sample_submission.csv -p data
kaggle competitions download "$COMP" -f train_images/train_00000.png -p data
kaggle competitions download "$COMP" -f train_images/train_00001.png -p data
kaggle competitions download "$COMP" -f test_images/test_0000.png -p data
kaggle competitions download "$COMP" -f test_images/test_0001.png -p data
uv run python src/knicks_cifar.py smoke --data data --out runs/local-smoke --device cpu
```

Smoke outputs are marked ineligible for submission. The prediction command refuses smoke, calibration, and incomplete checkpoints.

Stage a three-epoch calibration before full training.

```bash
uv run python src/knicks_cifar.py stage --config configs/basic.json \
  --run-id calibration --out runs/calibration/kernel --mode calibrate
kaggle kernels push -p runs/calibration/kernel --accelerator NvidiaTeslaT4 -t 1800
uv run python scripts/wait_kaggle.py moawadalla/knicks-calibration \
  runs/calibration/output --max-seconds 1800
```

The waiter downloads real output files and logs. A wait timeout does not cancel the remote kernel.

When a trained model has generated and validated predictions:

```bash
COMP=cifar-10-res-net-architecture-challenge-dl
kaggle competitions submit "$COMP" \
  -f submissions/submission.csv -m "ResNet baseline"
```

Keep credentials outside this repository. The submission command uploads predictions and consumes a submission attempt; run it only for a deliberate candidate.

## Compute choice

**NYU HPC is optional.** The course PDF lists it as an optional resource. A single GPU is an appropriate starting point for this compact CIFAR-10 model; no distributed cluster setup is needed.

Start with Kaggle GPU notebooks if your account has accelerator access and sufficient quota. Use NYU HPC if you already have access and need more experiments, longer uninterrupted jobs, or more control over the environment. Use the Mac for editing, inspecting data, and short correctness checks. Benchmark a representative training epoch on the chosen machine before budgeting all experiments.

The CLI can upload and execute a prepared notebook or script on Kaggle GPU infrastructure; installing the CLI alone does not provide a local GPU or start training. Account quota and accelerator availability remain constraints. Check `kaggle quota` rather than assuming a fixed allowance.

## Compare recipes

`configs/basic.json` and `configs/cutout.json` differ only in augmentation. Both specify WRN-16-4, 200 epochs, seed 0, SGD with Nesterov momentum, five warmup epochs, and cosine decay. Choose the shared epoch horizon after full-data GPU calibration, before either comparison. Preserve the same horizon for final retraining.

The immutable hash-ranked split contains 45,000 training rows and 5,000 validation rows, with 500 validation rows per class. Normalization uses only the rows trained on. Validation has no augmentation. For the cosine recipe, per-epoch validation evaluates the ordinary model. For SWA, `manifest.json` records `final_val_*` metrics from the saved, reloaded averaged checkpoint. Compare the paired arms with those final-checkpoint metrics, not the last cosine-model row in `metrics.jsonl`.

The current [portfolio protocol](docs/portfolio-protocol.md) compares three prespecified WRN-22-4 treatments against a fresh matched control using seed 1. The user chose coverage of all three treatments instead of three-seed confirmation. A candidate qualifies only if its final checkpoint reaches at least 4,843 correct predictions out of 5,000 and improves on the fresh control by at least 25 correct predictions. These bounds preserve a 0.5-point gain over both the incumbent and the control. Select the highest qualifying final validation accuracy, breaking a tie by lower measured complete elapsed time. Public scores do not choose the recipe. One seed does not establish stability across seeds.

Cutout improved final validation accuracy from 95.38% to 96.36%. WRN-22-4 with Cutout finished at 96.32%; WRN-16-4 with RandAugment N=2, M=9 plus Cutout finished at 96.20%. Neither follow-up improved on Cutout on the fixed split. The all-data Cutout model scored 87.10% publicly, 0.80 points above the 45,000-row Cutout submission. It has no held-out validation score because it trained on all 50,000 official images. The last observed first-place public score was 91.40%, 4.30 points above our best submission at that check.

The new recipes are `configs/wrn22-augmix.json`, `configs/wrn22-sam.json`, and `configs/wrn22-cutmix.json`. They retain the shared WRN-22-4 architecture, fixed split, and 200-epoch horizon. AugMix replaces Cutout and adds the reference three-view JSD objective. SAM retains Cutout and changes the optimizer update. CutMix replaces Cutout without smoothing, SWA, or TTA. The implemented SWA option remains available but is deferred for this portfolio.

The instantiated WRN-16-4 has 2,748,890 trainable parameters, WRN-22-4 has 4,298,970, and the wider, shallower WRN-16-5 has 4,289,754. All have residual connections and random initialization. The runtime checks the exact count and the 5,000,000 cap before optimization.

```bash
uv run python src/knicks_cifar.py stage --config configs/basic.json \
  --run-id basic --out runs/basic/kernel
kaggle kernels push -p runs/basic/kernel --accelerator NvidiaTeslaT4
uv run python scripts/wait_kaggle.py moawadalla/knicks-basic runs/basic/output \
  --max-seconds 27000
```

Run directories contain the configuration, split IDs, per-epoch metrics, final checkpoint, validation predictions, and provenance manifest. Existing output directories are rejected. The default internal training deadline is seven hours. Timed-out runs remain ineligible.

Staged production training reloads the completed checkpoint, predicts all official test images, and validates `submission.csv`. `--mode calibrate` runs three epochs and never generates a candidate CSV. `--mode compare` preserves the configured full epoch horizon, keeps the validation split, and also disables candidate eligibility and prediction. `--mode smoke` exercises the available official images without making a submission candidate.

The full training-to-2,000-row-CSV rehearsal passed on a Tesla T4. Its one-epoch validation accuracy was 43.04%. At the user's request, its actual predictions became the first developmental submission and scored 30.20%. Count processed batches, successful optimizer updates, and AMP skips separately; the learning-rate schedule advances by processed batches. Prediction requests deterministic cuDNN convolutions. The initial CPU checkpoint replay checked two locally available official test images; a later CPU predict-only run of the full-data small-teacher KD checkpoint reproduced its complete 2,000-row GPU-generated submission CSV byte for byte.

For final training, copy the selected configuration and set `split` to `all`. Keep the selected augmentation, architecture, and epoch horizon. Train from scratch on all 50,000 official training rows. The standalone `predict` command accepts a completed checkpoint with its adjacent manifest and reads the official test images.

```bash
uv run python src/knicks_cifar.py predict --checkpoint PATH/checkpoint.pt \
  --data PATH_TO_OFFICIAL_DATA --out submissions/submission.csv --device cpu
uv run python src/knicks_cifar.py check --submission submissions/submission.csv \
  --sample data/sample_submission.csv
```

See [Wide Residual Networks](https://arxiv.org/abs/1605.07146) and [Cutout](https://arxiv.org/abs/1708.04552) for the referenced methods. Projection shortcuts follow the [WRN reference implementation](https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py). Cutout samples centers anywhere in the image and clips masks at its edges, following the [Cutout reference implementation](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py). This repository implements the network directly and does not load pretrained weights.

## Run the corrected portfolio

Calibrate each recipe before committing the measured GPU budget to full runs. Use a fresh output directory and run ID each time.

```bash
uv run python src/knicks_cifar.py stage --config configs/wrn22-sam.json \
  --run-id sam-calibration-example --out runs/sam-calibration-example/kernel \
  --mode calibrate --seed 1
kaggle kernels push -p runs/sam-calibration-example/kernel --accelerator NvidiaTeslaT4
```

Generated scripts contain the canonical modules listed in `SOURCE_MODULES` and verify their hashes. They can also run locally on the official smoke images with `python -I kernel.py --working WORK_DIR --data data --device cpu` after staging in smoke mode.

Colab runs the same canonical CLI in an isolated environment. The verified paired environment uses PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, NumPy `2.1.3`, and Pillow `13.0.0`. Official training images are transferred privately with their CSV and image hashes. No Kaggle credentials or evaluation images are transferred. The Colab AugMix comparison uses `train --config CONFIG --data OFFICIAL_DATA_ROOT --out OUTPUT --device cuda --max-seconds 32400 --calibrate`. The longer timeout does not change the 200-epoch horizon or learning-rate schedule.

After the fresh control and all three full screening runs complete, compare their reloaded-checkpoint predictions using `scripts/compare_portfolio.py`. Pass the directories containing each run's `manifest.json`, `config.json`, `split.json`, `checkpoint.pt`, and `val_pred.npy`.

```bash
uv run python scripts/compare_portfolio.py --data data \
  --control CONTROL_OUTPUT --candidate AUGMIX_OUTPUT \
  --candidate SAM_OUTPUT --candidate CUTMIX_OUTPUT
```

The comparator verifies checkpoint-bound configuration and provenance, the canonical fixed split, source and runtime agreement, final prediction accuracy, seed 1, and all three distinct treatments. It applies both validation margins and returns the selected candidate or no adoption. It never reads evaluation labels or public scores. Comparison checkpoints remain ineligible for submission. Retrain a selected recipe from scratch on all permitted training images before final inference.

AugMix adapts the [authors' training and operations](https://github.com/google-research/augmix), retaining Apache attribution. SAM follows the [paper](https://arxiv.org/abs/2010.01412) with exact perturbed-weight and BatchNorm-buffer rollback. CutMix follows the [authors' patch and label-area method](https://github.com/clovaai/CutMix-PyTorch), retaining MIT attribution.

## Distill the teacher ensemble

`configs/kd-cutmix22-val.json` and `configs/kd-cutmix22-full.json` train one WRN-22-4 CutMix student against the averaged temperature-4 predictions of three from-scratch teachers: SAM-Cutout WRN-22-4, CutMix WRN-22-4, and Cutout WRN-16-4. The loss is 0.9 times the scaled KL divergence plus 0.1 times the CutMix loss ([Hinton et al.](https://arxiv.org/abs/1503.02531)). Teachers see the student's augmented pixels under their own normalization statistics. Only the student is submitted; its 4,298,970 parameters satisfy the cap. The course rules permit distillation from teachers trained from scratch on permitted data.

Each config pins teacher checkpoint names and SHA-256 hashes. The validation config uses teachers trained on the 45,000 training rows only, and the loader rejects any teacher that saw validation rows. The full-data config uses the full-data teachers. Majority voting of the 45,000-row teachers reached 4,864 of 5,000 validation images, against 4,846 for the best single teacher.

Teacher checkpoints are uploaded as the private Kaggle dataset `moawadalla/knicks-kd-teachers-001` and found under `/kaggle/input` at run time. Pass `--teachers DIR` to search a local directory instead.

```bash
uv run python src/knicks_cifar.py stage --config configs/kd-cutmix22-full.json \
  --run-id kd-cutmix22-full-001 --out runs/kd-cutmix22-full-001/kernel \
  --teachers-dataset moawadalla/knicks-kd-teachers-001
```

## Train large teachers on NYU HPC cloud bursting

Course work runs on HPC Cloud Bursting through Open OnDemand (`ood.burst.hpc.nyu.edu`, Slurm account `cs_gy_6923-2026fa`), not the Torch cluster. A JupyterLab session on `c24m170-a100-2` provides two A100 40 GB GPUs; its terminal runs the scripts in `hpc/` under `/scratch/$USER/knicks`.

- `hpc/setup.sh` checks the private training export's archive hash and all 50,000 image hashes, and builds the Kaggle-matched environment (PyTorch 2.11.0+cu128, torchvision 0.26.0, NumPy 2.1.3, Pillow 13.0.0). No evaluation images are transferred.
- `hpc/teachers.sh` trains six teacher-only WRN-28-10 models (36,479,194 parameters): CutMix, SAM with Cutout, and Cutout, each on the 45,000-row split and on all 50,000 rows. `wrn28_10` exceeds the 5,000,000 cap, so its checkpoints are never submission-eligible and `predict` refuses them.
- `hpc/ensemble_eval.py` reports single and soft-ensemble validation accuracy for 45,000-row teachers, so the teacher set is chosen on validation.
- `hpc/student_config.py NAME EPOCHS SPLIT STEM...` generates WRN-22-4 distillation configs with hash-pinned teachers. `hpc/run_student.sh GPU CONFIG_NAME` trains a single configured student; use fresh run names. WRN-16-5 uses the explicit configs below.

Full-data students are scored on Kaggle, where the test images live. Upload `NAME.pt` with its run manifest as `NAME.manifest.json` to a private dataset, then stage a CPU kernel that pins the checkpoint hash:

```bash
uv run python src/knicks_cifar.py stage-predict --run-id predict-NAME --out runs/predict-NAME/kernel \
  --dataset moawadalla/DATASET --checkpoint PATH/NAME.pt
kaggle kernels push -p runs/predict-NAME/kernel
```

Use a kernel run ID that differs from the dataset slug; Kaggle rejects a kernel whose slug matches an existing dataset. A parity run on the full-data distilled student reproduced its GPU-generated `submission.csv` byte for byte.


## Wider student and targeted fine-tuning

`configs/kd-wrn16-5-val.json` changes only the architecture of the 200-epoch big-teacher WRN-22-4 recipe to WRN-16-5. It keeps the same seed, split, CutMix policy, teacher checkpoint hashes, optimizer, schedule, and distillation objective. `configs/kd-wrn16-5-full.json` is the corresponding conditional full-data recipe, not authorization to launch it before validation. Local official-image smoke passed with real, available under-cap teachers: 4,289,754 parameters, an applied distillation update, and exact checkpoint-reload logits. The actual WRN-28-10 teacher recipe completed three A100 calibration epochs with 1,055 applied updates and one AMP skip, at about 13 seconds per epoch. The full 200-epoch comparison finished at 4,858 of 5,000 (97.16%), 14 fewer correct than the incumbent: 35 examples fixed and 49 broken. Its checkpoint hash, source/configuration, split, and saved predictions were independently verified after download. No full-data retrain or submission was made. All three campaign checkpoints are saved under `runs/wider-student-001/recovered/`; the decision is in `runs/wider-student-001/decision.json`.

The prospective plan in `runs/wider-student-001/plan.json` retains a 25-image (0.5-point) improvement margin over the current 4,872-correct incumbent: the wider student's final checkpoint must reach at least 4,897 of 5,000. This is an operational selection rule, not a statistical confidence claim or a forecast of the public score.

`scripts/experiment_finetune.py` compares two five-epoch, training-data-only updates from the same 45,000-row checkpoint: classifier-only versus final residual block, final BatchNorm affine parameters, and classifier. Both use crop/flip, SGD, a 0.001-to-zero cosine schedule, and frozen BatchNorm buffers. Selection uses final checkpoints and the existing 2,000-image tuning partition, not intermediate validation peaks. On A100, classifier-only retained 4,872 of 5,000 correct with identical predictions; last-block reached 4,873, but its 3,000-image audit gain was zero (three fixed, three broken) and overall cross-entropy worsened. Neither met the predeclared 15-image audit margin. Both applied 1,760 updates without skips, preserved frozen tensors, and reloaded with identical logits. Saved predictions were independently rescored locally. The original model remains unchanged. Use CPU or CUDA; the tested MPS backend crashes in frozen-BatchNorm backward. Results: `runs/wider-student-001/finetune-result.json`.

## Direct weight-edit experiment

`scripts/experiment_weight_edits.py` tests 260 label-independent edits to `head.weight`: individual weight scaling, neuron suppression/scaling, magnitude pruning, and seeded random perturbations. It requires a completed 45,000-row checkpoint and matching manifests. It selects one edit on 2,000 validation images and audits it once on the remaining 3,000, with 200/300 images per class. The partition and selection rule are recorded before evaluation. Do not tune further against the audit results: these images were also used in earlier recipe comparisons, so they are not a pristine test set.

```bash
uv run python scripts/experiment_weight_edits.py --source PATH_TO_VALIDATION_RUN \
  --data PATH_TO_OFFICIAL_TRAINING_DATA --out runs/weight-edit-new --device mps
```

The first run on the 97.44% validation model selected suppression of classifier input neuron 23 (10 outgoing weights). Tuning accuracy improved from 1,950 to 1,952 of 2,000, but audit accuracy fell from 2,922 to 2,918 of 3,000. The edit was rejected. All 5,000 baseline predictions matched the saved GPU predictions, and a real forward pass with the edited network matched the cached-feature calculation. The source checkpoint remained unchanged; the experimental weights are not submission-eligible. Artifacts are in `runs/weight-edits-001/experiment/`.

## Frozen AugMix32 inference

The [researched inference-only candidate](docs/research-90-decision.md) is implemented in `src/knicks_tta.py`, separately from the training AugMix policy. It follows the MEMO authors' mild level-1 CIFAR view generator, uses 32 views and per-image-ID local RNG streams with seed 20261105, applies checkpoint normalization, and averages probabilities in FP32. It never updates weights or BatchNorm statistics and verifies all parameters and buffers remain unchanged. There is no MEMO adaptation step or extra clean 33rd view.

```bash
uv run python scripts/validate_frozen_tta.py --source runs/weight-edits-001/source \
  --data runs/weight-edits-001/data --out runs/augmix32-reproduction-NEW --device mps --batch 128
```

The one-shot decision required positive net accuracy gains on both existing 2,000/3,000 validation partitions; these partitions have historical reuse and are not an independent confirmation set. The completed run failed: tuning fell from 1,950 to 1,944 correct, audit from 2,922 to 2,917, and the total from 4,872 to 4,861 (97.44% to 97.22%). It fixed 24 errors but broke 35; lower NLL did not override the accuracy rule. Saved predictions were independently rescored. No policy/seed retuning, full-data TTA inference, or Kaggle submission followed. The 88.40% single-view incumbent is unchanged.

Verification included 54 passing tests, 64 reference-matched augmentation views on two official training images with zero numerical difference, CPU/MPS agreement, exact replay of all 5,000 baseline predictions, and unchanged source/model state. Evidence is under `runs/augmix32-001/validation/`, with the final decision in `runs/augmix32-001/decision.json`.

For reproduction, `predict` and `stage-predict` accept `--inference augmix32`; the default remains `single`. AugMix predictions write a `*.inference.json` sidecar binding the policy, source/checkpoint/CSV hashes, versions, and state-preservation checks. The staged CPU inference bundle was prepared but not launched after validation failed. The flag is not evidence of model-selection success or staff permission.

## Project folders

| Folder | Purpose |
| --- | --- |
| `docs/` | Preparation plan and methodology |
| `data/` | Official competition files, excluded from Git |
| `src/` | Standalone residual model, data handling, training, and inference |
| `configs/` | Controlled basic and Cutout recipes |
| `notebooks/` | Executed project analysis and documented checkpoint reevaluation |
| `runs/` | Checkpoints and training logs, excluded from Git |
| `scripts/` | Bounded Kaggle status and artifact retrieval |
| `submissions/` | Generated prediction CSVs, excluded from Git |

## Deadlines and constraints

- Competition closes November 6, 2026, at 11:59 p.m. Eastern.
- Report due November 11, 2026, at 11:59 p.m. Brightspace displays this deadline in the account's New York time zone, GMT-5.
- Residual CNN trained from scratch, at most 5 million trainable parameters.
- Only staff-permitted training data; no external datasets or pretrained weights.
- At most 5 submissions per day for the entire entry.
- Report PDF at most 8 pages including citations, with a public GitHub repository link on page one.
- The project is 20% of the course grade, with 10 points for the report and 10 for the codebase. The TA's accuracy target is at least 80%. The final student's 85.90% private test score clears that target, but report and code quality also count.

## References

- [Competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)
- [Course PDF](https://brightspace.nyu.edu/content/enforced/619245-FA26_CS-GY_6953_1_I_crse/project1.pdf)
- [Official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
- [Competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/competitions.md)
- [Notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md)

Prepared November 1, 2026.
