# Compact residual models under the 5M cap

This note compares residual architectures and training methods that might fit the course limit. Published CIFAR-10 test results describe different models, data splits, and training budgets. They do not predict the competition score.

## Current model and results

The current network is a CIFAR-style pre-activation WRN-16-4. It has two residual blocks in each of three stages, stage widths 64, 128, and 256, and 2,748,890 trainable parameters including the classifier ([model code](../src/knicks_cifar.py)). The tested WRN-22-4 has 4,298,970 parameters.

The fixed 45K/5K split produced 95.38% validation accuracy for WRN-16-4 with basic augmentation and 96.36% with Cutout. WRN-22-4 with Cutout reached 96.32%. WRN-16-4 with RandAugment plus Cutout reached 96.20%. The latter two models were not adopted under the frozen 0.5-point gate. That gate had no variance analysis, so differences below a point are weak evidence for ranking recipes ([experiment record](preparation.md)).

The submitted WRN-16-4 Cutout model scored 86.3% on the public Kaggle portion. Its 10.06-point gap from fixed-split validation does not identify the cause. It does not prove that the competition test data are CIFAR-10.1 or otherwise shifted. No hidden labels or external image matches belong in this comparison.

## What the primary sources establish

### Width and depth in Wide ResNets

The WRN paper argues for reducing depth and increasing residual-block width. It reports CIFAR-10 error rates for several widths and depths. Its WRN-40-2 has 2.2M parameters and 5.33% error with ZCA preprocessing. The paper also reports that width gains recur across its tested depths, while very deep, wide settings can lose accuracy. These findings support a width-versus-depth hypothesis, not a prediction for WRN-16-5 or this competition ([paper](https://arxiv.org/abs/1605.07146)).

The authors' CIFAR training script uses 200 epochs, initial learning rate 0.1, Nesterov momentum, and learning-rate drops by 0.2 at epochs 60, 120, and 160. It also uses reflection padding for random crops ([reference script](https://github.com/szagoruyko/wide-residual-networks/blob/master/scripts/train_cifar.sh)). The current recipe keeps the 200-epoch horizon, batch size 128, SGD with Nesterov momentum 0.9, learning rate 0.1, and weight decay 0.0005, but replaces the step schedule with five warmup epochs and cosine decay ([configuration](../configs/cutout.json), [training loop](../src/knicks_cifar.py)). The horizon and crop padding match the reference more closely than the learning-rate schedule does.

### Shake-Shake

Shake-Shake randomly blends two residual branches in the forward pass and uses independently sampled branch weights for the backward pass. Its 26-layer, 2-branch, 32-channel model is reported at about 2.9M parameters. The paper reports 3.55% CIFAR-10 test error for Shake-Shake-Image and 3.47% for Shake-Even-Image. It trains with horizontal flips and translations for 1,800 epochs, batch size 128, initial learning rate 0.2, and cosine annealing without restarts on two GPUs ([paper](https://arxiv.org/abs/1705.07485)).

The 26 2x32d model is a single multi-branch network, not a test-time ensemble. Its published parameter count is rounded, not an exact count for a new implementation. At 200 epochs, an adaptation would use one ninth of the paper's horizon. There is no evidence that the reported accuracy survives that reduction. The 26 2x96d model, at 26.2M parameters, is over the course cap and must not be used to justify a compact result.

### PyramidNet and ShakeDrop

PyramidNet grows channel width gradually across residual units. The authors report additive PyramidNet-110 with alpha 84 at 3.8M parameters and 4.27% CIFAR-10 error. Their alpha-270 model has 28.3M parameters and is over budget ([author implementation and results](https://github.com/jhkim89/PyramidNet)). The repository's counts are rounded to 0.1M.

ShakeDrop perturbs residual-branch outputs during training. Its authors report 3.33% CIFAR-10 error for PyramidNet-110 alpha 270, which is the 28.3M-parameter over-budget model. Their CIFAR recipe runs 300 epochs with Nesterov SGD, momentum 0.9, batch size 128, weight decay 0.0001, and learning-rate drops at epochs 150 and 225. It uses random crop and horizontal flip. It reports no under-5M PyramidNet plus ShakeDrop benchmark ([paper](https://arxiv.org/abs/1802.02375), [implementation](https://github.com/imenurok/ShakeDrop)).

A low-alpha PyramidNet plus ShakeDrop is therefore an adaptation, not a published compact result. ShakeDrop adds no trainable weights, but the exact count of that combined implementation remains unverified. The nearest author-reported baseline is PyramidNet-110 alpha 84 at 3.8M parameters, before adding the regularizer. The 300-epoch schedule is 50% longer than the current horizon, and the reference used four GPUs. No matching local epoch-time measurement exists.

### Weight averaging and SAM

SWA averages checkpoints collected along SGD training while using a constant or cyclical learning rate in its averaging phase. It leaves one model at inference. The authors' CIFAR-10 results include PreResNet-164, reported at 1.7M parameters, which is below the course cap. Its test accuracy rose from 95.28% ± 0.10 for SGD to 95.56% ± 0.11 at one SGD training budget under SWA, across the authors' runs. This is direct evidence for the method on a compact residual model, but not on this WRN-16-4, this Cutout recipe, or the competition evaluation distribution. The paper also reports a WRN-28-10 result at 36.5M parameters, which exceeds the cap. SWA adds no inference-time ensemble. Batch-normalization statistics need one post-training recalibration pass ([paper](https://arxiv.org/abs/1803.05407), [authors' results and code](https://github.com/timgaripov/swa)).

SAM updates require two gradient computations per training step. Its CIFAR benchmarks use WRN-28-10, Shake-Shake 26 2x96d, and PyramidNet variants, all larger than 5M. The paper explicitly doubles non-SAM epochs when comparing compute. SAM is plausible in principle, but the evidence does not show a compact, 200-epoch gain large enough to justify roughly twice the per-step gradient work here ([paper](https://arxiv.org/abs/2010.01412), [official code](https://github.com/google-research/sam)).

### Augmentation strength and schedule

The current Cutout mask is 16 by 16 pixels. It can cross image boundaries, and it is applied after normalization. This matches the Cutout paper's CIFAR-10 mask size and boundary handling. The paper chose length 16 on WRN-28-10 and trained for 200 epochs with SGD, Nesterov momentum 0.9, weight decay 0.0005, batch size 128, and step drops. Its result does not establish that the same mask is optimal for WRN-16-4 ([paper](https://arxiv.org/abs/1708.04552)).

The current RandAugment call uses two operations and magnitude 9. Torchvision's default magnitude-bin count is 31. The RandAugment paper defines operation magnitudes on its own scales, and finds that the best strength depends on model and training-set size. It tunes on held-out CIFAR examples and uses Cutout as part of its CIFAR baseline ([paper](https://arxiv.org/abs/1909.13719), [Torchvision API](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html)). The numeric value 9 is not a verified match to a paper setting. The local RandAugment-plus-Cutout result is lower than Cutout alone on the fixed split, so more augmentation search is not the best-supported next change.

## Candidate changes

These are proposed adaptations. None has been trained in this project. Keep the fixed 45K/5K split and final-epoch comparison. Do not infer a Kaggle score from CIFAR-10 test results.

| Priority | Candidate and mechanism | Parameter count | Budget and compute | Evidence and limit |
| --- | --- | --- | --- | --- |
| 1 | Compare WRN-16-4 Cutout with the same recipe plus SWA. Weight averaging tests whether a late-training average improves generalization without adding an inference model. | 2,748,890, exact for both arms. | Fix the budget at 200 epochs. Keep the incumbent schedule through epoch 149, then test the paper's constant SWA learning rate of 0.01 while averaging checkpoints. Recompute batch-normalization statistics using permitted training images. Match the same three fresh seeds across arms if the remaining schedule allows. | A 1.7M-parameter PreResNet-164 improved by 0.28 points at the paper's one-budget setting, but that result does not establish a gain for this architecture, Cutout setup, or leaderboard. Compare matched seeds and report per-seed results. |
| 2 | Change WRN-16-4 to WRN-16-5 and keep Cutout. This tests width rather than more depth. | 4,289,754 by the current block-count formula. The runtime must instantiate and verify the count before training. | Match the current 200-epoch Cutout recipe. Measure one epoch before committing to a full run. | The WRN paper supports width as a design axis, but the exact model is untested here. WRN-22-4 finished 0.04 validation points below WRN-16-4. No width run is justified ahead of SWA evidence. |
| 3 | If time and implementation evidence remain, consider a compact PyramidNet-110 alpha-84 with ShakeDrop. | The published base network is reported as 3.8M rounded. The exact combined implementation count is unverified. | The reference schedule uses 300 epochs and four GPUs. Do not begin without checking that the implementation fits the remaining course deadline and accelerator quota. | The compact PyramidNet benchmark is vanilla, not ShakeDrop. The reported 3.33% ShakeDrop result uses the 28.3M alpha-270 model. This combination is a speculative adaptation. |

## Do not carry these headline results across

- WRN-28-10 has 36.5M parameters. Shake-Shake 26 2x96d has 26.2M. PyramidNet-110 alpha 270 has 28.3M. Their accuracy results violate the 5M limit and do not rank eligible models.
- Shake-Shake's 1,800 epochs and ShakeDrop's 300 epochs are not equal-budget comparisons with this 200-epoch recipe. Do not attribute their published performance to a 200-epoch adaptation.
- The under-cap SWA result is a stronger research lead than the previous note suggested. It still does not establish this project's expected gain. The cited WRN-28-10 SWA accuracy remains out of budget.
- Do not add pretrained weights, external training data, ensembles, or test-time augmentation. Published CIFAR test accuracy is not a forecast of either Kaggle half.

## Primary sources

- Zagoruyko and Komodakis, [Wide Residual Networks](https://arxiv.org/abs/1605.07146), and [authors' training script](https://github.com/szagoruyko/wide-residual-networks/blob/master/scripts/train_cifar.sh).
- Gastaldi, [Shake-Shake regularization](https://arxiv.org/abs/1705.07485).
- Han, Kim, and Kim, [Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915), and [authors' results](https://github.com/jhkim89/PyramidNet).
- Yamada et al., [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375).
- Izmailov et al., [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407).
- Foret et al., [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412).
- DeVries and Taylor, [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552).
- Cubuk et al., [RandAugment](https://arxiv.org/abs/1909.13719), and [Torchvision RandAugment API](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html).
