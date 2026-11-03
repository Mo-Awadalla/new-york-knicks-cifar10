# Validation-led three-experiment portfolio

## Scope

The user selected validation-led selection after providing an external proposal. This protocol replaces the proposal's public-score-led selection and combined third candidate. After real Kaggle and Colab calibration, the user explicitly chose one matched seed per arm instead of three-seed confirmation. This selection rule is fixed before any new full screening results. Calibration results are verification and budget inputs, not model-selection results.

Use only the official staff-provided training images. No external datasets, pretrained weights, label lookup, copied labels for evaluation predictions, ensembles, or uncleared test-time augmentation. Evaluation-image statistics must not select training policies. The existing 87.10% public full-data Cutout model remains the fallback.

## Shared comparison

- Architecture is the existing pre-activation WRN-22-4, counted at 4,298,970 trainable parameters.
- Use the existing immutable stratified 45,000/5,000 split, with normalization derived only from the training rows.
- Train every screening arm from random initialization for 200 epochs.
- Keep batch size 128, SGD with Nesterov momentum 0.9, maximum learning rate 0.1, weight decay 0.0005, five warmup epochs, and per-batch cosine decay.
- The control is WRN-22-4 with crop, horizontal flip, and clipped 16-by-16 Cutout. The archived seed-zero control scored 96.32% validation. It is context, not a replacement for a fresh matched control.
- Validation and inference have no augmentation. The primary metric is accuracy of the saved, strictly reloaded checkpoint on all 5,000 fixed validation IDs.
- Separate model-initialization, ordering, crop/flip, and method RNG streams. Match streams between arms where the algorithms share a transformation.

## Candidate arms

| Arm | Intervention relative to the control | Exclusions |
| --- | --- | --- |
| AugMix with JSD | Replace Cutout with three-chain AugMix, depth sampled from 1 to 3, severity 3, Dirichlet(1,1,1) chain weights and Beta(1,1) mixture weight. Use a clean view and two independently mixed views in one concatenated model forward. Loss is clean cross-entropy plus 12 times three-view Jensen-Shannon divergence. | No Cutout, SAM, smoothing, SWA, or TTA. |
| SAM with Cutout | Keep Cutout and add nonadaptive SAM with rho 0.05 and the existing SGD base optimizer. Reuse the identical augmented minibatch in both passes. | No changed architecture, additional augmentation, smoothing, SWA, or TTA. |
| CutMix | Replace Cutout with minibatch CutMix, Beta(1,1) box sampling, and label weights calculated from the actual clipped patch area. Initially use probability 0.5 as an explicit adaptation of the authors' CIFAR-100 example, not a claimed CIFAR-10 optimum. | No Cutout, SAM, smoothing, SWA, or TTA. |

AugMix uses the nine default reference operations. Grayscale and extra ColorJitter are not silently added. Its probability and KL computations use numerically safe FP32 arithmetic. SAM updates BatchNorm running statistics only on the first unperturbed forward, keeps batch statistics for the perturbed forward without running-stat updates, and restores original weights on every exit path. AMP overflow must never leave perturbed weights eligible for export.

These are adaptations to WRN-22-4 and our fixed-split recipe. Paper results are not forecasts of Kaggle accuracy.

## Verification and budget gates

1. Verify algorithm behavior and preserve existing config, checkpoint, offline staging, and inference contracts.
2. Exercise each new production training path in a short real GPU calibration. Record processed source images, model views, successful optimizer updates, AMP skips, peak memory, and complete elapsed time.
3. Do not replace JSD with a different loss after timing a run while still calling it the same treatment. If calibration makes a recipe infeasible, report the blocked recipe and proposed tradeoff before reducing its scope.
4. Budget the complete single-seed screening block and selected full-data retraining, inference, and recovery before full launches. Verified Kaggle allowance is 22.14 GPU hours. The approved allocation projects 14.7 Kaggle GPU hours including the worst selected final-training reserve, plus 8.3 Colab hours for AugMix screening. These are extrapolations with 20% epoch-time headroom and 15 minutes of overhead per job. Preserve at least four hours for final training, inference, and recovery, increasing the reserve for AugMix. Future free Colab sessions are not assumed.
5. If the complete allocation does not fit, do not silently shorten an arm's horizon or promise all three arms will complete. Present the measured tradeoff.

## Prospective selection

- Screen all three prespecified candidates against a fresh same-source seed-1 control. Every arm completes 200 epochs. Do not add seeds or choose an intermediate epoch because of a result.
- Select using the saved, strictly reloaded checkpoint's predictions on all 5,000 fixed validation IDs. A candidate qualifies only if it exceeds the fresh control by at least 25 correct predictions and exceeds the incumbent's 4,818 correct predictions by at least 25. The absolute qualification floor is therefore 4,843 of 5,000, or 96.86%.
- Among qualifying candidates, select the highest final validation accuracy. Break a tie in favor of lower measured complete elapsed time. If none qualifies, retain the existing full-data Cutout fallback.
- Report all screening outcomes and paired prediction disagreements. One matched seed does not establish stability across seeds. The operator chose this coverage-versus-replication tradeoff before full results.
- Public leaderboard scores do not choose the recipe. Retrain the selected recipe from scratch on all 50,000 permitted training images for the same 200 epochs with seed 1. Verify predictions before a deliberate official submission.

## Primary references

- [Official AugMix training and JSD implementation](https://github.com/google-research/augmix/blob/master/cifar.py).
- [Official AugMix operations](https://github.com/google-research/augmix/blob/master/augmentations.py).
- [SAM paper and official implementation](https://github.com/google-research/sam).
- [PyTorch SAM example with first-pass BatchNorm updates](https://github.com/davda54/sam/blob/main/example/train.py). This is a third-party implementation, not the original authors' code.
- [Official CutMix implementation and CIFAR-100 probability example](https://github.com/clovaai/CutMix-PyTorch).
- [AdA paper, including the AugMix and SAM comparison table](https://arxiv.org/abs/2104.01086).
