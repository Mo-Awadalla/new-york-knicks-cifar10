# Interpreting the validation and public-score gap

## What the scores show

The two submitted models are both from-scratch WRN-16-4 residual CNNs with 2,748,890 trainable parameters. The first used basic crop-and-flip augmentation. The second added 16-by-16 Cutout. Both used the same fixed 45,000/5,000 train-validation split and 200-epoch schedule. [The experiment record](../README.md#compare-recipes) reports the final-epoch validation scores and Kaggle public scores.

| Model | Validation accuracy, 5,000 images | Public accuracy, 1,000 images | Validation minus public |
| --- | ---: | ---: | ---: |
| WRN-16-4, basic augmentation | 95.38% | 84.9% | 10.48 points |
| WRN-16-4, Cutout | 96.36% | 86.3% | 10.06 points |

Cutout's observed gain over the basic model was 0.98 points on validation and 1.4 points on the public leaderboard. The direction agrees on this pair. It does not establish that validation will rank other recipes correctly. The model seed and split were shared, but the runs used separate augmentation and sample-order random streams. [The experiment record](../README.md#compare-recipes)

The scores support a large difference between performance on this fixed internal holdout and performance on this public evaluation subset. They do not identify the cause. Possible causes include a difference in image difficulty or class mix, a data-generation shift, repeated selection against the fixed validation set, or an unobserved data or scoring mismatch. These remain hypotheses. The pipeline checks found correct validation prediction counts and sample ordering. CPU replay matched two official test images, which is not a full test-pipeline proof. No hidden labels or test-image identities were checked.

The gaps are numerically similar to drops found in the CIFAR-10.1 study, but that similarity does not identify this competition's evaluation images as CIFAR-10.1 or as any particular dataset. Recht et al. collected about 2,000 new images by closely matching CIFAR-10's Tiny Images keyword distribution. They observed 4-to-10-point accuracy drops across models, while model rankings mostly persisted. They also found that some methods, including Cutout, gained more on the new test set than on the original set. This is evidence that internal accuracy gains can transfer despite a sizable absolute gap. It is not evidence about the source or labels of this competition's test images. [1]
The full-data Cutout model trained for 200 epochs on all 50,000 official training images and scored 87.10% publicly. This is 0.80 points above the 45,000-row Cutout submission. The full-data model has no internal validation score because it trained on all labeled images. First place currently scores 91.40%, leaving the team 4.30 points behind. [The experiment record](../README.md#current-state)

The public score covers only 1,000 of the competition's 2,000 evaluation images. The other 1,000 determine the private score. The full-data model's 87.10% public score does not establish its private rank. It remains 4.30 points below the current 91.40% public leader. [The project record](../README.md#current-state)

## How much confidence to put in the 0.5-point rule

The old 0.5-point adoption threshold is not justified by the current evidence. On 5,000 validation images, 0.5 points equals 25 net examples, and one example changes accuracy by 0.02 points. A paired calculation from saved validation predictions uses the same 5,000 IDs for each comparison. For each ID, d is candidate correctness minus Cutout correctness, with each encoded as one or zero. The intervals below use the reported normal approximation, mean(d) plus or minus 1.96 times sample SD(d) divided by the square root of 5,000. They describe one seed and do not include training-seed variation or multiple-comparison correction.

| Candidate relative to WRN-16-4 Cutout | Paired accuracy difference | Candidate-only correct | Cutout-only correct | Approximate 95% interval |
| --- | ---: | ---: | ---: | ---: |
| WRN-16-4 basic augmentation | -0.98 points | 69 | 118 | [-1.515, -0.445] |
| WRN-22-4 Cutout | -0.04 points | 79 | 81 | [-0.536, 0.456] |
| WRN-16-4 RandAugment N=2, M=9 plus Cutout | -0.16 points | 82 | 90 | [-0.674, 0.354] |

The basic-versus-Cutout interval excludes zero for this fixed seeded pair. The depth and RandAugment intervals include zero. These comparisons do not establish performance across retraining seeds or on natural shifts. A practical margin can be useful only if its cost or value is stated before results. The 0.5-point margin had no such justification. Dietterich's paired test accounts for shared test examples, but one fitted-model comparison does not estimate variation from training data or algorithm randomness. [2]

The paired results above sharpen the single-run comparisons. WRN-22-4 and RandAugment plus Cutout remain inconclusive on validation. The earlier 0.5-point rule would reject both, but it does not explain whether their small differences are real or worth acting on. Do not promote a candidate by changing the metric after seeing the results. [The experiment record](../README.md#compare-recipes)

## What prior augmentation work does and does not say

RandAugment's CIFAR-10 study uses horizontal flips, padded crops, and Cutout. It trains Wide ResNets for 200 epochs with learning rate 0.1, batch size 128, weight decay 5e-4, and cosine decay. The authors use N=3 and test magnitudes 4, 5, 7, 9, and 11. They report best magnitudes 4 for WRN-28-2 and 5 for WRN-28-10. Results average ten runs. The paper also reports N=2, M=9 for ImageNet ResNet-50, not CIFAR-10 WRNs. [3]

The project's N=2, M=9 plus Cutout run is therefore not a published CIFAR-10 RandAugment recipe. Its architecture and augmentation choice are an unvalidated adaptation. It shares the CIFAR paper's 200-epoch horizon, base learning rate, batch size, weight decay, crop-and-flip, and Cutout. The project adds five warmup epochs. The supplement does not identify the optimizer in its CIFAR training paragraph. The project uses SGD with Nesterov momentum. It uses WRN-16-4, not either WRN tested in the CIFAR table. The code applies RandAugment to each cropped, flipped uint8 image, then normalizes and applies Cutout. [3] [The experiment record](../README.md#compare-recipes)

The local `uv.lock` selects Torchvision 0.29.2. Kaggle uses its preinstalled libraries, and the completed run recorded PyTorch 2.11.0 but did not record its Torchvision version. The remote RandAugment implementation therefore has not been verified against the local version. The inspected [local-version Torchvision implementation](https://github.com/pytorch/vision/blob/v0.29.2/torchvision/transforms/autoaugment.py) uses 31 magnitude bins, nearest-neighbor interpolation by default, and `fill=None`. A matching N and M across libraries does not prove identical transformed pixels. Future records need the actual remote library version and operation defaults. [13]

AugMix targets corruption robustness and uncertainty as well as clean accuracy. Its CIFAR experiments use random crop and flip, three augmentation chains of one to three operations, mixtures of the chains, and a Jensen-Shannon consistency loss. The Wide ResNet experiment trains for 100 epochs. The paper evaluates corruption benchmarks as a distinct outcome from clean test accuracy. In its CIFAR-10-C ablation, combining AugMix with MixUp increased corruption error relative to AugMix alone. Do not stack AugMix and MixUp and then credit a change to either method. [4]

MixUp forms convex mixtures of two images and their labels. The CIFAR paper uses alpha=1, 200 epochs, batches of 128, SGD, and no dropout. Its Wide-ResNet-28-10 result is outside this competition's five-million-parameter cap, so the reported gain does not establish a gain for WRN-16-4. Larger alpha values can underfit. [5]

CutMix replaces a rectangle from one training image with a patch from another and mixes labels according to the actual patch area. Its CIFAR experiments include CIFAR-10 and CIFAR-100. The paper reports a CIFAR-10 gain, but its main CIFAR recipe uses a 26.8-million-parameter PyramidNet-200 for 300 epochs. Its small-model CIFAR-100 results include a 1.1-million-parameter ResNet-110. These results motivate a bounded test under this competition's architecture and data rules. They do not predict its outcome. [6]

Cutout, MixUp, CutMix, and RandAugment alter different parts of the input or target. Combining them changes the strength and type of regularization. Cutout plus MixUp helped in one CIFAR-100 experiment, but that result does not establish that every combination helps. The CutMix paper reports that incorrect area-to-label weighting and several CutMix variants degraded results. Keep the first comparison to one augmentation change. [6]

## Synthetic corruptions as auxiliary evidence

A fixed corruption suite generated from the permitted 5,000 held-out training images would provide useful, controlled evidence about sensitivity to those specified transformations. CIFAR-10-C defines 15 corruption types at five severity levels. The same clean image and inherited label can be evaluated under every fixed transformation. No external images, labels, or hidden competition data are needed. [7]

This would not establish performance on naturally collected images. Evidence on synthetic-to-natural transfer is mixed and depends on the shift. Taori et al. found that aggregate synthetic robustness was at most weakly predictive of natural-shift robustness across their ImageNet testbed. Hendrycks et al. found that some methods that helped on synthetic corruptions also helped on real blurry images and ImageNet-R, while other natural shifts showed little benefit. Laugros et al. found that natural-shift correlation depended on the synthetic corruption selection. Their balanced, larger categories correlated better in their ImageNet-100 experiments, but their result does not validate a CIFAR-10 suite or this competition's ranking. None of these studies identifies this competition's hidden evaluation set. [8] [9] [10]

Use the corruption suite as a diagnostic, not as a post hoc way to rescue RandAugment or any other candidate. Freeze the transformations, severity levels, aggregation, and decision rule before evaluating candidate models. Report clean accuracy and corruption accuracy separately. Resample or group uncertainty by original source image because the five severities and multiple corruptions derived from one image are not independent observations. High severities may also obscure class cues, so preserve per-type and per-severity results instead of relying only on one average. Laugros et al. also show that category balance and benchmark size affect measured correlations. Their ImageNet-derived categories should not be copied as CIFAR categories without evidence. [10]


## Recommended next experiment

Do not launch another single-seed augmentation or depth sweep. The strongest primary-source lead is Stochastic Weight Averaging (SWA), with an important limit: its closest under-cap CIFAR-10 result is PreResNet-164, not this WRN-16-4 with Cutout or the competition test distribution.

The SWA authors report CIFAR-10 accuracy of 95.28% ± 0.10 for PreResNet-164 with SGD and 95.56% ± 0.11 with SWA at one training budget. They report the model at 1.7M parameters, below the course cap. The WRN-28-10 result in the same paper exceeds the cap. These results justify one matched test of SWA on WRN-16-4. They do not predict the Kaggle score. [14] [Authors' CIFAR results and implementation](https://github.com/timgaripov/swa)
The proposed SWA adaptation keeps WRN-16-4, Cutout, the fixed 45,000/5,000 split, and a 200-epoch total budget. Match the incumbent schedule through epoch 149. Use a constant learning rate of 0.01 for epochs 150 through 199 and average model weights after each averaging-phase epoch. Recompute BatchNorm statistics on permitted training images. This changes the late schedule as part of the SWA method, so describe the comparison as the incumbent recipe versus the SWA recipe, not as weight averaging alone. [14]
Run three fresh matched seeds per arm if the remaining deadline and GPU allowance permit. Match initialization and data-order seeds within each pair. Earlier seed-zero results followed selection among four recipes on this split, so report them as context only. Use the reloaded checkpoint's `manifest.json` `final_val_correct`, `final_val_accuracy`, and `val_pred.npy` for both arms. The SWA arm's final manifest metric is separate from ordinary epoch-level validation of the base model. Never compare the final cosine checkpoint to a last-epoch row from the SWA training model.

The pre-registered adoption rule is deliberately conservative: every one of the three paired SWA-minus-cosine validation differences must be positive, and the paired mean must be at least 25 of 5,000 images (0.5 percentage points). This is an operational decision rule, not a significance claim or a course requirement. Report all three deltas and an uncertainty interval. If SWA does not meet the rule, keep the cosine-Cutout recipe. Do not add seeds or change the rule after observing outcomes. The full-data Cutout model remains the fallback.

The 0.5-point adoption threshold used for the earlier single-seed experiments was a project heuristic, not a course rule or a validated statistical margin. The paired intervals show that both WRN-22-4 and RandAugment comparisons remain compatible with no clean-accuracy difference on this one split. Do not use synthetic corruption performance to rescue a candidate after seeing these results. [2]

## Why synthetic corruptions are not the next selection gate

CIFAR-10-C transformations applied to held-out training images would measure performance on those particular synthetic corruptions. They would not identify or reproduce the hidden evaluation distribution. A 204-model ImageNet study found that aggregate synthetic corruption robustness was weakly predictive of ImageNetV2 performance. Some individual corruptions correlated with specific natural shifts, but those correlations were not universal. Treat a corruption suite as a separate diagnostic, not a substitute for validation or a reason to adopt an otherwise losing recipe. [8] [10]

Do not infer that the hidden test set is CIFAR-10.1 from the 10.06-point validation-to-public gap. The CIFAR-10.1 study found 4-to-10-point drops on a separately curated test set while model rankings mostly persisted. That supports continued work on clean accuracy. It does not identify this competition's hidden images or predict its score. [1]

## Sources

1. Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. [Do CIFAR-10 classifiers generalize to CIFAR-10?](https://arxiv.org/abs/1806.00451), arXiv first posted 2018.
2. Thomas G. Dietterich. [Approximate statistical tests for comparing supervised classification learning algorithms](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf), 1998.
3. Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719), 2020. See the [CIFAR training details](https://openaccess.thecvf.com/content_CVPRW_2020/supplemental/Cubuk_Randaugment_Practical_Automated_CVPRW_2020_supplemental.pdf).
4. Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. [AugMix: A simple data processing method to improve robustness and uncertainty](https://arxiv.org/abs/1912.02781), 2020.
5. Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. [mixup: Beyond empirical risk minimization](https://arxiv.org/abs/1710.09412), 2018.
6. Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. [CutMix: Regularization strategy to train strong classifiers with localizable features](https://arxiv.org/abs/1905.04899), 2019.
7. Dan Hendrycks and Thomas Dietterich. [Benchmarking neural network robustness to common corruptions and perturbations](https://arxiv.org/abs/1903.12261), 2019.
8. Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. [Measuring robustness to natural distribution shifts in image classification](https://proceedings.neurips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html), NeurIPS 2020.

9. Dan Hendrycks et al. [The many faces of robustness: A critical analysis of out-of-distribution generalization](https://arxiv.org/abs/2006.16241), 2021.
10. Alfred Laugros, Alice Caplier, and Matthieu Ospici. [Using synthetic corruptions to measure robustness to natural distribution shifts](https://arxiv.org/abs/2107.12052), 2021.
11. Sergey Zagoruyko and Nikos Komodakis. [Wide residual networks](https://arxiv.org/abs/1605.07146), arXiv first posted 2016.
12. Terrance DeVries and Graham W. Taylor. [Improved regularization of convolutional neural networks with Cutout](https://arxiv.org/abs/1708.04552), 2017.
13. PyTorch Vision. [Torchvision 0.29.2 RandAugment implementation](https://github.com/pytorch/vision/blob/v0.29.2/torchvision/transforms/autoaugment.py).
14. Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407), 2018.
