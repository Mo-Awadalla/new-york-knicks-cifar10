# Primary-source compact recipes: a concrete alternative, not a 90% promise

Research date: November 5, 2026. Research only: no training, submissions, model changes, external image/weight downloads, or access to anyone's prediction/label files. Sources below are papers and author-maintained code; published results have **not** been reproduced locally in this investigation.

## Verdict

**The best-supported conventional residual-network alternative found is Shake-Shake-26-2x32d with the authors' AutoAugment/Cutout recipe, trained for the full 1,800 epochs.** AutoAugment reports **2.5% ± 0.1% CIFAR-10 error, i.e. 97.5% accuracy**, averaged over five runs, for the approximately 2.9M-parameter model. Its authors release the architecture and training code and explicitly say that the code replicates this result. This is a substantially firmer lead than inventing a compact PyramidNet+ShakeDrop+KD mixture or another WRN width change. A bounded modern-family check also identifies **DrNAS at 2.46 ± 0.03% error in 600 epochs**, with a source-derived final count under the cap; its tiny cross-paper numerical advantage does not establish superiority. See §6. [A1, A2, S1, M2]

**It is not evidence of a gain over our current student, and not sufficient evidence to commit automatically to a new implementation immediately before the deadline.** Our supplied current evidence is 97.44% on a fixed 5,000-image holdout after training on 45,000 permitted images; the paper's 97.5% is on the standard 10,000-image CIFAR test set after training on all 50,000. These are different experiments. The nominal 0.06-point difference is not a measured improvement. No source found reports this compact AutoAugment model on our split, our competition, or a verified matching evaluation distribution.

**Recommendation:** retain the 88.40% submitted incumbent. If the final decision is to authorize exactly one genuinely different architecture, use the faithful **released AutoAugment Shake-Shake-32 recipe below**, not a shortened 200-epoch version or an untested combination with KD/SAM/CutMix. Authorizing it remains a controlled diversification decision, not an evidence-backed expectation of crossing 90%. The implementation, calibration, fixed-split comparison, and full-data retrain would all still have to fit before the actual submission deadline. This investigation neither authorizes nor launches any of them.

## 1. What is being compared

Current project evidence comes from the supplied shared brief, not the historical state in [research-compact-models.md](research-compact-models.md):

- Incumbent: WRN-22-4, **4,298,970 final parameters**, CutMix plus KD from scratch-trained permitted-data teachers; **4,872/5,000 = 97.44%** on the fixed 45K/5K experiment; its full-data retrain scores **88.40% public**, with no held-out score for that full-data model.
- A 400-epoch student fell to 4,864/5,000; WRN-16-5 at a similar parameter count reached 4,858/5,000. Neither supports another depth/width adjustment as the next research recommendation.
- The earlier N=2, M=9 RandAugment trial and AugMix result are not experiments on the faithful Shake-Shake/AutoAugment recipe. They do not establish its outcome either way.
- Only staff-provided training images may be used. Models start from scratch. The final inferred model must be at most 5,000,000 parameters including its classifier. The parent owns the live rules interpretation and deadline check.

### Compact comparison table

Ranks are evidence/implementation priorities **if a new recipe is authorized**, not predicted Kaggle ordering. The first two are alternative augmentation recipes for the same architecture, not two independent architectural discoveries.

| Rank / disposition | Concrete recipe and official implementation | Final size | Published clean CIFAR-10 result | Training and evaluation budget | Verdict against current evidence |
| --- | --- | --- | --- | --- | --- |
| **1 — only new-architecture lead recommended conditionally** | Shake-Shake-26-2x32d + AutoAugment + Cutout; [author TF implementation](https://github.com/tensorflow/models/tree/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment) | **2.9M reported**; **2,923,162 trainable scalars derived statically for the released TF variant**, including the 10-way head; not runtime-counted here | **2.5 ± 0.1% error / 97.5% accuracy**, five runs. Same paper's model: baseline 3.6%, Cutout 3.0%. [A1] | **1,800 epochs**, batch 128; Nesterov SGD, LR 0.01 cosine, weight decay 0.001, clip norm 5. One deterministic model evaluation; no SWA/EMA/TTA or inference ensemble. [A2–A5] | A matched published **0.5-point** gain over its own Cutout baseline is real evidence for the complete recipe; it is **not** a 0.5-point gain over our KD student. Large implementation/schedule change. |
| **2 — corroborating alternative, not another experiment to add** | Same compact Shake-Shake + **Population Based Augmentation (PBA)** + Cutout; [author result script](https://github.com/arcelien/pba/blob/master/scripts/table_1_cifar10.sh), model option `ss_32` | **2.9M architecture family reported**; exact PBA implementation count not derived/runtime-checked here | **2.54 ± 0.10% error / 97.46% accuracy**, mean final error over five random initializations. The paper's AA comparison is cited from AutoAugment, **not** an independent AA re-run. [P1] | **1,800 epochs**, batch 128; cosine cycle, clipping 5; script LR 0.01 / decay 0.001. Use released 200-epoch policy schedule stretched linearly to 1,800. No test-time augmentation or averaging is proposed. [P1, P2] | Confirms that this small architecture plus a fully specified strong-augmentation schedule reaches the same clean benchmark neighborhood. Adds schedule machinery without evidence of superiority to rank 1 or to our incumbent. |
| **3 — genuine modern NAS alternative; no evidence to displace rank 1 now** | **DrNAS CIFAR-10 genotype**, 20-cell convolutional DAG with skip connections; [author evaluation code](https://github.com/xiangning-chen/DrNAS/tree/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/DARTS-space) | **4.1M reported excluding auxiliary tower**; statically derived **4,099,294 main-network parameters / 4,575,720 including the training auxiliary tower**, not runtime-counted | **2.46 ± 0.03% error / 97.54% accuracy**, from-scratch 50K training / standard CIFAR test. [M2] | **600 epochs**, batch 64; SGD momentum 0.9, LR 0.025 cosine, decay 0.0003; Cutout 16, drop-path ramp toward 0.3, auxiliary CE weight 0.4, clipping 5. No averaging/TTA in reference evaluation. [M2, M3] | Real eligible-size published recipe, not an ImageNet weight transfer. Only **0.04 points** numerically above AA across unmatched papers; no paired or matching-shift result. Different cell/auxiliary machinery and staff interpretation of residual-DAG eligibility remain integration/rule risks. |

**No entry demonstrates a gain under a comparable project split.** None establishes a 90% competition result. Larger published models' results must not be used as if they were results of these compact variants.

## 2. Exact reference configuration for rank 1

The specification here deliberately follows **the released AutoAugment authors' implementation**, rather than silently mixing its hyperparameters with the original Shake-Shake paper or a convenient PyTorch port. The official folder snapshot is [A2]; the model itself has not changed since the linked [2018 model commit](https://github.com/tensorflow/models/commit/5a66a397d314d0dcd5fdde162b5d28d5c239dd54).

### Architecture and parameter accounting

From [`shake_shake.py`](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/shake_shake.py) and [`custom_ops.py`](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/custom_ops.py):

- `model_name=shake_shake_32`, depth 26, `shake_shake_widen_factor=2`.
- Bias-free 3×3 stem, 3→16 channels, **with affine BN** in this TF variant.
- Three stages of **four residual blocks each**; branch widths **32, 64, 128**; downsample by 2 in the first blocks of stages 2 and 3.
- **Two learned residual branches per block**, each `ReLU → 3×3 conv → BN → ReLU → 3×3 conv → BN`. A single skip path is added; this is one residual network, not an inference ensemble.
- On channel changes, use the released two-path shortcut: 1×1 average-pool sampling plus 1×1 convolution in each path, shift the second path by one pixel, concatenate, then BN. Do not substitute an ordinary WRN projection.
- Training uses image-level random forward branch coefficients and independent backward coefficients. **The released TF code normalizes two independent uniform samples to sum to one** for each pass; that is not the same sampling distribution as drawing one uniform alpha and setting the other coefficient to `1-alpha`. At evaluation both branches have coefficient **0.5**. [A3, S1]
- Final ReLU, global average pooling, **128→10 affine classifier**. No dropout, learned auxiliary head, or extra inference model.

Static scalar accounting for exactly those shapes, counting affine BN scale/bias but excluding nontrainable running statistics:

| Component | Derived trainable scalars |
| --- | ---: |
| Stem convolution and BN | 464 |
| 32-channel stage, including shortcut | 139,840 |
| 64-channel stage, including shortcut | 557,184 |
| 128-channel stage, including shortcut | 2,224,384 |
| Classifier including bias | 1,290 |
| **Total** | **2,923,162** |

For a stage with input width `I`, branch width `C`, and four blocks, the accounting is `2*(9*I*C + 9*C*C) + 3*(4*9*C*C)` branch-convolution weights, `32*C` branch-BN parameters, and `I*C + 2*C` shortcut parameters. This is a **source-derived exact shape sum, not an instantiated-model assertion**; any implementation change still requires its own full parameter count before use. In particular, the Fast AutoAugment PyTorch port has a biased stem and no stem BN, so its exact count is not identical. [A3, A6, F3]

### Optimizer, schedule, objective, and checkpoint

The released `train_cifar.py` / `helper_utils.py` specify:

| Setting | Reference value |
| --- | --- |
| Data | Standard CIFAR-10 **50,000 training / 10,000 test** benchmark; not CIFAR-100, SVHN extra images, ImageNet, or external pretraining |
| Epochs / batch | **1,800 / 128**; `floor(train_size/128)` updates per epoch |
| Optimizer | SGD with **momentum 0.9, Nesterov enabled** |
| Initial LR | **0.01**, not original Shake-Shake's 0.2 |
| LR schedule | One **per-update cosine** decay toward zero over all 1,800 epochs; no warmup or restarts in this TF release |
| Objective | Ordinary mean softmax cross-entropy; no label smoothing, KD, mixup, CutMix, SAM, or auxiliary loss |
| Regularization | Add **0.001 × sum(tf.nn.l2_loss(v)) over all trainable variables**, including BN affine parameters and classifier bias; equivalently 0.0005 × sum of squared scalars in the objective. Do not accidentally halve/double the gradient coefficient or exempt BN while claiming fidelity |
| Clipping | Global gradient-norm clipping at **5.0**, after gradients of the full regularized objective |
| BN | Decay **0.9**, epsilon **1e-5**, trainable scale and offset; ordinary running-statistic evaluation |
| Initialization | Convolutions: normal std `sqrt(2/(kernel_height*kernel_width*out_channels))`; FC: uniform ±`1/sqrt(num_output_classes)` and zero bias |
| Endpoint | Final trained checkpoint, deterministic unaugmented inference; no EMA/SWA, checkpoint ensemble, Monte Carlo shaking, or crop/flip voting |

Sources: [A3–A6]. The model code draws coefficients in float32; adopting modern AMP is an engineering adaptation, not something demonstrated by these original results.

### Augmentation: preserve the actual release, not just the name

The paper describes **25 sub-policies**, crop/flip then AutoAugment then fixed **16×16 Cutout**. However, the released code is more specific and differs in observable details:

1. [`policies.py`](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/policies.py) returns **95 two-operation sub-policies**: 4 + 7 + 8 groups of five, not the paper's 25. One is sampled independently for each image. Probabilities, magnitudes, operation order, random signs, PIL behavior, and fill handling are part of the recipe.
2. [`data_utils.py`](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/data_utils.py) actually applies **AutoAugment first, then four-pixel zero padding/random crop, then horizontal flip, then fixed 16-pixel Cutout**. Padding/Cutout zeros are in normalized space. The crop helper samples top/left with `randint(0,8)`, i.e. offsets 0 through 7, not all nine possible offsets. [A7, A8]
3. Released normalization is mean `[0.49139968, 0.48215841, 0.44653091]`, std `[0.24703223, 0.24348513, 0.26158784]` on `[0,1]` images. Evaluation only normalizes; there is no random crop, flip, or AutoAugment. [A7, A8]
4. A policy's own Cutout may occur in addition to the final fixed Cutout. A stock `torchvision.AutoAugment(CIFAR10)` name is **not evidence of byte-for-byte equivalence** to the release's 95-policy pool, magnitudes, fill, and ordering.

These are not reasons to dismiss AutoAugment; they are reasons **not to promise paper reproduction after a casual port**. A future implementation must choose and record whether its reference is the paper policy or the released author code. The concrete recommendation here is the **released-code specification above**, with necessary course-data/split changes explicitly identified.

### Necessary project adaptations — not demonstrated results

A legally usable project version cannot run the public research loader unchanged:

- Replace automatic public CIFAR downloads and test-batch evaluation with the **staff-provided manifests/images**. Do not download the standard test set to validate this project or use public author checkpoints. Public code is reference material only.
- For selection, use the existing fixed **45K/5K membership** and normalization statistics computed only on the 45K training rows. That changes the paper's 50K full-training setting and its fixed normalization constants. Keep validation unaugmented and do not expose labels/images to augmentation-policy search.
- Train the 45K comparison for the full **1,800 epochs**, then—only if selected under a declared rule—retrain from scratch on all permitted 50K using the frozen selected recipe. A final full-data model has no held-out accuracy.
- Preserve one final inferred network and deterministic single-view inference. Do not add TTA or an ensemble based on the fact that the architecture itself has two residual branches. The former are separate rule questions; the latter are internal layers of one model.

Any altered Shake sampling distribution, stem, shortcuts, 200/400-epoch horizon, norm, policy pool, mixed augmentation, KD/SAM objective, batch size, LR, or averaging method is a **proposed adaptation**, not the reported 97.5% experiment.

## 3. Corroboration and why compact ResNeXt/PyramidNet do not displace it

### Two other primary research programs reach the same small-model neighborhood

**PBA:** the authors report **2.54 ± 0.10%** for Shake-Shake-32 over five initializations. Their reproduction script uses LR 0.01 / weight decay 0.001 and the released `schedules/rcifar10_16_wrn.txt`, scaled from a 200-epoch policy schedule to a 1,800-epoch training run. It is a real reproducible author recipe, but it does not beat the AutoAugment mean. Importantly, PBA explicitly says its AA/baseline numbers are **taken from AutoAugment**. Counting that AA column as a second independent AA reproduction would be wrong. [P1, P2]

**Fast AutoAugment:** its primary paper reports Shake-Shake-32 at **2.7% error with a transferred reduced-CIFAR policy** versus **2.5% with direct search on the full target dataset/model**. The released [compact YAML](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/confs/shake26_2x32d_cifar.yaml) selects `fa_reduced_cifar10`, 1,800 epochs, batch 128, LR 0.01, decay 0.001, SGD/Nesterov, no EMA, and a five-epoch warmup wrapper. Therefore that immediately reusable YAML corresponds to the **2.7% transfer route, not an automatic 2.5% result**. The paper's original AA column is a comparison source, not proof this port reproduced original AA. The PyTorch release is useful reference code, but substituting it silently changes stem/normalization/shake sampling and training details. No policy search is recommended before this deadline. [F1–F4]

### PyramidNet correction to the old research note

The author repository has an eligible, stronger-than-basic **bottleneck PyramidNet-164 alpha=84, 3.8M, 3.96% error**. The old note emphasized basic PyramidNet-110 alpha=84 at about 4.27%. This correction does **not** rescue the speculative compact ShakeDrop recommendation: the headline AutoAugment PyramidNet+ShakeDrop model is the **272-layer, alpha=200, approximately 26M-parameter** family, not an eligible compact variant. Shrinking its alpha/depth and retaining its 1.5% clean or 4.4% natural-shift error is unsupported. [Y1, Y2, A1]

### Compact ResNeXt: a useful architecture family, no verified superior under-cap recipe here

The original ResNeXt paper's tabulated CIFAR winners are **29-layer 8×64d: 34.4M, 3.65% error** and **16×64d: 68.1M, 3.58%**, each averaging ten runs. Both exceed the cap. The official repository also warns that its later original-block/batch-128 setting differs from the arXiv preactivation/batch-256 setup, and that changing GPU count requires care with per-GPU BN batch size. [R1, R2]

A narrower grouped model may fit, but the sources examined do not provide a **specific under-5M configuration with a stronger published result and equally complete author recipe** that displaces Shake-Shake-32. Do not relabel an over-cap 4×64d/8×64d natural-shift model as a compact result, or invent `4×32d` and inherit another model's accuracy. This is a limitation of the evidence reviewed, not a claim that no compact ResNeXt can work.

### Public NYU course report: corroboration, not a competition solution

Thakur, Chauhan and Gupta's **Efficient ResNets: Residual Network Design** is explicitly a NYU Tandon course project. Its final table reports **4,697,742 trainable parameters and 96.04% standard CIFAR-10 test accuracy**, with residual blocks `[4,4,3]`, channels `[64,128,256]`, SE, SGD/Lookahead, cosine scheduling, and 200-epoch training discussed in the text. These are author-reported results, not independently reproduced here. Crucially, §3.1 says **“we’ll simply use the test set as our validation set.”** Thus its architecture/strategy selection is not a fresh held-out evaluation comparable to our fixed 45K/5K protocol. Neither the 96.04% claim nor its much weaker unaugmented ResNet18 comparison establishes a stronger candidate than our 97.44% validation result or a 90% result in this competition. Only the paper was inspected; no prediction/label files or checkpoints were accessed. [C1]

## 4. Natural-shift evidence — benchmarks only

No evaluation-set identity, labels, or source are inferred from the course image count or our scores. These are **published benchmark observations**, not a claim that the competition uses those data.

- Recht et al.'s primary 2018 study, **Table 1**, reports **Shake-Shake-26-32d: 96.6% original accuracy → 89.8% on its new natural-image test set**, with new-set interval **[88.4%, 91.1%]**. Appendix A identifies depth 26, base width 32, Shake-Shake-Image. **This is the eligible compact baseline without AutoAugment; it is not a measured AA result.** [N1]
- The same study's compact **basic PyramidNet-110 alpha=84** gives **95.7% → 89.3%**, interval **[87.8%, 90.6%]**. This is not bottleneck PyramidNet-164, and not ShakeDrop. [N1]
- Its stronger **Shake-Shake-64d+Cutout** gives **97.1% → 93.0%**, but the wider architecture is over the final-model cap: applying the same source-shape accounting to widths 64/128/256 gives about **11.65M** for the TF variant, not a runtime count of the study's PyTorch port. This result is not evidence for a 32d model. [N1, A3, A6]
- AutoAugment's own paper evaluates its **large PyramidNet+ShakeDrop** on the newly proposed test set and reports **4.4% error / 95.6% accuracy** versus 1.5% clean error. The approximately **26M** final model is ineligible. This is evidence that that large recipe improved that benchmark, **not** that compact AutoAugment achieves it, nor permission to use that external dataset. [A1, Y1]

**What survives these caveats:** compact Shake-Shake has directly measured natural-shift performance, and larger-model augmentation advances sometimes transfer beyond the original test set. **What does not survive:** an estimate of our hidden-test accuracy, a proven 1.6-point gain, or a reason to transfer a big-model natural-shift headline to a 2.9M model. The exact eligible Shake-Shake-32+AA natural-shift result was not found in these primary sources.

## 5. Implementation, rule, and deadline decision

### Concrete risks

1. **New implementation, not a config tweak.** The original Shake-Shake repository is Lua/Torch. AutoAugment's official code requires **TensorFlow 1.x and Python 2**, explicitly not TF2. Porting requires careful custom backward scaling, shortcut sampling, initialization, all-variable L2 and clipping, policy/fill semantics, and BN equivalence. New GPU hardware alone does not remove these correctness risks. [S2, A2–A8]
2. **Full horizon is part of the evidence.** 1,800 epochs is **nine times the epoch count** of our 200-epoch runs. At the reference full-data batch size there are `390 × 1800 = 702,000` optimizer updates; the fixed-split version has `351 × 1800 = 631,800`. These are schedule arithmetic, **not measured runtime estimates**. Two residual branches and different depth/width mean existing WRN/teacher timings cannot simply be scaled into an A100 ETA. The original Shake-Shake paper explicitly attributes the long schedule to its stochasticity. [S1, A4, A5]
3. **Fixed-split selection plus final retraining takes two runs.** The published mean is not a basis for skipping the held-out comparison, or for submitting an unfinished-horizon checkpoint. With the supplied roughly one-day deadline context, a fresh port plus two full schedules has no measured completion evidence here. Compute availability is not proof of end-to-end readiness.
4. **Course-data boundary.** Author loaders download public data and evaluate the public CIFAR test set. Those paths cannot be used here. Import no author weights, extra training images, policy-search examples, or predictions. A published CIFAR augmentation-policy table is an algorithm/configuration, not pretrained network weights, but staff retains interpretation authority if learned policy reuse is questioned.
5. **No unsolicited recipe combination.** Neither the small AA result nor the PBA result demonstrates AA+CutMix+SAM+KD+SWA. Adding all available mechanisms is a new hypothesis, not faithful reproduction.

### What decision is justified now?

- **Do not launch another WRN width tweak or compact PyramidNet/ResNeXt mixture based on headline numbers.** The relevant direct project evidence is already negative for the width tweak, and the claimed stronger compact mixture is not established.
- **If one new architectural implementation is nevertheless authorized, choose only rank 1 and its complete released specification.** It is the cleanest concrete experimental alternative found, not a guaranteed replacement. PBA and Fast AutoAugment are corroboration, not three additional late experiments.
- **Absent a ready implementation and measured full-run completion margin, the evidence is insufficient to recommend a new training implementation before the deadline.** Keeping the current real 88.40% score is preferable to presenting a speculative adaptation as a route known to reach 90%. The main agent's separate lower-integration-risk analysis may offer a better final action.

### The 0.5-point adoption gate is not a course rule

The supplied current policy requires **97.94% = 4,897/5,000** to clear the current 97.44% by 25 images. That is not a requirement in these papers or the course constraints, nor a significance test. It is also above the compact AA paper's **97.5% mean on a different split**. Thus it would be inconsistent to sell the literature as a strong reason to launch this model while ignoring that the project's current gate would likely reject a merely paper-level outcome. “Likely” here is an inference about the gate, **not a predicted fixed-split score**.

The gate should be explicitly reconsidered by the decision owner if architectural diversification is desired; it must not be quietly waived after seeing a result. Nothing in this note changes it. A fair future comparison would predeclare the full schedule and endpoint, retain the fixed split, and examine paired changed predictions and run variability rather than demanding an arbitrary gain or using public-score probing to choose hyperparameters.

## 6. Bounded modern-family check: ConvMixer and NAS

### ConvMixer: a fitting 512/16 variant is not a published accuracy result

Trockman and Kolter's primary **Patches Are All You Need?**, Appendix B/Table 4, reports its strongest listed CIFAR-10 setting as **ConvMixer-256/16, patch size 1, depthwise kernel 8, weight decay 0: 96.74% accuracy**, with **1,339K reported parameters**. The paper gives the exact parameter formula, including classifier, convolution biases and affine BN:

`h * [d * (k² + h + 6) + 3*p² + n_classes + 3] + n_classes`.

For ten classes this gives **1,339,402** for the published 256/16/8/1 model; proposed **512/16/8/1** gives **4,775,946**, and **512/16/9/1** gives **4,915,210**. These are formula-derived counts, not instantiated checks. Thus some 512/16 CIFAR variants genuinely fit; the ImageNet table's 5.4M count, with its different patch embedding and 1,000-way head, should not be carried over unchanged. Conversely, **no CIFAR-10 accuracy for either of those exact 512/16 settings was identified in the primary paper or author repository**. Fitting the cap is not evidence that widening exceeds Shake-Shake-AA. [M1]

The paper's CIFAR experiments run **200 epochs** with RandAugment, mixup, CutMix, random erasing, gradient clipping, and timm augmentation; its general setup uses AdamW and a triangular LR schedule. The paper is not an ImageNet-pretraining transfer experiment for these CIFAR results. However, the official README supplies its exact command for an **ImageNet** model, not a complete exact command tied to the 96.74% CIFAR row. The separate author `convmixer-cifar10` script explicitly says **“not intended to perfectly replicate the results in our paper”**, changes augmentation implementations, and omits mixup/CutMix. Its concrete examples reach approximately 92.5%/25 epochs and above 94%/100 epochs. This does **not** provide a faithful, stronger under-5M replacement recipe. ConvMixer also has a residual connection around the depthwise subblock, rather than the project's conventional two-convolution residual unit; staff decides eligibility. [M1]

### DrNAS: a real under-cap, from-scratch 600-epoch alternative, but not a demonstrated substantial gain

The **DrNAS** primary paper explicitly states: **“The evaluation phase uses the entire 50K training set to train the network from scratch for 600 epochs.”** Its CIFAR-10 table reports **2.46 ± 0.03% error and 4.1M parameters**, not an ImageNet-pretrained transfer. The released final `DrNAS_cifar10` genotype avoids needing to rerun NAS. The numerical clean mean is 97.54%, only **0.04 percentage points above** compact AA's 97.5% across different papers/protocols; this is not a paired comparison or evidence of a competition gain. [M2]

The complete public final-training recipe is `DARTS-space/train.py --cutout --auxiliary` with its released defaults: **20 cells, 36 initial channels**, 3×3/108-channel stem, normal/reduction genotype in `genotypes.py`, reduction at cells 6 and 13, concatenated cell outputs, global pooling and 576→10 classifier; **600 epochs, batch 64, SGD LR 0.025, momentum 0.9, Nesterov off, weight decay 0.0003, cosine to zero, ordinary CE plus 0.4-weight auxiliary CE, norm clipping 5**, crop 32 with padding 4, horizontal flip, normalization, **Cutout 16**, drop-path linearly ramped toward 0.3. Reference normalization is mean `[0.49139968,0.48215827,0.44653124]`, std `[0.24703233,0.24348505,0.26158768]`. Evaluation is deterministic single-view with drop-path disabled and no EMA/SWA/TTA. The script logs best standard-test accuracy while writing the current weights each epoch; a project adaptation must predeclare its selection endpoint and must not use the competition test for checkpoint selection. [M3]

**Final parameter cap audit from source shapes:** the author's `utils.count_parameters_in_MB` explicitly excludes names containing `auxiliary`. Static accounting gives stem **3,132**, cells **4,090,392**, main classifier **5,770**, totaling **4,099,294**. The auxiliary tower has **476,426** scalars: `576*128 + 2*128 + 128*768*4 + 2*768 + 768*10 + 10`. Even retaining it totals **4,575,720**, below the cap. Removing the training-only tower from the final inference artifact gives the **4,099,294** final model. Counts are source-derived, not runtime-verified. The cell tally uses 18 normal cells and two reduction cells; normal operations contribute `13*C²+191*C`, reduction operations `10*C²+270*C`, plus each cell's two preprocessors `(C_prev_prev+C_prev)*C+4*C`. [M3]

**Decision:** this materially improves the shortlist's coverage and is a more concrete alternative than compact PyramidNet speculation. It does **not** change the recommendation to avoid an unverified new implementation immediately before the deadline. If a later decision prioritizes a shorter published horizon and accepts a skip-connected convolutional DAG as a residual CNN, DrNAS is the specific modern candidate to assess—not an arbitrary widened ConvMixer. New loader/split integration, operation/drop-path/auxiliary-loss fidelity, final parameter validation, GPU calibration, and a held-out comparison remain prerequisites. Its 600 epochs are not an A100 wall-clock measurement. No matching natural-shift result was identified in this bounded check.

### Other NAS headlines: distinguish means, best runs, and over-cap models

- The **ProxylessNAS authors' own project page** says **“2.08% test error with only 5.7M parameters”**: over the cap. It cannot justify an eligible candidate. [M4]
- **DARTS+ Table 1** reports **3.7M: best 2.32%, mean 2.50 ± 0.11%**, and **4.3M: best 2.20%, mean 2.37 ± 0.13%**, with **2,000-epoch** full training. Its **1.68%** result is the larger **7.2M** variant, over cap. The reported 4.3M mean is nominally stronger than AA but only by 0.13 points across unmatched studies, while requiring a longer schedule; it is not appropriate to compare its best run against AA's five-run mean. Exact final code-level counts/auxiliary accounting and a complete public reproduction command were not established in this bounded check, so it is not promoted above the fully traced recipes. [M5]

## Primary sources and decisive quotations

- **[A1] Cubuk et al., AutoAugment (CVPR 2019), §4.1 and Table 2.** [Paper](https://arxiv.org/html/1805.09501#S4.SS1); [Table 2](https://arxiv.org/html/1805.09501#S4.T2). Compact row: “Shake-Shake (26 2x32d) … 2.5 ± 0.1.” Text: **“All of the reported results are averaged over 5 runs.”** The 4.4% natural-shift result is explicitly for **“our best model, PyramidNet+ShakeDrop trained with AutoAugment.”**
- **[A2] AutoAugment authors, official README.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/README.md). **“The code replicates the results … Shake-Shake (26 2x32d)”**; settings table **0.01 / 1e-3 / 1800 / 128**. **“Be sure to run the code using python2 and not python3.”**
- **[A3] Official AutoAugment `shake_shake.py`.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/shake_shake.py). Defines all stages, independent normalized forward/backward samples, custom skip, and evaluation `x *= 1.0 / 2`.
- **[A4] Official AutoAugment `train_cifar.py`.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/train_cifar.py). Defines Shake-Shake-32 hyperparameters, BN settings, Nesterov 0.9, clipping 5, final evaluation, and default 50,000/0 split plus standard-test evaluation.
- **[A5] Official AutoAugment `helper_utils.py`.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/helper_utils.py). Defines per-batch cosine, ordinary CE, and L2 over **all** trainable variables.
- **[A6] Official AutoAugment `custom_ops.py`.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/custom_ops.py). Defines trainable convolution/FC/BN shapes and initialization used in the static parameter sum.
- **[A7] Official AutoAugment data and policies.** [Data pipeline](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/data_utils.py); [95-policy release](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/policies.py). These expose the paper/code policy-count and operation-order differences.
- **[A8] Official AutoAugment transforms.** [Pinned source](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/augmentation_transforms.py). Contains exact normalization, PIL transformations, crop offsets, fill handling, and fixed Cutout.
- **[S1] Gastaldi, Shake-Shake regularization (2017), §2.1 and §2.3.** [Paper](https://arxiv.org/html/1705.07485). **“Due to the introduced stochasticity, all models were trained for 1800 epochs.”** Reports the small 32d architecture at **2.9M**, original S-S-I error 3.55%, S-E-I error 3.47%; original LR **0.2** and two GPUs. This is **not** the AutoAugment LR 0.01 recipe.
- **[S2] Gastaldi, original author implementation.** [Repository](https://github.com/xgastaldi/shake-shake). Torch/Lua code; linked by the original paper.
- **[P1] Ho et al., Population Based Augmentation (ICML 2019), Table 2 and §4.1.** [Paper](https://arxiv.org/html/1905.05393#S3.T2). **“We report the mean final test error of 5 random model initializations.”** Compact row **2.54 ± 0.10**. **“Additionally, we report Baseline, Cutout, and AutoAugment (AA) results found in (Cubuk et al., 2018).”** [Publication record](https://proceedings.mlr.press/v97/ho19b.html).
- **[P2] PBA authors, CIFAR-10 reproduction script.** [Source](https://github.com/arcelien/pba/blob/master/scripts/table_1_cifar10.sh). `ss_32` selects `shake_shake_32`, LR 0.01, WD 0.001, and `rcifar10_16_wrn.txt` with policy horizon 200.
- **[F1] Lim et al., Fast AutoAugment (NeurIPS 2019), Table 2 and §4.1.** [Paper](https://arxiv.org/html/1905.00397#S4.T2). Compact row: **“2.7 / 2.5”**, explicitly **transfer / direct**. [Official repository](https://github.com/kakaobrain/fast-autoaugment).
- **[F2] Fast AutoAugment authors, compact configuration.** [Pinned YAML](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/confs/shake26_2x32d_cifar.yaml). Selects transferred policy, full horizon, optimizer, warmup, and EMA=0.
- **[F3] Fast AutoAugment authors, PyTorch architecture.** [Model](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/FastAutoAugment/networks/shakeshake/shake_resnet.py); [shake/shortcut](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/FastAutoAugment/networks/shakeshake/shakeshake.py). Source evidence of differences from the TF release, not an executed port.
- **[F4] Fast AutoAugment authors, training and data code.** [Training](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/FastAutoAugment/train.py); [data](https://github.com/kakaobrain/fast-autoaugment/blob/2424224291616a2158a2265069ee95a78a2d13d9/FastAutoAugment/data.py). Distinguishes augmentation choices and normalization, no-EMA configuration, automatic downloads, and fixed last-epoch reporting option.
- **[Y1] Han, Kim and Kim, Deep Pyramidal Residual Networks (CVPR 2017), Table 4 and §4.1.** [Paper](https://arxiv.org/html/1610.02915). Exact schedule/optimizer; basic alpha=84 **3.8M / 4.26 ± 0.23**; 272/200 **26.0M**. The paper states **“we did not involve”** dropout or stochastic depth in its CIFAR comparison.
- **[Y2] PyramidNet authors, updated architecture results.** [Torch repository](https://github.com/jhkim89/PyramidNet); [author PyTorch repository](https://github.com/dyhan0920/PyramidNet-PyTorch). Updated bottleneck row **164 / alpha=84 / 3.8M / 3.96%**; distinct from the original paper's basic model.
- **[R1] Xie et al., Aggregated Residual Transformations (CVPR 2017), Table 7.** [Paper](https://arxiv.org/html/1611.05431). **“Our results are the average of 10 runs.”** Tabulated CIFAR models **34.4M and 68.1M**.
- **[R2] ResNeXt authors, reference implementation.** [Repository](https://github.com/facebookresearch/ResNeXt). **“In the arXiv paper CIFAR results are based on pre-activated bottleneck blocks and a batch size of 256.”** Later code recommends original blocks/batch 128; documents per-GPU batch/LR changes.
- **[N1] Recht et al., Do CIFAR-10 Classifiers Generalize to CIFAR-10? (2018), Table 1 and Appendix A.** [Benchmark table](https://arxiv.org/html/1806.00451#S4.T1); [model descriptions](https://arxiv.org/html/1806.00451#A1). Compact Shake-Shake-32 **96.6→89.8** and basic PyramidNet-110-84 **95.7→89.3**. These are published benchmark observations only; no associated images, labels, or model outputs were accessed here.
- **[C1] Thakur, Chauhan and Gupta, Efficient ResNets: Residual Network Design (2023), §3.1 and final Table 1.** [Paper](https://arxiv.org/html/2306.12100). **4,697,742 parameters / 96.04% standard CIFAR-10 test accuracy**; explicitly **“Use test set for validation”**. Public course-report corroboration only, not an independently reproduced current-competition result.
- **[M1] Trockman and Kolter, Patches Are All You Need? (2022), Appendix B/Table 4 and Eq. 4.** [Primary paper](https://arxiv.org/html/2201.09792#A2), [official model](https://github.com/locuslab/convmixer/blob/main/convmixer.py), [author repository](https://github.com/locuslab/convmixer), [simplified CIFAR script caveat](https://github.com/locuslab/convmixer-cifar10). Primary best listed CIFAR result **96.74%**, exact formula supplies the static compact counts; no 512/16 CIFAR result claimed.
- **[M2] Chen et al., DrNAS: Dirichlet Neural Architecture Search (ICLR 2021), §4.1/Table 2.** [Primary paper](https://arxiv.org/html/2006.10355#S4.SS1), [official repository](https://github.com/xiangning-chen/DrNAS). **“train the network from scratch for 600 epochs”**; **2.46 ± 0.03% / 4.1M**. Do not confuse its CIFAR model with its **5.7M ImageNet** model.
- **[M3] DrNAS official pinned final-training implementation.** [Train](https://github.com/xiangning-chen/DrNAS/blob/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/DARTS-space/train.py), [genotype](https://github.com/xiangning-chen/DrNAS/blob/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/DARTS-space/genotypes.py), [model](https://github.com/xiangning-chen/DrNAS/blob/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/DARTS-space/model.py), [operations](https://github.com/xiangning-chen/DrNAS/blob/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/DARTS-space/operations.py), [normalization/counting helper](https://github.com/xiangning-chen/DrNAS/blob/f2caa4dbdd96c1be29b56cb7ed47a74a184e6ef0/utils.py). Source basis for exact static counts and complete final-training defaults; no external code executed.
- **[M4] Cai, Zhu and Han, ProxylessNAS (ICLR 2019), author project page.** [Primary author page](https://hanlab.mit.edu/projects/proxylessnas). **“2.08% test error with only 5.7M parameters.”**
- **[M5] Liang et al., DARTS+: Improved Differentiable Architecture Search with Early Stopping (2020), Table 1 and Appendix B.** [Primary paper](https://arxiv.org/html/1909.06035#S3.T1). Separates **best/mean** results and **3.7M / 4.3M / 7.2M** models; specifies **2,000 epochs**, batch 96, Cutout, path dropout 0.2 and auxiliary weight 0.4.
