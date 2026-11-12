# Concrete endgame decision: reuse the model, change inference once

Research completed November 5, 2026. No training, model changes, TTA evaluation, GPU allocation, or Kaggle submission was performed during the research phase.

**Execution update:** the user subsequently authorized the one fixed validation check. AugMix32 reached 4,861/5,000 versus the original 4,872, with tuning net −6 and audit net −5. It failed the declared positive-gain rule on both partitions and was rejected without retuning. Parameters and buffers were unchanged, no full-data TTA predictions or official submission were made, and the 88.40% incumbent was retained. See [measured decision](../runs/augmix32-001/decision.json). The research recommendation below is retained as historical reasoning, not as an adopted improvement.

## Research recommendation before execution

**Stop new training campaigns. The most concrete higher-upside, low-integration-cost candidate found is a fixed 32-view AugMix probability average using the existing 88.40% student, with all parameters and BatchNorm statistics frozen.** This is the **non-adaptive `+TTA` baseline** in the MEMO paper, not MEMO's test-time learning algorithm. It is a researched candidate, not a method known to score 90% on this competition.

This recommendation prioritizes a documented natural-recollection gain and reuse of an already verified model over another architecture port. A two-view horizontal-flip average is the more conservative, cheaper alternative; [the inference evidence review](research-90-generalization.md) recommends that when minimizing risk is the primary objective. For the user's goal of a meaningful improvement without another training campaign, the integration decision here prioritizes the one fixed 32-view policy. **That preference is an inference, not evidence that it outperforms two-view averaging on our model.** Do not test both and then expand into an open-ended augmentation search.

The current 88.40% checkpoint and submission stay unchanged. Nothing in this memo authorizes a rule violation, changes the existing selection protocol automatically, or establishes a new score.

## Why this is a concrete lead

Zhang, Levine and Finn's NeurIPS 2022 paper reports the following **frozen test-time averaging** results for its ResNet-26, separately from adaptation:

| Benchmark in the paper | Single-view accuracy | Frozen AugMix TTA accuracy | Gain |
| --- | ---: | ---: | ---: |
| Standard CIFAR-10 | 90.8% | 92.7% | +1.9 percentage points |
| CIFAR-10.1 natural-recollection benchmark | 81.6% | 85.2% | +3.6 percentage points |

These accuracies are 100 minus the errors reported in [Table 1 of the primary paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf). The [official supplement, Appendix A](https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Supplemental-Conference.pdf) specifies 32 AugMix views for the CIFAR model and says the TTA baseline uses the same augmentations as MEMO.

**The limiting evidence matters:** that is a much weaker, GroupNorm ResNet-26, not our distilled BatchNorm WRN-22-4. The paper does not measure our model, our training recipe, or our competition. CIFAR-10.1 is a published benchmark, **not an identification of the hidden competition dataset**. The result justifies considering frozen averaging; adding 3.6 points to our 88.4% would be an unsupported forecast. Augmentations can also hurt.

For stronger CIFAR models, the [Greedy Policy Search paper, Table 3](https://arxiv.org/html/2002.09103#A2.T3) reports smaller crop/flip gains: its 1.7M PreResNet-110 goes from 97.17% to 97.54% with ten random crop/flip views. That is a different policy and training recipe, not the effect of the proposed 32-view policy or of a two-view flip on our student.

## Exact proposed inference contract

1. Load the existing student once. Keep `model.eval()` and inference/no-gradient mode. Do not alter any parameter or running-statistic buffer.
2. For each input, generate **32 independently sampled AugMix views** using a recorded, label-independent RNG scheme. Freeze the policy and seed procedure before any new validation measurement.
3. Follow the authors' CIFAR augmentation helper, not our old severity-3 training helper by name alone:
   - Random 32x32 crop after four-pixel padding and random horizontal flip.
   - Three chains, each with one to three operations, Dirichlet(1,1,1) chain weights, and Beta(1,1) blending with the preaugmented clean component.
   - The released helper uses nine operations at **level 1**, not our old training severity 3; its affine/rotation calls use the literal `fillcolor=128`.
   - There is **no separate 33rd unaugmented prediction** in the proposed average. Each mixed view already contains a clean component.
4. Normalize each view with **this checkpoint's saved training normalization**, not the reference ResNet's hard-coded mean/std of 0.5. This is a necessary model-specific adaptation.
5. Average the **32 softmax probability vectors** in FP32, then take argmax. Do not average independently trained models or change the view weights, temperature, or severity after inspecting results.
6. Preserve original ID mapping, sample order, and CSV validation. Assert that model parameters and buffers are unchanged after inference. The network still has **4,298,970 unique learned parameters**; fixed transforms and averaging introduce none.

Implementation references: [author augmentation helper](https://github.com/zhangmarvin/memo/blob/main/cifar-10-exps/utils/third_party.py), [model/helper definition](https://github.com/zhangmarvin/memo/blob/main/cifar-10-exps/utils/test_helpers.py), and the paper's Eq. 1. The authors [warn that released code was cleaned after producing the paper results](https://github.com/zhangmarvin/memo), so this is not a byte-for-byte reproduction claim.

**Do not run the author's `test_adapt.py` as the solution.** It calls `loss.backward()` and `optimizer.step()` using each test image. Even setting its iteration count to zero would only produce ordinary single-view inference, not the required probability average. We need only the frozen marginal-prediction calculation and view generator. No MEMO/Tent updates, pseudo-label training, hidden-image BN calibration, external checkpoints, or external datasets.

## Rules and one bounded decision

I re-read the authenticated [competition Rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules) and [Overview](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview) during this research.

- Section 4 caps the **final model**, including all trainable components, at 5M parameters.
- Section 5 permits distillation from scratch-trained teachers using allowed training data; no separate teacher-size cap is stated.
- Section 3 forbids hidden evaluation data for training, validation, or model selection outside the official Kaggle submission process. Section 11 describes the public leaderboard as feedback.
- TTA and inference ensembles are not explicitly named. A shared-weight frozen transform/average is consistent with the parameter/data restrictions as written **by interpretation**, not explicit staff confirmation. Staff retains final authority. The narrow remaining question is whether fixed shared-weight inference averaging is allowed—not whether distillation in general is allowed.
- The Overview tooltip gives the deadline as **November 6, 2026, 23:59:59 Eastern (GMT-0500)**.

If the user authorizes implementation, the bounded plan is one offline paired check followed by, at most, one official candidate:

1. Preserve the incumbent. Implement and verify the single fixed inference contract above.
2. Evaluate it once on the **45K-trained counterpart** and the existing 5,000 held-out training images. Never use the all-50K model to claim validation accuracy on those images.
3. Before evaluation, explicitly approve the decision rule. A proposed cost-aware rule for this inference-only candidate is positive paired top-1 improvement on both existing 2,000/3,000 partitions, with paired counts and uncertainty reported. This is not a significance guarantee: both partitions have historical reuse. Ties or regressions do not justify replacing the incumbent, and a failed check does not authorize another policy/seed sweep.
4. Subject to rule clearance and that declared comparison, apply the identical frozen policy to the full-data model and verify one new CSV. Make at most one deliberately labeled official submission, preserving the original.

This is still an empirical check: no research can replace measuring an unknown hidden-set score. It eliminates the new training campaign, not the need for verification. **No existing project threshold was changed during this research.**

## Corrections to earlier advice

- **The 0.5-point gate is not a Kaggle rule.** It was a project heuristic, not a significance test. Escalating it to 97.94% as the incumbent improved can reject useful small gains; retaining it requires an explicit practical justification. Reconsider it before a new candidate, never after observing that candidate. Relaxing it would not retroactively turn the latest wider model's -14 examples or the last-block fine-tune's zero audit gain into improvements.
- **No hard model ceiling was established.** We measured a limited set of training runs, not a bound on what a 4.3M residual network can achieve.
- **The validation/public difference does not prove a particular shift.** The latest 97.44% and 88.40% also belong to different fitted checkpoints (45K comparison versus all-50K retrain). Even the earlier same-checkpoint gap identifies a symptom, not its cause.
- **No demonstrated export fix was found.** Source inspection shows the same RGB decode, checkpoint normalization, evaluation mode and ID/sample-order mapping in the inference path. The complete prior small-KD CPU/GPU prediction CSVs were re-compared during this research and are byte-identical, SHA-256 `e25f03b6a5c9a91074af76bccc3d0781657153a29fc55e53bd2db3f40392a77d`. That reduces support for a local export/backend discrepancy; it does not verify hidden labels or eliminate every possible evaluation issue.

## Why not another architecture or a new bundle now?

The [primary recipe review](research-90-primary-recipes.md) found real author-backed under-cap alternatives, but not a demonstrated better recipe for this competition:

- **Shake-Shake-26-2x32d + AutoAugment/Cutout:** approximately 2.9M parameters, reported 97.5% standard CIFAR accuracy, **1,800 epochs**. The [official recipe](https://github.com/tensorflow/models/blob/fd34f711f319d8c6fe85110d9df6e1784cc5a6ca/research/autoaugment/README.md) uses LR0.01, decay0.001, batch128 and an old TF1/Python2 implementation. Policy count, augmentation ordering and branch-weight sampling have fidelity traps. This is not an automatic improvement over our 97.44% on a different split.
- **DrNAS:** a genuine from-scratch 600-epoch alternative with reported 97.54% standard CIFAR accuracy and about4.1M inference parameters. It introduces a new convolutional DAG and another architectural rule interpretation, without a matching natural-shift or project result.
- **SWA:** valid single-model averaging with under-cap CIFAR evidence, but requires a properly sampled training trajectory and training-only BN refresh. It is not a free operation on our one final checkpoint, and the measured gains in the cited compact model are modest.
- **Model merging/re-basin:** the [primary paper](https://arxiv.org/html/2209.04836v6) specifically describes improved loss **but not accuracy** for its split-data model-patching comparison and documents failures of naive interpolation. It does not establish a +1.6-point shortcut for our models.

I also inspected the [same-competition public reproduction thread](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/discussion/754332) and [Saber's published aggregate README](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle). Their two same-recipe development runs report +0.22 points from flip averaging, but their bundled 300-epoch robustness/10-view result is only86.2% public; this does **not** isolate TTA's effect. Their [exploratory aggregate record](https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle/blob/feat/five-day-autoresearch/research/README.md) also records a large synthetic-stress improvement that failed to improve the public score. Those are author-reported corroborating observations, not independently reproduced results or permission to inspect their prediction files. No current-competition 90% winning recipe was verified in the public sources examined.

## What is and is not delivered

Research-phase deliverables were two primary-source reviews, this integration decision, a live rules/deadline check, and a local inference-parity audit. That phase established no new accuracy result and launched no training or submission. The later authorized execution is reported at the top of this document: the fixed policy failed validation and was not adopted. The 88.40% model remains intact; no result here establishes 90%.
