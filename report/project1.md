# Compact residual learning under a five-million-parameter limit

Mohamed Awadalla | Team The New York Knicks

Deep Learning Project 1 | November 11, 2026

https://github.com/Mo-Awadalla/new-york-knicks-cifar10

### Abstract

This project studies how augmentation and knowledge distillation improve a compact residual image classifier trained from scratch. The submitted WRN-22-4 has 4,298,970 trainable parameters. Kaggle reports 85.90% private test accuracy and 88.40% public test accuracy for submission 57968810. A separate model of the same recipe reaches 97.44% on the fixed validation split. Cutout improves the initial baseline, while CutMix and distillation produce the retained submission. Further validation experiments do not justify replacing it. Validation improvements do not consistently transfer to the hidden test portions. These results do not identify the hidden test distribution.

Table 1. Final outcome and its separate sources.

| Quantity | Result | Attribution |
| --- | --- | --- |
| Private test accuracy | 85.90% | Kaggle submission 57968810 |
| Public test accuracy | 88.40% | Same submission, different hidden test portion |
| Recipe validation accuracy | 97.44% | Separate 45,000-image model; 4,872 of 5,000 correct |
| Trainable parameters | 4,298,970 | WRN-22-4 |

### 1. Data and evaluation protocol

The official competition provides 50,000 labeled training images and 2,000 test images with hidden labels. Only staff-provided images are used. No external training data or pretrained weights are used. The residual classifier must fit a 5,000,000-trainable-parameter limit.

A deterministic, class-stratified split reserves 5,000 images for validation and 45,000 for training. Within each class, image IDs are ranked by SHA-256 of the prefix knicks-v1: followed by the ID. The first 500 images per class form validation. Channel normalization uses only the rows used to train each model. Validation has no training augmentation.

## 2. Architecture and selected methodology

The final WRN-22-4 is a preactivation Wide Residual Network [1] with 4,298,970 trainable parameters. The count includes convolution weights, BatchNorm affine parameters, and the classifier. BatchNorm running statistics are buffers, not trainable parameters.

Table 2. The final student architecture for a 32 by 32 RGB image.

| Stage | Output | Structure |
| --- | --- | --- |
| Stem | 16 x 32 x 32 | 3 x 3 convolution, stride 1 |
| Residual group 1 | 64 x 32 x 32 | 3 residual blocks, stride 1 |
| Residual group 2 | 128 x 16 x 16 | 3 residual blocks, first block stride 2 |
| Residual group 3 | 256 x 8 x 8 | 3 residual blocks, first block stride 2 |
| Readout | 10 logits | BatchNorm, ReLU, global average pooling, linear 256 to 10 |

Each branch applies BatchNorm and ReLU before each of its two 3 by 3 convolutions. A 1 by 1 projection matches the shortcut when channels or resolution change. Otherwise, the shortcut is the identity. Global average pooling avoids a large fully connected layer.

### Optimization and augmentation

The selected recipe trains for 200 epochs with seed 1, batch size 128, and SGD with Nesterov momentum 0.9. The maximum learning rate is 0.1. A 5-epoch linear warmup precedes cosine decay. Weight decay is 0.0005. Random crops and horizontal flips provide the basic training augmentation. CutMix [3] replaces a patch with pixels from another training image and mixes labels by the actual pasted area. Its probability is 0.5 and its beta-distribution alpha is 1.

### Distillation into one inference model

The student learns from the average softened probabilities of 2 scratch-trained WRN-28-10 teachers [4]. One uses CutMix and the other SAM with Cutout [2, 5]. Each has 36,479,194 parameters. At temperature 4, the objective weights CutMix loss by 0.1 and teacher-to-student KL divergence by 0.9, with temperature-squared scaling. Teachers receive the student's augmented pixels with their own normalization.

Validation teachers see only the training split. Full-data teachers and the final student see all official training images. Inference uses one student and one view, without test-time augmentation. No explicit staff confirmation of a parameter-cap exemption for the larger training-only teachers was found. This is an unresolved rules interpretation, not approval or a finding of violation.

## 3. Experiments and scored submissions

Table 3 uses final checkpoints and the same 5,000-image validation split. All runs train for 200 epochs. Initial comparisons use seed 0. The matched control and later portfolio use seed 1. Each comparison has one seed per treatment.

Table 3. Validation and Kaggle scores in percent. Test training rows describe the model actually submitted, not the validation model.

| Recipe | Model | Validation | Test training rows | Public test | Private test |
| --- | --- | --- | --- | --- | --- |
| Crop and flip | WRN-16-4 | 95.38 | 45,000 | 84.90 | 83.40 |
| Cutout | WRN-16-4 | 96.36 | 45,000 | 86.30 | 84.50 |
| WRN-22-4 Cutout | WRN-22-4 | 96.32 | Not submitted | None | None |
| Matched Cutout control | WRN-22-4 | 96.46 | Not submitted | None | None |
| SAM with Cutout | WRN-22-4 | 96.92 | 50,000 | 87.40 | 85.50 |
| CutMix | WRN-22-4 | 96.74 | 50,000 | 87.90 | 85.70 |
| Small-teacher distillation | WRN-22-4 | 97.32 | 50,000 | 87.50 | 85.30 |
| Large-teacher distillation | WRN-22-4 | 97.44 | 50,000 | 88.40 | 85.90 |

Missing scores mean no associated submission. For full-data submissions, validation belongs to a separate training-split checkpoint. The final full-data model has no held-out validation score.

### What improved, and what did not

Cutout adds 0.98 validation points over crop and flip. Extra depth with Cutout alone does not improve that result. Against the matched control, SAM adds 23 correct predictions and CutMix adds 14. Neither meets the required 25-image gain. This 0.5-point margin was operational, not statistical. Their full-data retrains proceeded by explicit user override, not because every stage met the promotion rule.

Small-teacher distillation adds 29 validation successes over CutMix. Larger teachers add only 6 more. The final private score exceeds the initial baseline by 2.50 points. Validation ranking does not consistently transfer. Small-teacher distillation beats CutMix on validation but scores lower on both Kaggle portions.

AugMix with a consistency objective [6] also failed to improve the matched-control validation result. Published robustness gains did not guarantee a gain in this comparison.

## 4. Learning dynamics and metric interpretation

![Figure 1. Actual per-epoch records from the fixed training and validation split. Panel a shows the full validation trajectory. Panel b magnifies its final quarter. Panel c shows the selected student's mixed-target distillation objective on a logarithmic axis. It is not validation cross-entropy.](figures/learning-curves.png)

Figure 1. Actual per-epoch records from the fixed training and validation split. Panel a shows the full validation trajectory. Panel b magnifies its final quarter. Panel c shows the selected student's mixed-target distillation objective on a logarithmic axis. It is not validation cross-entropy.

The full trajectories show why early-epoch accuracy is a poor basis for selecting these recipes. The late-epoch view separates improvements that the full vertical range compresses. Final checkpoint results, rather than the highest point on a curve, determine the reported validation score. The small separation near the end does not establish an architecture ceiling or predict a hidden-test gain.

### Mixed targets change the meaning of training metrics

CutMix presents images with two weighted labels. Distillation adds a temperature-scaled probability target. The logged primary-target training accuracy compares the prediction with only the original label. It ignores the pasted image's target weight and the teacher distribution. It is a diagnostic, not ordinary accuracy on clean training images.

For the selected validation run, the final primary-target diagnostic is 87.83%, while clean held-out validation accuracy is 97.44%. These values do not show that the model fits validation better than training. They measure different tasks. The final training objective is 0.1603, while validation cross-entropy is 0.0988. Their scales and targets differ, so subtracting them is not a meaningful generalization-gap estimate.

## 5. Validation errors and the generalization gap

![Figure 2. Aggregate confusion for the separate validation checkpoint, with 4,872 of 5,000 correct. Rows are true class IDs and columns are predicted IDs. Color shows row percentages. Nonzero annotations are image counts. No staff images or per-image labels are shown.](figures/validation-confusion.png)

Figure 2. Aggregate confusion for the separate validation checkpoint, with 4,872 of 5,000 correct. Rows are true class IDs and columns are predicted IDs. Color shows row percentages. Nonzero annotations are image counts. No staff images or per-image labels are shown.

The model makes 128 validation errors. Class 3 has the lowest recall, 93.20%. Class 8 has the highest, 99.40%. Classes 3 and 5 have the largest two-way confusion, with 34 errors. Hidden labels are unavailable, so these counts cannot explain private-test errors.

Recipe validation exceeds private test accuracy by 11.54 points. This compares different checkpoints and evaluation sets, not training and test accuracy of one model. Public accuracy exceeds private accuracy by 2.50 points for the same submitted model. These are separate hidden portions, not repeated validation measurements.

The split supports comparisons within the official training pool, not a guarantee of hidden-test transfer. Historical validation reuse can make later choices optimistic. The score discrepancy alone identifies neither its cause nor a specific test dataset, corruption process, or class balance.

## 6. Follow-up experiments and lessons

The follow-ups in Table 4 did not justify replacing the retained student. The repository records their outcomes and rejection decisions. Their reused validation partitions are not independent confirmation sets.

Table 4. Follow-ups that did not replace the selected single-view student.

| Candidate | Observed validation outcome | Decision |
| --- | --- | --- |
| Wider, shallower WRN-16-5 | Fewer correct predictions than the retained student | No full-data retrain |
| Classifier-only fine-tuning | No change in validation predictions | Rejected |
| Final-block fine-tuning | A small overall gain, but no net gain on its audit partition | Rejected |
| Direct classifier weight edits | The selected edit improved tuning accuracy but reduced audit accuracy | Rejected |
| Augmented-view averaging | Accuracy fell on both tuning and audit partitions | No test submission |

### A local plateau is not a universal limit

Larger teachers add little for this student. The wider student and targeted fine-tuning also fail to provide a useful replacement. This is diminishing return for the tested recipes, not a universal accuracy limit. One seed and reused validation cannot establish that limit.

### Selection discipline matters as much as another candidate

The selected weight edit improves its tuning subset but harms its audit subset. Prior validation reuse limits that audit. Augmented-view averaging also shows that lower probability loss need not improve top-1 accuracy. The failed policy was not used for submitted test predictions.

### What the project supports

The final result is 85.90% private test accuracy, not the public score or 97.44% validation accuracy. Augmentation and distillation improve the baseline without closing this discrepancy. Stronger conclusions need a prespecified protocol and independent evaluation, not more tuning on the same held-out rows.

## 7. Reproducibility and references

The repository includes the canonical model, training and inference code, configuration files, aggregate evidence, and an executed analysis notebook. The notebook constructs the architecture, prints the parameter count and separately attributed scores, and analyzes learning curves and validation errors. The report and notebook load checksum-checked evidence from results. The private score comes from the captured Kaggle response, not a local evaluation with hidden labels.

The source-only analysis needs neither credentials nor the original images and checkpoints. It regenerates the aggregate analysis, not the trained weights. Checkpoint replay requires the authorized official data and the matching saved model. The guide at docs/reproduction.md documents that distinction and the training and inference commands. Staff images, per-image labels, credentials, and checkpoint blobs are excluded from the repository.

### Checkpoint provenance

Final model trained on 50,000 images. SHA-256 follows.

`9a5a28fa5e00dc4e929d5f0b25f7c3bd14a5123c43d6cb6409407be039460a49`

Separate validation model trained on 45,000 images. SHA-256 follows.

`42fa268b43159b51eb9a936f9d00e91a5cfdd0194e7abb884bb79333658f9719`

### References

[1] Sergey Zagoruyko and Nikos Komodakis. Wide Residual Networks. 2016. arXiv:1605.07146. https://arxiv.org/abs/1605.07146

[2] Terrance DeVries and Graham W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. arXiv:1708.04552. https://arxiv.org/abs/1708.04552

[3] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. 2019. arXiv:1905.04899. https://arxiv.org/abs/1905.04899

[4] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. 2015. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

[5] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. 2020. arXiv:2010.01412. https://arxiv.org/abs/2010.01412

[6] Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. 2019. arXiv:1912.02781. https://arxiv.org/abs/1912.02781
