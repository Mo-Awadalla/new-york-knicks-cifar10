"""Build the Project 1 PDF and Markdown from the same report content."""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from project1_evidence import ProjectEvidence, load_evidence


ROOT = Path(__file__).resolve().parents[1]
NAVY = "#17324d"
BLUE = "#2164a8"
ORANGE = "#cc6c29"


@dataclass(frozen=True)
class Text:
    text: str
    style: str = "body"


@dataclass(frozen=True)
class ReportTable:
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    widths: tuple[float, ...]
    highlighted_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class Figure:
    path: Path
    caption: str
    width: float
    height: float


@dataclass(frozen=True)
class Page:
    title: str
    blocks: tuple[Text | ReportTable | Figure, ...]


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def arch_name(value: str) -> str:
    return value.replace("wrn", "WRN-", 1).replace("_", "-")


def make_figures(evidence: ProjectEvidence, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    selected_name = next(
        row["name"] for row in evidence.experiments
        if row["submission_ref"] == evidence.final.submission_ref
    )
    selected = evidence.curves[selected_name]
    series = (
        ("Crop and flip", "Crop and flip", "#707b85"),
        ("Cutout", "Cutout", ORANGE),
        ("CutMix", "CutMix", "#41866b"),
        (selected_name, "Selected distilled student", BLUE),
    )
    fig = plt.figure(figsize=(7, 4.34))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 1))
    full = fig.add_subplot(grid[0, :])
    late = fig.add_subplot(grid[1, 0])
    objective = fig.add_subplot(grid[1, 1])
    final_epoch = selected[-1]["epoch"]
    late_start = int(0.75 * final_epoch)
    for name, label, color in series:
        records = evidence.curves[name]
        epochs = [row["epoch"] for row in records]
        accuracy = [100 * row["val_accuracy"] for row in records]
        full.plot(epochs, accuracy, label=label, color=color, linewidth=1.4)
        late.plot(epochs, accuracy, color=color, linewidth=1.4)
    full.set(title="(a) Held-out validation accuracy", ylabel="Accuracy (%)")
    full.set_xlim(1, final_epoch)
    full.set_ylim(30, 100)
    full.legend(loc="lower right", ncol=2, frameon=False)
    late.set(
        title="(b) Final quarter of training", xlabel="Epoch",
        ylabel="Validation accuracy (%)", xlim=(late_start, final_epoch),
    )
    late_values = [
        100 * row["val_accuracy"]
        for name, _, _ in series for row in evidence.curves[name]
        if row["epoch"] >= late_start
    ]
    late.set_ylim(min(late_values) - 0.25, max(late_values) + 0.25)
    objective.plot(
        [row["epoch"] for row in selected],
        [row["train_loss"] for row in selected],
        color=BLUE, linewidth=1.5,
    )
    objective.set(
        title="(c) Student training objective", xlabel="Epoch",
        ylabel="Distillation plus CutMix loss", xlim=(1, final_epoch),
    )
    objective.set_yscale("log")
    for axis in (full, late, objective):
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    learning = directory / "learning-curves.png"
    fig.savefig(learning, dpi=200)
    plt.close(fig)

    counts = np.asarray(evidence.confusion, dtype=int)
    row_percent = 100 * counts / counts.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(6, 4.7))
    image = axis.imshow(row_percent, vmin=0, vmax=100, cmap="Blues")
    axis.set(
        xlabel="Predicted class ID", ylabel="True class ID",
        xticks=np.arange(len(counts)), yticks=np.arange(len(counts)),
    )
    for row in range(len(counts)):
        for column in range(len(counts)):
            if counts[row, column]:
                axis.text(
                    column, row, str(counts[row, column]),
                    ha="center", va="center", fontsize=11,
                    color="white" if row_percent[row, column] > 50 else NAVY,
                )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.035)
    colorbar.set_label("Share of true class (%)")
    fig.tight_layout(pad=1.2)
    confusion = directory / "validation-confusion.png"
    fig.savefig(confusion, dpi=200)
    plt.close(fig)
    return {"learning": learning, "confusion": confusion}


def make_pages(
    evidence: ProjectEvidence,
    config: dict,
    manifest: dict,
    repository: str,
    figures: dict[str, Path],
) -> tuple[Page, ...]:
    final = evidence.final
    validation = evidence.validation
    experiments = {row["name"]: row for row in evidence.experiments}
    basic = experiments["Crop and flip"]
    cutout = experiments["Cutout"]
    control = experiments["Matched Cutout control"]
    sam = experiments["SAM with Cutout"]
    cutmix = experiments["CutMix"]
    small = experiments["Small-teacher distillation"]
    selected = next(
        row for row in evidence.experiments
        if row["submission_ref"] == final.submission_ref
    )
    last_epoch = evidence.curves[selected["name"]][-1]
    counts = np.asarray(evidence.confusion, dtype=int)
    off_diagonal = counts.copy()
    np.fill_diagonal(off_diagonal, 0)
    pair_counts = off_diagonal + off_diagonal.T
    pair_row, pair_column = np.unravel_index(np.argmax(pair_counts), pair_counts.shape)
    recall = counts.diagonal() / counts.sum(axis=1)
    weakest = int(np.argmin(recall))
    strongest = int(np.argmax(recall))
    teachers = manifest["distillation_teachers"]
    epochs = config["epochs"]
    architecture = arch_name(final.arch)
    table_rows = tuple(
        (
            row["name"],
            arch_name(row["arch"]),
            f"{100 * row['validation_accuracy']:.2f}",
            "Not submitted" if row["test_training_rows"] is None
            else f"{row['test_training_rows']:,}",
            "None" if row["public_accuracy"] is None
            else f"{100 * row['public_accuracy']:.2f}",
            "None" if row["private_accuracy"] is None
            else f"{100 * row['private_accuracy']:.2f}",
        )
        for row in evidence.experiments
    )
    gain_cutout = 100 * (cutout["validation_accuracy"] - basic["validation_accuracy"])
    gain_private = 100 * (final.private_accuracy - basic["private_accuracy"])
    gap_private = 100 * (validation.accuracy - final.private_accuracy)
    gap_public = 100 * (final.public_accuracy - final.private_accuracy)
    hard_weight = 1 - config["kd_alpha"]

    return (
        Page("Compact residual learning under a five-million-parameter limit", (
            Text("Mohamed Awadalla | Team The New York Knicks", "byline"),
            Text("Deep Learning Project 1 | November 11, 2026", "byline"),
            Text(repository, "repository"),
            Text("Abstract", "heading"),
            Text(
                f"This project studies how augmentation and knowledge distillation improve a compact "
                f"residual image classifier trained from scratch. The submitted {architecture} has "
                f"{final.params:,} trainable parameters. Kaggle reports {percent(final.private_accuracy)} "
                f"private test accuracy and {percent(final.public_accuracy)} public test accuracy for "
                f"submission {final.submission_ref}. A separate model of the same recipe reaches "
                f"{percent(validation.accuracy)} on the fixed validation split. Cutout improves the "
                f"initial baseline, while CutMix and distillation produce the retained submission. "
                f"Further validation experiments do not justify replacing it. Validation "
                f"improvements do not consistently transfer to the hidden test portions. "
                f"These results do not identify the hidden test distribution."
            ),
            ReportTable(
                "Table 1. Final outcome and its separate sources.",
                ("Quantity", "Result", "Attribution"),
                (
                    ("Private test accuracy", percent(final.private_accuracy),
                     f"Kaggle submission {final.submission_ref}"),
                    ("Public test accuracy", percent(final.public_accuracy),
                     "Same submission, different hidden test portion"),
                    ("Recipe validation accuracy", percent(validation.accuracy),
                     f"Separate {validation.train_rows:,}-image model; "
                     f"{validation.correct:,} of {validation.rows:,} correct"),
                    ("Trainable parameters", f"{final.params:,}", architecture),
                ),
                (151, 91, 262),
                (0,),
            ),
            Text("1. Data and evaluation protocol", "heading"),
            Text(
                f"The official competition provides {final.train_rows:,} labeled training images "
                f"and {manifest['test_rows']:,} test images with hidden labels. Only staff-provided "
                f"images are used. No external training data or pretrained weights are used. "
                f"The residual classifier must fit a 5,000,000-trainable-parameter limit."
            ),
            Text(
                f"A deterministic, class-stratified split reserves {validation.rows:,} images for "
                f"validation and {validation.train_rows:,} for training. Within each class, image IDs "
                f"are ranked by SHA-256 of the prefix knicks-v1: followed by the ID. The first "
                f"{validation.rows // len(counts):,} images per class form validation. Channel "
                f"normalization uses only the rows used to train each model. Validation has no "
                f"training augmentation."
            ),
        )),
        Page("2. Architecture and selected methodology", (
            Text(
                f"The final {architecture} is a preactivation Wide Residual Network [1] with "
                f"{final.params:,} trainable parameters. The count includes convolution weights, "
                f"BatchNorm affine parameters, and the classifier. BatchNorm running statistics "
                f"are buffers, not trainable parameters."
            ),
            ReportTable(
                "Table 2. The final student architecture for a 32 by 32 RGB image.",
                ("Stage", "Output", "Structure"),
                (
                    ("Stem", "16 x 32 x 32", "3 x 3 convolution, stride 1"),
                    ("Residual group 1", "64 x 32 x 32", "3 residual blocks, stride 1"),
                    ("Residual group 2", "128 x 16 x 16", "3 residual blocks, first block stride 2"),
                    ("Residual group 3", "256 x 8 x 8", "3 residual blocks, first block stride 2"),
                    ("Readout", "10 logits", "BatchNorm, ReLU, global average pooling, linear 256 to 10"),
                ),
                (119, 115, 270),
            ),
            Text(
                "Each branch applies BatchNorm and ReLU before each of its two 3 by 3 "
                "convolutions. A 1 by 1 projection matches the shortcut when channels or "
                "resolution change. Otherwise, the shortcut is the identity. Global average "
                "pooling avoids a large fully connected layer."
            ),
            Text("Optimization and augmentation", "heading"),
            Text(
                f"The selected recipe trains for {epochs} epochs with seed {config['seed']}, "
                f"batch size {config['batch']}, and SGD with Nesterov momentum "
                f"{config['momentum']:g}. The maximum learning rate is {config['lr']:g}. "
                f"A {config['warmup']}-epoch linear warmup precedes cosine decay. Weight decay "
                f"is {config['wd']:g}. Random crops and horizontal flips provide the basic "
                f"training augmentation. CutMix [3] replaces a patch with pixels from another "
                f"training image and mixes labels by the actual pasted area. Its probability is "
                f"{manifest['cutmix_probability']:g} and its beta-distribution alpha is "
                f"{manifest['cutmix_alpha']:g}."
            ),
            Text("Distillation into one inference model", "heading"),
            Text(
                f"The student learns from the average softened probabilities of {len(teachers)} "
                f"scratch-trained WRN-28-10 teachers [4]. One uses CutMix and the other SAM "
                f"with Cutout [2, 5]. Each has {teachers[0]['params']:,} parameters. At temperature "
                f"{config['kd_temperature']:g}, the objective weights CutMix loss by "
                f"{hard_weight:.1f} and teacher-to-student KL divergence by "
                f"{config['kd_alpha']:g}, with temperature-squared scaling. Teachers receive "
                f"the student's augmented pixels with their own normalization."
            ),
            Text(
                "Validation teachers see only the training split. Full-data teachers and the "
                "final student see all official training images. Inference uses one student "
                "and one view, without test-time augmentation. No explicit staff confirmation "
                "of a parameter-cap exemption for the larger training-only teachers was found. "
                "This is an unresolved rules interpretation, not approval or a finding of violation."
            ),
        )),
        Page("3. Experiments and scored submissions", (
            Text(
                f"Table 3 uses final checkpoints and the same {validation.rows:,}-image "
                f"validation split. All runs train for {epochs} epochs. Initial comparisons "
                f"use seed {basic['seed']}. The matched control and later portfolio use seed "
                f"{control['seed']}. Each comparison has one seed per treatment."
            ),
            ReportTable(
                "Table 3. Validation and Kaggle scores in percent. Test training rows describe "
                "the model actually submitted, not the validation model.",
                ("Recipe", "Model", "Validation", "Test training rows", "Public test", "Private test"),
                table_rows,
                (136, 66, 70, 80, 76, 76),
                (len(table_rows) - 1,),
            ),
            Text(
                "Missing scores mean no associated submission. For full-data submissions, "
                "validation belongs to a separate training-split checkpoint. The final "
                "full-data model has no held-out validation score."
            ),
            Text("What improved, and what did not", "heading"),
            Text(
                f"Cutout adds {gain_cutout:.2f} validation points over crop and flip. Extra "
                f"depth with Cutout alone does not improve that result. Against the matched "
                f"control, SAM adds {sam['validation_correct'] - control['validation_correct']} "
                f"correct predictions and CutMix adds "
                f"{cutmix['validation_correct'] - control['validation_correct']}. Neither meets "
                f"the required 25-image gain. This 0.5-point margin was operational, not "
                f"statistical. Their full-data retrains proceeded by explicit user override, "
                f"not because every stage met the promotion rule."
            ),
            Text(
                f"Small-teacher distillation adds "
                f"{small['validation_correct'] - cutmix['validation_correct']} validation "
                f"successes over CutMix. Larger teachers add only "
                f"{validation.correct - small['validation_correct']} more. The final private "
                f"score exceeds the initial baseline by {gain_private:.2f} points. Validation "
                f"ranking does not consistently transfer. Small-teacher distillation beats "
                f"CutMix on validation but scores lower on both Kaggle portions."
            ),
            Text(
                "AugMix with a consistency objective [6] also failed to improve the "
                "matched-control validation result. Published robustness gains did not "
                "guarantee a gain in this comparison."
            ),
        )),
        Page("4. Learning dynamics and metric interpretation", (
            Figure(
                figures["learning"],
                "Figure 1. Actual per-epoch records from the fixed training and validation split. "
                "Panel a shows the full validation trajectory. Panel b magnifies its final quarter. "
                "Panel c shows the selected student's mixed-target distillation objective on a "
                "logarithmic axis. It is not validation cross-entropy.",
                504, 312.48,
            ),
            Text(
                "The full trajectories show why early-epoch accuracy is a poor basis for "
                "selecting these recipes. The late-epoch view separates improvements that "
                "the full vertical range compresses. Final checkpoint results, rather than "
                "the highest point on a curve, determine the reported validation score. "
                "The small separation near the end does not establish an architecture ceiling "
                "or predict a hidden-test gain."
            ),
            Text("Mixed targets change the meaning of training metrics", "heading"),
            Text(
                "CutMix presents images with two weighted labels. Distillation adds a "
                "temperature-scaled probability target. The logged primary-target training "
                "accuracy compares the prediction with only the original label. It ignores "
                "the pasted image's target weight and the teacher distribution. It is a "
                "diagnostic, not ordinary accuracy on clean training images."
            ),
            Text(
                f"For the selected validation run, the final primary-target diagnostic is "
                f"{percent(last_epoch['train_primary_target_accuracy'])}, while clean held-out "
                f"validation accuracy is {percent(validation.accuracy)}. These values do not "
                f"show that the model fits validation better than training. They measure "
                f"different tasks. The final training objective is "
                f"{last_epoch['train_loss']:.4f}, while validation cross-entropy is "
                f"{last_epoch['val_loss']:.4f}. Their scales and targets differ, so subtracting "
                f"them is not a meaningful generalization-gap estimate."
            ),
        )),
        Page("5. Validation errors and the generalization gap", (
            Figure(
                figures["confusion"],
                f"Figure 2. Aggregate confusion for the separate validation checkpoint, with "
                f"{validation.correct:,} of {validation.rows:,} correct. Rows are true class IDs "
                f"and columns are predicted IDs. Color shows row percentages. Nonzero "
                f"annotations are image counts. No staff images or per-image labels are shown.",
                430, 337.03,
            ),
            Text(
                f"The model makes {validation.rows - validation.correct} validation errors. "
                f"Class {weakest} has the lowest recall, {percent(float(recall[weakest]))}. "
                f"Class {strongest} has the highest, {percent(float(recall[strongest]))}. "
                f"Classes {pair_row} and {pair_column} have the largest two-way confusion, "
                f"with {int(pair_counts[pair_row, pair_column])} errors. Hidden labels are "
                f"unavailable, so these counts cannot explain private-test errors."
            ),
            Text(
                f"Recipe validation exceeds private test accuracy by {gap_private:.2f} points. "
                f"This compares different checkpoints and evaluation sets, not training and "
                f"test accuracy of one model. Public accuracy exceeds private accuracy by "
                f"{gap_public:.2f} points for the same submitted model. These are separate "
                f"hidden portions, not repeated validation measurements."
            ),
            Text(
                "The split supports comparisons within the official training pool, not a "
                "guarantee of hidden-test transfer. Historical validation reuse can make "
                "later choices optimistic. The score discrepancy alone identifies neither "
                "its cause nor a specific test dataset, corruption process, or class balance."
            ),
        )),
        Page("6. Follow-up experiments and lessons", (
            Text(
                "The follow-ups in Table 4 did not justify replacing the retained student. "
                "The repository records their outcomes and rejection decisions. Their reused "
                "validation partitions are not independent confirmation sets."
            ),
            ReportTable(
                "Table 4. Follow-ups that did not replace the selected single-view student.",
                ("Candidate", "Observed validation outcome", "Decision"),
                (
                    ("Wider, shallower WRN-16-5", "Fewer correct predictions than the retained student", "No full-data retrain"),
                    ("Classifier-only fine-tuning", "No change in validation predictions", "Rejected"),
                    ("Final-block fine-tuning", "A small overall gain, but no net gain on its audit partition", "Rejected"),
                    ("Direct classifier weight edits", "The selected edit improved tuning accuracy but reduced audit accuracy", "Rejected"),
                    ("Augmented-view averaging", "Accuracy fell on both tuning and audit partitions", "No test submission"),
                ),
                (139, 259, 106),
            ),
            Text("A local plateau is not a universal limit", "heading"),
            Text(
                "Larger teachers add little for this student. The wider student and targeted "
                "fine-tuning also fail to provide a useful replacement. This is diminishing "
                "return for the tested recipes, not a universal accuracy limit. One seed "
                "and reused validation cannot establish that limit."
            ),
            Text("Selection discipline matters as much as another candidate", "heading"),
            Text(
                "The selected weight edit improves its tuning subset but harms its audit "
                "subset. Prior validation reuse limits that audit. Augmented-view averaging "
                "also shows that lower probability loss need not improve top-1 accuracy. "
                "The failed policy was not used for submitted test predictions."
            ),
            Text("What the project supports", "heading"),
            Text(
                f"The final result is {percent(final.private_accuracy)} private test accuracy, "
                f"not the public score or {percent(validation.accuracy)} validation accuracy. "
                f"Augmentation and distillation improve the baseline without closing this "
                f"discrepancy. Stronger conclusions need a prespecified protocol and independent "
                f"evaluation, not more tuning on the same held-out rows."
            ),
        )),
        Page("7. Reproducibility and references", (
            Text(
                "The repository includes the canonical model, training and inference code, "
                "configuration files, aggregate evidence, and an executed analysis notebook. "
                "The notebook constructs the architecture, prints the parameter count and "
                "separately attributed scores, and analyzes learning curves and validation "
                "errors. The report and notebook load checksum-checked evidence from results. "
                "The private score comes from the captured Kaggle response, not a local "
                "evaluation with hidden labels."
            ),
            Text(
                "The source-only analysis needs neither credentials nor the original images "
                "and checkpoints. It regenerates the aggregate analysis, not the trained "
                "weights. Checkpoint replay requires the authorized official data and the "
                "matching saved model. The guide at docs/reproduction.md documents that distinction "
                "and the training and inference commands. Staff images, per-image labels, "
                "credentials, and checkpoint blobs are excluded from the repository."
            ),
            Text("Checkpoint provenance", "heading"),
            Text(f"Final model trained on {final.train_rows:,} images. SHA-256 follows."),
            Text(final.checkpoint_sha256, "hash"),
            Text(
                f"Separate validation model trained on {validation.train_rows:,} images. "
                f"SHA-256 follows."
            ),
            Text(validation.checkpoint_sha256, "hash"),
            Text("References", "heading"),
            Text(
                "[1] Sergey Zagoruyko and Nikos Komodakis. Wide Residual Networks. "
                "2016. arXiv:1605.07146. https://arxiv.org/abs/1605.07146",
                "reference",
            ),
            Text(
                "[2] Terrance DeVries and Graham W. Taylor. Improved Regularization of "
                "Convolutional Neural Networks with Cutout. 2017. arXiv:1708.04552. "
                "https://arxiv.org/abs/1708.04552",
                "reference",
            ),
            Text(
                "[3] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, "
                "and Youngjoon Yoo. CutMix: Regularization Strategy to Train Strong "
                "Classifiers with Localizable Features. 2019. arXiv:1905.04899. "
                "https://arxiv.org/abs/1905.04899",
                "reference",
            ),
            Text(
                "[4] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge "
                "in a Neural Network. 2015. arXiv:1503.02531. "
                "https://arxiv.org/abs/1503.02531",
                "reference",
            ),
            Text(
                "[5] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. "
                "Sharpness-Aware Minimization for Efficiently Improving Generalization. "
                "2020. arXiv:2010.01412. https://arxiv.org/abs/2010.01412",
                "reference",
            ),
            Text(
                "[6] Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, "
                "and Balaji Lakshminarayanan. AugMix: A Simple Data Processing Method "
                "to Improve Robustness and Uncertainty. 2019. arXiv:1912.02781. "
                "https://arxiv.org/abs/1912.02781",
                "reference",
            ),
        )),
    )


def paragraph_markup(text: str) -> str:
    fragments = re.split(r"(https://[^\s]+)", text)
    return "".join(
        f'<link href="{escape(fragment, quote=True)}" color="{BLUE}">{escape(fragment)}</link>'
        if fragment.startswith("https://") else escape(fragment)
        for fragment in fragments
    )


def pdf_styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body", fontName="Times-Roman", fontSize=10.5, leading=14,
        textColor=colors.HexColor("#1e2730"), spaceAfter=8,
        alignment=TA_LEFT, allowWidows=0, allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=body, fontName="Helvetica-Bold", fontSize=17,
            leading=21, textColor=colors.HexColor(NAVY), spaceAfter=14,
            keepWithNext=True,
        ),
        "cover": ParagraphStyle(
            "cover", parent=body, fontName="Helvetica-Bold", fontSize=22,
            leading=26, textColor=colors.HexColor(NAVY), spaceAfter=13,
            keepWithNext=True,
        ),
        "heading": ParagraphStyle(
            "heading", parent=body, fontName="Helvetica-Bold", fontSize=11.5,
            leading=15, textColor=colors.HexColor(NAVY), spaceBefore=7,
            spaceAfter=6, keepWithNext=True,
        ),
        "byline": ParagraphStyle(
            "byline", parent=body, fontName="Helvetica", fontSize=10.5,
            leading=14, spaceAfter=3, keepWithNext=True,
        ),
        "repository": ParagraphStyle(
            "repository", parent=body, fontName="Helvetica", fontSize=10,
            leading=14, spaceAfter=12, keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "caption", parent=body, fontSize=10, leading=12.5,
            textColor=colors.HexColor("#485462"), spaceAfter=9,
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=10, leading=12.5, spaceAfter=0,
        ),
        "table_header": ParagraphStyle(
            "table_header", parent=body, fontName="Helvetica-Bold",
            fontSize=10, leading=12.5, textColor=colors.white, spaceAfter=0,
        ),
        "hash": ParagraphStyle(
            "hash", parent=body, fontName="Courier", fontSize=10,
            leading=13, spaceAfter=9,
        ),
        "reference": ParagraphStyle(
            "reference", parent=body, fontSize=10.2, leading=13.2,
            leftIndent=17, firstLineIndent=-17, spaceAfter=7,
        ),
    }


def pdf_table(content: ReportTable, styles: dict[str, ParagraphStyle]) -> Table:
    cells = [
        [Paragraph(escape(value), styles["table_header"]) for value in content.headers]
    ]
    cells.extend(
        [Paragraph(escape(value), styles["cell"]) for value in row]
        for row in content.rows
    )
    table = Table(cells, colWidths=content.widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f2f5f7"), colors.white]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#c5ced6")),
    ]
    for row in content.highlighted_rows:
        commands.append(("BACKGROUND", (0, row + 1), (-1, row + 1), colors.HexColor("#e6eff8")))
    table.setStyle(TableStyle(commands))
    return table


def draw_footer(canvas, document) -> None:
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#c5ced6"))
    canvas.setLineWidth(0.5)
    canvas.line(54, 42, width - 54, 42)
    canvas.setFillColor(colors.HexColor("#536272"))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(54, 28, "The New York Knicks | Deep Learning Project 1")
    canvas.drawRightString(width - 54, 28, str(document.page))
    canvas.restoreState()


def write_pdf(pages: tuple[Page, ...], path: Path) -> None:
    styles = pdf_styles()
    story = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(escape(page.title), styles["cover" if index == 0 else "title"]))
        for block in page.blocks:
            if isinstance(block, Text):
                story.append(Paragraph(paragraph_markup(block.text), styles[block.style]))
            elif isinstance(block, ReportTable):
                caption = Paragraph(escape(block.caption), styles["caption"])
                caption.keepWithNext = True
                story.extend((caption, pdf_table(block, styles), Spacer(1, 8)))
            else:
                image = Image(str(block.path), width=block.width, height=block.height)
                image.hAlign = "CENTER"
                story.append(KeepTogether([
                    image, Spacer(1, 5), Paragraph(escape(block.caption), styles["caption"]),
                ]))
    document = SimpleDocTemplate(
        str(path), pagesize=letter, rightMargin=48, leftMargin=48,
        topMargin=48, bottomMargin=56,
        title=pages[0].title, author="Mohamed Awadalla",
        subject="Deep Learning Project 1, The New York Knicks",
        pageCompression=1,
    )
    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)


def write_markdown(pages: tuple[Page, ...], path: Path) -> None:
    lines = []
    for index, page in enumerate(pages):
        lines.extend((f"{'#' if index == 0 else '##'} {page.title}", ""))
        for block in page.blocks:
            if isinstance(block, Text):
                if block.style == "heading":
                    lines.extend((f"### {block.text}", ""))
                elif block.style == "hash":
                    lines.extend((f"`{block.text}`", ""))
                else:
                    lines.extend((block.text, ""))
            elif isinstance(block, ReportTable):
                lines.extend((block.caption, ""))
                lines.append("| " + " | ".join(block.headers) + " |")
                lines.append("| " + " | ".join("---" for _ in block.headers) + " |")
                for row in block.rows:
                    lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
                lines.append("")
            else:
                relative = block.path.relative_to(path.parent).as_posix()
                lines.extend((f"![{block.caption}]({relative})", "", block.caption, ""))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    evidence_directory = ROOT / "results"
    evidence = load_evidence(evidence_directory)
    config = json.loads((evidence_directory / "final-full-config.json").read_text())
    manifest = json.loads((evidence_directory / "final-full-manifest.json").read_text())
    summary = json.loads((evidence_directory / "summary.json").read_text())
    report_directory = ROOT / "report"
    figures = make_figures(evidence, report_directory / "figures")
    pages = make_pages(evidence, config, manifest, summary["repository"], figures)
    write_markdown(pages, report_directory / "project1.md")
    write_pdf(pages, report_directory / "project1.pdf")
    print("Wrote report/project1.md and report/project1.pdf.")
    print(f"Final student: {arch_name(evidence.final.arch)}, {evidence.final.params:,} trainable parameters.")
    print(f"Kaggle private test accuracy: {percent(evidence.final.private_accuracy)}.")
    print(f"Kaggle public test accuracy: {percent(evidence.final.public_accuracy)}.")
    print(f"Separate recipe validation accuracy: {percent(evidence.validation.accuracy)}.")


if __name__ == "__main__":
    main()
