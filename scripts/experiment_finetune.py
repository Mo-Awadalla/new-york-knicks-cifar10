"""Compare two fixed, training-data-only fine-tunes of a completed 45k model.

Classifier-only and final-block recipes share initialization, data order,
augmentation, optimizer, and five-epoch schedule. BatchNorm statistics are frozen.
No checkpoint is picked by intermediate validation accuracy. Selection uses the
existing 2k tuning partition; the previously used 3k audit is descriptive evidence,
not a pristine test set. Experiment checkpoints cannot create submissions.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import knicks_cifar as kc
from knicks_methods import Rngs, TrainingMethod
from experiment_weight_edits import paired_report, partition


SCOPES = ("classifier", "last_block")
EPOCHS = 5
SEED = 1
LR = 0.001
BATCH = 128


def configure_scope(model, scope):
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope}")
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    if scope == "last_block":
        model.blocks[-1].requires_grad_(True)
        model.bn.requires_grad_(True)
    # eval() freezes BatchNorm buffers, not gradients of the selected parameters.
    model.eval()
    return {name for name, value in model.named_parameters() if value.requires_grad}


def logits_on_rows(model, images, rows, mean, std):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            index = torch.tensor(rows[start:start + BATCH], device=images.device)
            chunks.append(model(kc.normalize(images[index], mean, std)).cpu())
    return torch.cat(chunks)


def tune(model, scope, images, labels, train_rows, mean, std, epochs, out):
    """Only training rows are available to optimization; validation lives outside."""
    out.mkdir(parents=True, exist_ok=False)
    allowed = configure_scope(model, scope)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005, nesterov=True)
    order_rng = torch.Generator().manual_seed(SEED * 16 + 1)
    rngs = Rngs(torch.Generator().manual_seed(SEED * 16 + 2),
                torch.Generator().manual_seed(SEED * 16 + 3), np.random.default_rng(SEED * 16 + 6))
    method = TrainingMethod("basic", "sgd", model=model, optimizer=optimizer,
                            mean=mean, std=std, rngs=rngs, amp=images.device.type == "cuda")
    rows = torch.tensor(train_rows, dtype=torch.long, device=images.device)
    steps_per_epoch = math.ceil(len(rows) / BATCH)
    total_steps = steps_per_epoch * epochs
    updates, skips, step = 0, 0, 0
    started = time.monotonic()
    with (out / "metrics.jsonl").open("x") as log:
        for epoch in range(epochs):
            kc.synchronize(images.device)
            epoch_started = time.monotonic()
            permutation = torch.randperm(len(rows), generator=order_rng).to(images.device)
            loss, correct, seen = 0.0, 0, 0
            for start in range(0, len(rows), BATCH):
                index = rows[permutation[start:start + BATCH]]
                rate = LR * 0.5 * (1 + math.cos(math.pi * (step + 1) / total_steps))
                for group in optimizer.param_groups:
                    group["lr"] = rate
                outcome = method.step(images[index], labels[index])
                updates += outcome.update == "applied"
                skips += outcome.update != "applied"
                loss += outcome.loss * outcome.rows
                correct += outcome.correct
                seen += outcome.rows
                step += 1
            kc.synchronize(images.device)
            result = {"scope": scope, "epoch": epoch + 1, "rows": seen, "train_loss": loss / seen,
                      "train_correct": correct, "lr": rate, "optimizer_updates": updates,
                      "skipped_updates": skips, "seconds": time.monotonic() - epoch_started}
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps(result), flush=True)
    final = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    changed = {name for name in final if not torch.equal(final[name], initial[name])}
    if not changed or not changed.issubset(allowed):
        raise ValueError(f"updates escaped the allowed parameter set: {sorted(changed - allowed)}")
    displacement = sum((final[name] - initial[name]).square().sum().item() for name in allowed)
    norm = sum(initial[name].square().sum().item() for name in allowed)
    return final, {"scope": scope, "epochs": epochs, "optimizer_updates": updates,
                   "skipped_updates": skips, "train_rows": len(train_rows),
                   "trainable_parameters": sum(p.numel() for p in params),
                   "changed_tensors": sorted(changed), "batchnorm_buffers_unchanged": True,
                   "frozen_tensors_unchanged": True, "relative_l2_update": math.sqrt(displacement / norm),
                   "elapsed_seconds": time.monotonic() - started}


def run(args):
    torch.set_num_threads(4)
    source = args.source / "checkpoint.pt"
    source_hash = kc.sha256_file(source)
    saved = json.loads((args.source / "manifest.json").read_text())
    if saved["status"] != "complete" or saved["checkpoint_sha256"] != source_hash:
        raise ValueError("requires a complete, hash-verified source")
    device = kc.device_for(args.device)
    checkpoint, original, mean, std = kc.checkpoint_model(source, device)
    config = kc.RunConfig(**checkpoint["config"])
    config.validate()
    if config.split != "val" or checkpoint["completed_epochs"] != config.epochs or checkpoint["params"] > kc.MAX_PARAMS:
        raise ValueError("requires a completed under-cap 45k source, not a full-data model")
    if any(t["split"] != "val" for t in checkpoint["provenance"].get("distillation_teachers", [])):
        raise ValueError("source teachers must exclude validation rows")
    manifest, _, _ = kc.official_manifests(args.data)
    split = kc.stratified_split(manifest, "val")
    ids = [manifest.ids[i] for i in split.val_rows]
    if checkpoint["split_sha256"] != split.sha256 or json.loads((args.source / "split.json").read_text())["val_ids"] != ids:
        raise ValueError("source validation split differs from the canonical split")
    if checkpoint["provenance"]["csv_sha256"] != kc.input_hashes(args.data):
        raise ValueError("input CSVs differ from source training")
    val_labels = torch.tensor([manifest.labels[i] for i in split.val_rows], dtype=torch.long)
    tuning, audit = partition(ids, val_labels.tolist())
    args.out.mkdir(parents=True, exist_ok=False)
    kc.json_write(args.out / "plan.json", {
        "source_checkpoint_sha256": source_hash, "script_sha256": kc.sha256_file(Path(__file__)),
        "canonical_source_sha256": kc.source_sha256(), "scopes": list(SCOPES), "epochs": EPOCHS,
        "seed": SEED, "batch": BATCH, "lr": LR, "schedule": "cosine_to_zero_no_warmup",
        "optimizer": "SGD_Nesterov", "momentum": 0.9, "weight_decay": 0.0005,
        "loss": "cross_entropy_on_training_labels", "augmentation": "crop_flip",
        "batchnorm_buffers": "frozen", "intermediate_validation": False,
        "tuning_ids": [ids[i] for i in tuning], "audit_ids": [ids[i] for i in audit],
        "selection": "final tuning accuracy then tuning loss; unchanged baseline is eligible",
        "promotion": "tuning improvement and at least 15 net additional correct on 3000 audit images",
        "claim_limit": "Both validation partitions have historical reuse; no pristine test or seed-stability claim.",
        "device": args.device, "torch": str(torch.__version__), "submission_eligible": False,
    })
    print("Plan saved; decoding official training data", flush=True)
    images = kc.decode(manifest, tuple(range(len(manifest.ids))), kc.image_index(args.data), device)
    labels = torch.tensor(manifest.labels, dtype=torch.long, device=device)
    baseline = logits_on_rows(original, images, split.val_rows, mean, std)
    if not np.array_equal(baseline.argmax(1).numpy(), np.load(args.source / "val_pred.npy", allow_pickle=False)):
        raise ValueError("baseline does not reproduce the saved GPU predictions")
    print(f"Baseline: {int((baseline.argmax(1) == val_labels).sum())}/5000; exact saved predictions", flush=True)
    outputs, training_records = {}, {}
    best = {"scope": "unchanged", "correct": int((baseline[tuning].argmax(1) == val_labels[tuning]).sum()),
            "loss": torch.nn.functional.cross_entropy(baseline[tuning], val_labels[tuning]).item()}
    for scope in SCOPES:
        model = copy.deepcopy(original)
        state, record = tune(model, scope, images, labels, split.train_rows, mean, std, EPOCHS, args.out / scope)
        trained = dict(checkpoint, state_dict=state, kind="fine_tune_experiment",
                       fine_tuning={**record, "source_checkpoint_sha256": source_hash}, submission_eligible=False)
        torch.save(trained, args.out / scope / "checkpoint.pt")
        _, restored, saved_mean, saved_std = kc.checkpoint_model(args.out / scope / "checkpoint.pt", device)
        model_logits = logits_on_rows(model, images, split.val_rows, mean, std)
        logits = logits_on_rows(restored, images, split.val_rows, saved_mean, saved_std)
        if not torch.equal(model_logits, logits):
            raise ValueError("fine-tuned checkpoint reload changed evaluation logits")
        outputs[scope], training_records[scope] = logits, record
        score = {"scope": scope, "correct": int((logits[tuning].argmax(1) == val_labels[tuning]).sum()),
                 "loss": torch.nn.functional.cross_entropy(logits[tuning], val_labels[tuning]).item()}
        base_correct = int((baseline[tuning].argmax(1) == val_labels[tuning]).sum())
        if score["correct"] > base_correct and (-score["correct"], score["loss"]) < (-best["correct"], best["loss"]):
            best = score
        kc.json_write(args.out / scope / "training.json", record)
        np.save(args.out / scope / "val_pred.npy", logits.argmax(1).numpy().astype(np.int8))
        print(json.dumps({"final_tuning": score, "checkpoint_reload_logits_exact": True}), flush=True)
        del model, restored
    # Lock selection before computing audit results for either fixed recipe.
    kc.json_write(args.out / "selection.json", best)
    comparison = {}
    for scope, logits in outputs.items():
        comparison[scope] = {"training": training_records[scope],
                             "tuning": paired_report(baseline[tuning], logits[tuning], val_labels[tuning]),
                             "audit": paired_report(baseline[audit], logits[audit], val_labels[audit]),
                             "all_validation_descriptive_only": paired_report(baseline, logits, val_labels),
                             "checkpoint_reload_logits_exact": True}
    if kc.sha256_file(source) != source_hash:
        raise ValueError("source checkpoint changed")
    qualifies = best["scope"] != "unchanged" and comparison[best["scope"]]["audit"]["net_images"] >= 15
    result = {"selected_on_tuning": best, "comparisons": comparison, "meets_audit_margin": qualifies,
              "source_checkpoint_unchanged": True, "submission_eligible": False,
              "claim_limit": "Two fixed recipes, historically reused validation data, one training seed; not a leaderboard forecast."}
    kc.json_write(args.out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="CPU or CUDA; MPS frozen-BatchNorm backward crashes on the tested PyTorch build")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
