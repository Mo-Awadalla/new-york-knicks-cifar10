"""Direct classifier-weight edits, selected on 2,000 images and audited on 3,000.

This is an exploratory experiment, not a submission pipeline. Both subsets come
from the existing validation split, which has already been used to choose recipes.
Only the tuning subset enters candidate selection; no competition test images are
read. All proposed edits start from the original checkpoint, never another trial.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import knicks_cifar as kc


SEED = 20261104
SPLIT_SALT = "knicks-weight-edit-v1"
CANDIDATE_COUNT = 260


def partition(ids, labels):
    """Fix 200 tuning and 300 audit examples per class, without model predictions."""
    tuning = []
    for label in range(10):
        rows = [i for i, value in enumerate(labels) if value == label]
        if len(rows) != 500:
            raise ValueError("requires the original stratified 5,000-image validation set")
        rows.sort(key=lambda i: hashlib.sha256(f"{SPLIT_SALT}:{ids[i]}".encode()).digest())
        tuning.extend(rows[:200])
    tuning = sorted(tuning)
    selected = set(tuning)
    return tuning, [i for i in range(len(ids)) if i not in selected]


def selected_coordinates(importance, top_count, random_count, rng):
    top = torch.argsort(importance, descending=True, stable=True)[:top_count].tolist()
    chosen = set(top)
    rest = [i for i in torch.randperm(len(importance), generator=rng).tolist() if i not in chosen]
    return top + rest[:random_count]


def proposals(original):
    """Fixed, label-independent candidate list; only head.weight is edited."""
    rng = torch.Generator().manual_seed(SEED)
    coordinates = selected_coordinates(original.flatten().abs(), 32, 32, rng)
    for index in coordinates:
        row, column = divmod(index, original.shape[1])
        for factor in (0.9, 1.1):
            changed = original.clone()
            changed[row, column] *= factor
            yield {"family": "single_weight", "row": row, "column": column, "factor": factor}, changed
    neurons = selected_coordinates(torch.linalg.vector_norm(original, dim=0), 16, 16, rng)
    for column in neurons:
        for factor in (0.0, 0.8, 1.2):
            changed = original.clone()
            changed[:, column] *= factor
            yield {"family": "neuron_gain", "column": column, "factor": factor}, changed
    smallest = torch.argsort(original.flatten().abs(), stable=True)
    for fraction in (0.01, 0.02, 0.05, 0.1):
        changed = original.clone()
        count = max(1, int(original.numel() * fraction))
        changed.view(-1)[smallest[:count]] = 0.0
        yield {"family": "magnitude_pruning", "fraction": fraction, "zeroed_weights": count}, changed
    rms = original.square().mean().sqrt()
    for relative_scale in (0.005, 0.01, 0.02, 0.05):
        for trial in range(8):
            changed = original + torch.randn(original.shape, generator=rng) * (rms * relative_scale)
            yield {"family": "random_perturbation", "relative_rms": relative_scale, "trial": trial}, changed


def metrics(logits, labels):
    return {"rows": len(labels), "correct": int((logits.argmax(1) == labels).sum()),
            "loss": F.cross_entropy(logits, labels).item()}


def search(features, labels, weight, bias, out):
    """Audit features and labels are deliberately not arguments to this function."""
    base = metrics(F.linear(features, weight, bias), labels)
    best = {"index": -1, "edit": {"family": "unchanged"}, **base}
    winner = weight.clone()
    with (out / "tuning_trials.jsonl").open("x") as log:
        log.write(json.dumps(best) + "\n")
        count = 0
        for index, (spec, candidate) in enumerate(proposals(weight)):
            result = metrics(F.linear(features, candidate, bias), labels)
            row = {"index": index, "edit": spec, **result}
            log.write(json.dumps(row) + "\n")
            # A loss-only change does not qualify as an accuracy improvement.
            if result["correct"] > base["correct"] and (-row["correct"], row["loss"]) < (-best["correct"], best["loss"]):
                best, winner = row, candidate
            count += 1
        if count != CANDIDATE_COUNT:
            raise ValueError(f"candidate count {count} differs from the declared plan")
    return best, winner, base


def features_and_logits(model, images, mean, std, batch):
    features, logits = [], []
    handle = model.head.register_forward_pre_hook(lambda module, args: features.append(args[0].detach().cpu()))
    try:
        with torch.inference_mode():
            for start in range(0, len(images), batch):
                logits.append(model(kc.normalize(images[start:start + batch], mean, std)).detach().cpu())
    finally:
        handle.remove()
    return torch.cat(features), torch.cat(logits)


def paired_report(baseline, candidate, labels):
    old, new = baseline.argmax(1) == labels, candidate.argmax(1) == labels
    difference = new.double() - old.double()
    n = len(labels)
    standard_error = difference.std(unbiased=True).item() / math.sqrt(n)
    delta = difference.mean().item()
    return {"rows": n, "baseline_correct": int(old.sum()), "edited_correct": int(new.sum()),
            "fixed": int((new & ~old).sum()), "broken": int((old & ~new).sum()),
            "net_images": int(new.sum() - old.sum()), "delta_percentage_points": 100 * delta,
            "paired_normal_95_interval_points": [100 * (delta - 1.96 * standard_error), 100 * (delta + 1.96 * standard_error)],
            "baseline_loss": F.cross_entropy(baseline, labels).item(),
            "edited_loss": F.cross_entropy(candidate, labels).item()}


def run(args):
    torch.set_num_threads(4)
    started = time.monotonic()
    source = args.source / "checkpoint.pt"
    metadata = json.loads((args.source / "manifest.json").read_text())
    source_hash = kc.sha256_file(source)
    if metadata["status"] != "complete" or source_hash != metadata["checkpoint_sha256"]:
        raise ValueError("requires a complete, hash-verified source run")
    checkpoint, model, mean, std = kc.checkpoint_model(source, kc.device_for(args.device))
    cfg = kc.RunConfig(**checkpoint["config"])
    cfg.validate()
    if cfg.split != "val" or checkpoint["completed_epochs"] != cfg.epochs or checkpoint["params"] > kc.MAX_PARAMS:
        raise ValueError("requires a completed under-cap 45k model, never a full-data model")
    if any(t["split"] != "val" for t in checkpoint["provenance"].get("distillation_teachers", [])):
        raise ValueError("source teachers must not have trained on validation rows")
    manifest, _, _ = kc.official_manifests(args.data)
    split = kc.stratified_split(manifest, "val")
    ids = [manifest.ids[i] for i in split.val_rows]
    saved_split = json.loads((args.source / "split.json").read_text())
    if ids != saved_split["val_ids"] or checkpoint["split_sha256"] != split.sha256:
        raise ValueError("source validation IDs differ from the canonical split")
    if checkpoint["provenance"]["csv_sha256"] != kc.input_hashes(args.data):
        raise ValueError("official CSV manifests differ from the source run")
    labels = torch.tensor([manifest.labels[i] for i in split.val_rows], dtype=torch.long)
    tuning, audit = partition(ids, labels.tolist())
    args.out.mkdir(parents=True, exist_ok=False)
    kc.json_write(args.out / "plan.json", {
        "source_sha256": source_hash, "script_sha256": kc.sha256_file(Path(__file__)),
        "canonical_source_sha256": kc.source_sha256(), "seed": SEED, "partition_salt": SPLIT_SALT,
        "tuning_ids": [ids[i] for i in tuning], "audit_ids": [ids[i] for i in audit],
        "candidate_count": CANDIDATE_COUNT, "edited_tensor": "head.weight", "selection": "tuning accuracy, loss tiebreak",
        "promotion_threshold": "at least +15 net audit images (0.5 points); otherwise no adoption",
        "claim_limit": "Historical validation reuse; audit is hidden only from this new search; no public-score inference.",
        "device": args.device, "torch": str(torch.__version__), "submission_eligible": False,
    })
    print("Plan saved; decoding the 5,000 official held-out training images", flush=True)
    images = kc.decode(manifest, split.val_rows, kc.image_index(args.data), kc.device_for(args.device))
    features, baseline_logits = features_and_logits(model, images, mean, std, args.batch)
    expected = np.load(args.source / "val_pred.npy", allow_pickle=False)
    if not np.array_equal(baseline_logits.argmax(1).numpy(), expected):
        raise ValueError("baseline predictions differ from the saved GPU predictions; do not compare devices without resolving this")
    baseline_correct = int((baseline_logits.argmax(1) == labels).sum())
    if baseline_correct != metadata["final_val_correct"]:
        raise ValueError("baseline accuracy differs from the source manifest")
    weight, bias = model.head.weight.detach().cpu().clone(), model.head.bias.detach().cpu().clone()
    cached_logits = F.linear(features, weight, bias)
    if not torch.equal(cached_logits.argmax(1), baseline_logits.argmax(1)):
        raise ValueError("cached classifier evaluation changed baseline predictions")
    print(f"Baseline replay: {baseline_correct}/5000; all saved predictions match", flush=True)
    np.savez_compressed(args.out / "features.npz", features=features.numpy(), labels=labels.numpy(), ids=np.asarray(ids))
    best, winner, base_tuning = search(features[tuning], labels[tuning], weight, bias, args.out)
    # Lock the edit before reading any audit accuracy. This file is also enough to replay the actual edit.
    kc.json_write(args.out / "selected_edit.json", best)
    torch.save({"source_sha256": source_hash, "head.weight": winner, "selection": best, "submission_eligible": False}, args.out / "selected_head.pt")
    print(json.dumps({"selected_edit": best["edit"], "tuning_before": base_tuning["correct"], "tuning_after": best["correct"]}), flush=True)
    # Audit via the actual edited network, not just cached head arithmetic.
    with torch.no_grad():
        model.head.weight.copy_(winner.to(model.head.weight.device))
    actual_predictions, _, actual_correct = kc.evaluate(model, images, labels.to(images.device), tuple(range(len(images))), mean, std, args.batch)
    edited_logits = F.linear(features, winner, bias)
    if not np.array_equal(np.asarray(actual_predictions), edited_logits.argmax(1).numpy()):
        raise ValueError("edited-network inference differs from cached head predictions")
    delta = winner - weight
    report = {
        "selected_edit": best["edit"], "candidate_count": CANDIDATE_COUNT,
        "tuning": paired_report(cached_logits[tuning], edited_logits[tuning], labels[tuning]),
        "audit": paired_report(cached_logits[audit], edited_logits[audit], labels[audit]),
        "all_validation_descriptive_only": paired_report(cached_logits, edited_logits, labels),
        "modified_weights": int((delta != 0).sum()), "relative_head_l2_change": (delta.norm() / weight.norm()).item(),
        "replay": {"baseline_matches_saved_predictions": True, "edited_network_matches_cached_predictions": True,
                   "edited_network_correct": actual_correct},
        "submission_eligible": False,
        "claim_limit": "Only one selected edit audited. The historical validation set was already used for recipe selection.",
    }
    report["meets_audit_margin"] = report["audit"]["net_images"] >= 15
    report["elapsed_seconds"] = time.monotonic() - started
    if kc.sha256_file(source) != source_hash:
        raise ValueError("source checkpoint changed during the experiment")
    report["source_checkpoint_unchanged"] = True
    np.save(args.out / "edited_val_pred.npy", np.asarray(actual_predictions, dtype=np.int8))
    kc.json_write(args.out / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    run(args)


if __name__ == "__main__":
    main()
