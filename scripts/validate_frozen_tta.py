"""Validate the single frozen AugMix-32 wrapper on the existing 5,000-row split.

This is a validation experiment, not a submission pipeline. Both partitions have
been used historically; this validation checkpoint and its output are ineligible.
"""
import argparse
from dataclasses import asdict
import json
import math
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import PIL
import torch
import torchvision

import knicks_cifar as kc
import knicks_tta as tta
from experiment_weight_edits import SPLIT_SALT, partition


SOURCE_CHECKPOINT_SHA256 = "42fa268b43159b51eb9a936f9d00e91a5cfdd0194e7abb884bb79333658f9719"
BASELINE_CORRECT = 4872
CLAIM_LIMIT = (
    "One fixed policy on one fitted model; both validation partitions have historical "
    "reuse. Paired normal intervals are descriptive, not independent test-set or "
    "seed-stability evidence, and do not predict public score."
)


def source_and_split(source, data, device):
    checkpoint_path = source / "checkpoint.pt"
    source_files = ("checkpoint.pt", "manifest.json", "config.json", "split.json", "val_pred.npy")
    hashes = {name: kc.sha256_file(source / name) for name in source_files}
    metadata = json.loads((source / "manifest.json").read_text())
    saved_config = json.loads((source / "config.json").read_text())
    saved_split = json.loads((source / "split.json").read_text())
    if (hashes["checkpoint.pt"] != SOURCE_CHECKPOINT_SHA256
            or metadata.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or metadata.get("status") != "complete"):
        raise ValueError("requires the fixed complete, hash-verified source checkpoint")

    checkpoint, model, mean, std = kc.checkpoint_model(checkpoint_path, device)
    cfg = kc.RunConfig(**checkpoint["config"])
    cfg.validate()
    if (asdict(cfg) != asdict(kc.RunConfig(**saved_config))
            or cfg.split != "val" or cfg.epochs != 200
            or checkpoint["completed_epochs"] != 200 or metadata.get("completed_epochs") != 200
            or checkpoint["kind"] != "calibrate" or metadata.get("kind") != "calibrate"
            or metadata.get("candidate_eligible") is not False
            or metadata.get("train_rows") != 45000 or metadata.get("val_rows") != 5000
            or checkpoint["params"] != kc.trainable_params(model)
            or checkpoint["params"] != metadata.get("params")
            or checkpoint["params"] > kc.MAX_PARAMS):
        raise ValueError("source must be the completed, under-cap 45k validation checkpoint")

    provenance = checkpoint["provenance"]
    for key in ("source_sha256", "source_sha256_by_file", "csv_sha256", "torch", "cuda",
                "gpu", "deterministic_algorithms", "numpy", "pillow", "torchvision"):
        if key not in provenance or provenance[key] != metadata.get(key):
            raise ValueError(f"source provenance mismatch: {key}")
    if kc.source_digest(provenance["source_sha256_by_file"]) != provenance["source_sha256"]:
        raise ValueError("source provenance digest is inconsistent")
    teachers = provenance.get("distillation_teachers", [])
    if (teachers != metadata.get("distillation_teachers", [])
            or any(teacher.get("split") != "val" for teacher in teachers)
            or {(t.get("file"), t.get("sha256")) for t in teachers} != set(cfg.teachers)
            or len(teachers) != len(cfg.teachers)):
        raise ValueError("teacher provenance is missing, inconsistent, or leaks validation rows")
    if (checkpoint["mean"] != metadata.get("mean") or checkpoint["std"] != metadata.get("std")
            or mean.shape != (3,) or std.shape != (3,)
            or not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item()
            or not (std > 0).all().item()):
        raise ValueError("invalid or inconsistent checkpoint normalization")

    manifest, _, _ = kc.official_manifests(data)
    split = kc.stratified_split(manifest, "val")
    ids = [manifest.ids[i] for i in split.val_rows]
    if (saved_split.get("train_ids") != [manifest.ids[i] for i in split.train_rows]
            or saved_split.get("val_ids") != ids
            or saved_split.get("validation_sha256") != split.sha256
            or checkpoint["split_sha256"] != split.sha256
            or metadata.get("split_sha256") != split.sha256):
        raise ValueError("source split membership, ordering, or fingerprint differs from canonical")
    csv_hashes = kc.input_hashes(data)
    if csv_hashes != provenance["csv_sha256"]:
        raise ValueError("official CSV manifest hashes differ from source training")
    labels = torch.tensor([manifest.labels[i] for i in split.val_rows], dtype=torch.long)
    expected = np.load(source / "val_pred.npy", allow_pickle=False)
    if (expected.shape != (5000,) or not np.issubdtype(expected.dtype, np.integer)
            or not np.all((expected >= 0) & (expected <= 9))
            or int((expected == labels.numpy()).sum()) != BASELINE_CORRECT
            or metadata.get("final_val_correct") != BASELINE_CORRECT
            or metadata.get("final_val_accuracy") != BASELINE_CORRECT / 5000):
        raise ValueError("invalid saved validation predictions or source baseline metric")
    return checkpoint, model, mean, std, manifest, split, ids, labels, expected, hashes, csv_hashes


def verify_image_hashes(data, manifest, rows, index):
    """Verify the validation PNGs when the private export's digest list is present."""
    path = data / "training-image-sha256.json"
    if not path.is_file():
        return {"available": False, "verified_rows": 0}
    hashes = json.loads(path.read_text())
    if not isinstance(hashes, dict):
        raise ValueError("invalid training image digest list")
    for row in rows:
        image = index[manifest.files[row]]
        key = image.relative_to(data).as_posix()
        if hashes.get(key) != kc.sha256_file(image):
            raise ValueError(f"validation image digest mismatch: {key}")
    return {"available": True, "sha256": kc.sha256_file(path), "verified_rows": len(rows)}


def model_tensors(model):
    tensors = {f"state:{name}": value for name, value in model.state_dict().items()}
    tensors.update({f"parameter:{name}": value for name, value in model.named_parameters()})
    tensors.update({f"buffer:{name}": value for name, value in model.named_buffers()})
    return tensors


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model_tensors(model).items()}


def state_proof(before, model):
    after = model_tensors(model)
    changed = sorted(name for name in set(before) | set(after)
                     if name not in before or name not in after
                     or not torch.equal(before[name], after[name].detach().cpu()))
    if changed:
        raise ValueError(f"model parameters or buffers changed: {changed}")
    digests = {name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
               for name, value in before.items()}
    return {"all_state_dict_tensors_unchanged": True, "all_parameters_and_buffers_unchanged": True,
            "tensor_count": len(before), "tensor_sha256": digests,
            "parameter_names": [name for name, _ in model.named_parameters()],
            "buffer_names": [name for name, _ in model.named_buffers()],
            "changed_tensors": changed,
            "all_parameter_gradients_none": all(p.grad is None for p in model.parameters())}


def baseline_logits(model, images, mean, std, batch):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            chunk = images[start:start + batch].to(mean.device)
            result.append(model(kc.normalize(chunk, mean, std)).float().cpu())
    return torch.cat(result)


def paired_metrics(baseline, candidate, labels, rows):
    rows = torch.as_tensor(rows, dtype=torch.long)
    labels = labels[rows]
    baseline, candidate = baseline[rows], candidate[rows]
    old = baseline.argmax(1) == labels
    new = candidate.argmax(1) == labels
    difference = new.double() - old.double()
    delta = difference.mean().item()
    error = difference.std(unbiased=True).item() / math.sqrt(len(rows))
    def nll(probabilities):
        return -probabilities.double()[torch.arange(len(labels)), labels].log().mean().item()
    return {
        "rows": len(rows), "baseline_correct": int(old.sum()), "tta_correct": int(new.sum()),
        "baseline_accuracy": old.double().mean().item(), "tta_accuracy": new.double().mean().item(),
        "fixed": int((new & ~old).sum()), "broken": int((old & ~new).sum()),
        "net_images": int(new.sum() - old.sum()), "delta_percentage_points": 100 * delta,
        "paired_normal_95_interval_points": [100 * (delta - 1.96 * error), 100 * (delta + 1.96 * error)],
        "baseline_nll": nll(baseline), "tta_nll": nll(candidate),
    }


def run(args):
    torch.set_num_threads(4)
    started = time.monotonic()
    device = kc.device_for(args.device)
    if args.out.resolve().is_relative_to(args.source.resolve()) or args.out.resolve().is_relative_to(args.data.resolve()):
        raise ValueError("output must not be inside the source or data directories")
    (checkpoint, model, mean, std, manifest, split, ids, labels, expected,
     source_hashes, csv_hashes) = source_and_split(args.source, args.data, device)
    tuning, audit = partition(ids, labels.tolist())
    index = kc.image_index(args.data)
    image_proof = verify_image_hashes(args.data, manifest, split.val_rows, index)
    args.out.mkdir(parents=True, exist_ok=False)
    kc.json_write(args.out / "plan.json", {
        "source_directory": str(args.source.resolve()), "data_directory": str(args.data.resolve()),
        "source_file_sha256": source_hashes, "source_training_provenance": checkpoint["provenance"],
        "csv_sha256": csv_hashes, "split_sha256": split.sha256, "validation_image_hashes": image_proof,
        "validator_sha256": kc.sha256_file(Path(__file__)),
        "partition_module_sha256": kc.sha256_file(Path(__file__).with_name("experiment_weight_edits.py")),
        "canonical_source_sha256_by_file": kc.source_hashes(),
        "tta_module_sha256": kc.sha256_file(Path(tta.__file__)), "policy": tta.policy_record(),
        "partition_salt": SPLIT_SALT, "tuning_ids": [ids[i] for i in tuning],
        "audit_ids": [ids[i] for i in audit], "baseline_required_correct": BASELINE_CORRECT,
        "decision_rule": "qualify iff tuning net_images > 0 AND audit net_images > 0; ties fail; no NLL or margin override",
        "claim_limit": CLAIM_LIMIT, "candidate_eligible": False,
        "device": str(device), "batch": args.batch, "python": platform.python_version(),
        "numpy": np.__version__, "pillow": PIL.__version__, "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
    })
    print("Plan saved; decoding held-out official training images", flush=True)
    images = kc.decode(manifest, split.val_rows, index, torch.device("cpu"))
    initial_state = snapshot(model)
    logits = baseline_logits(model, images, mean, std, args.batch)
    baseline_pred = logits.argmax(1).numpy()
    correct = int((baseline_pred == labels.numpy()).sum())
    if not np.array_equal(baseline_pred, expected) or correct != BASELINE_CORRECT:
        raise ValueError("baseline replay failed: predictions must match all saved rows and 4872 correct")
    np.save(args.out / "baseline_logits.npy", logits.numpy())
    np.save(args.out / "baseline_pred.npy", baseline_pred.astype(np.int8))
    kc.json_write(args.out / "baseline_replay.json", {
        "rows": len(ids), "correct": correct, "all_saved_predictions_match": True,
        "saved_predictions_sha256": source_hashes["val_pred.npy"],
    })
    print("Baseline replay: 4872/5000; all saved predictions match", flush=True)

    # The fixed policy is run once across all 5,000 images before any TTA metrics.
    probabilities = tta.augmix32_probabilities(model, images, ids, mean, std, batch_size=args.batch)
    if (probabilities.shape != (len(ids), 10) or probabilities.device.type != "cpu"
            or probabilities.dtype != torch.float32 or not torch.isfinite(probabilities).all().item()
            or (probabilities < 0).any().item()
            or not torch.allclose(probabilities.sum(1), torch.ones(len(ids)), atol=1e-5, rtol=1e-5)):
        raise ValueError("invalid AugMix probability output")
    np.save(args.out / "tta_probabilities.npy", probabilities.numpy())
    np.save(args.out / "tta_pred.npy", probabilities.argmax(1).numpy().astype(np.int8))
    proof = state_proof(initial_state, model)
    if not proof["all_parameter_gradients_none"]:
        raise ValueError("unexpected model gradients")
    after_hashes = {name: kc.sha256_file(args.source / name) for name in source_hashes}
    if after_hashes != source_hashes:
        raise ValueError("source files changed during validation")
    proof.update({"source_file_sha256_before": source_hashes, "source_file_sha256_after": after_hashes,
                  "source_checkpoint_unchanged": True})
    kc.json_write(args.out / "frozen_state_proof.json", proof)

    baseline_prob = torch.softmax(logits, dim=1)
    reports = {"tuning": paired_metrics(baseline_prob, probabilities, labels, tuning),
               "audit": paired_metrics(baseline_prob, probabilities, labels, audit),
               "all_validation_descriptive_only": paired_metrics(baseline_prob, probabilities, labels, range(len(ids)))}
    qualifies = reports["tuning"]["net_images"] > 0 and reports["audit"]["net_images"] > 0
    result = {"policy": tta.policy_record(), "comparisons": reports,
              "qualifies": qualifies, "decision": "qualify" if qualifies else "fail",
              "candidate_eligible": False, "claim_limit": CLAIM_LIMIT,
              "baseline_replay_matches_saved_predictions": True,
              "source_checkpoint_unchanged": True, "all_model_state_unchanged": True,
              "elapsed_seconds": time.monotonic() - started}
    kc.json_write(args.out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


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
