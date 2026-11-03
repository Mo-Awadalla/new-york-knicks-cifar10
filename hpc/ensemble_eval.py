"""Soft-ensemble validation accuracy of candidate 45k teacher sets (no test images used)."""
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import knicks_cifar as kc

root = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
device = torch.device("cuda")
manifest, _, _ = kc.official_manifests(root)
split = kc.stratified_split(manifest, "val")
images = kc.decode(manifest, split.val_rows, kc.image_index(root), device)
labels = torch.tensor([manifest.labels[i] for i in split.val_rows], device=device)
probs = {}
for path in paths:
    checkpoint, model, mean, std = kc.checkpoint_model(path, device)
    if checkpoint["config"]["split"] != "val" or checkpoint["split_sha256"] != split.sha256:
        raise ValueError(f"{path} saw validation rows")
    with torch.inference_mode():
        logits = torch.cat([model(kc.normalize(images[i:i + 500], mean, std)).float() for i in range(0, len(images), 500)])
    probs[path.stem] = F.softmax(logits, dim=1)
    print(json.dumps({"model": path.stem, "arch": checkpoint["config"]["arch"], "correct": int((logits.argmax(1) == labels).sum())}), flush=True)
for size in range(2, len(probs) + 1):
    for names in itertools.combinations(probs, size):
        correct = int((sum(probs[n] for n in names).argmax(1) == labels).sum())
        print(json.dumps({"ensemble": names, "correct": correct}), flush=True)
