"""Write a WRN-22-4 CutMix distillation config pinned to teacher checkpoint hashes.

Usage: student_config.py NAME EPOCHS SPLIT STEM...  (SPLIT is val or full; teachers/STEM-45k.pt or teachers/STEM-full.pt)
"""
import hashlib
import json
import sys

name, epochs, split, *stems = sys.argv[1:]
if split not in {"val", "full"} or not stems:
    raise SystemExit(__doc__)
suffix = "45k" if split == "val" else "full"
teachers = [[f"{stem}-{suffix}.pt", hashlib.sha256(open(f"teachers/{stem}-{suffix}.pt", "rb").read()).hexdigest()] for stem in stems]
config = {"arch": "wrn22_4", "aug": "cutmix", "split": "val" if split == "val" else "all", "epochs": int(epochs), "seed": 1,
          "batch": 128, "lr": 0.1, "momentum": 0.9, "wd": 0.0005, "warmup": 5, "kd_alpha": 0.9, "kd_temperature": 4.0, "teachers": teachers}
with open(f"configs/{name}.json", "w") as f:
    f.write(json.dumps(config, indent=2) + "\n")
print(f"configs/{name}.json", [t[0] for t in teachers])
