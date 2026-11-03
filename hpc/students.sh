#!/usr/bin/env bash
# Distill WRN-22-4 CutMix students from a chosen teacher set at two epoch horizons.
# Usage: hpc/students.sh SET_NAME TEACHER_STEM...   (stems name teachers/STEM-45k.pt and teachers/STEM-full.pt)
# GPU 0 trains the 200-epoch pair (45k validation run, then full data); GPU 1 the 400-epoch pair.
set -euo pipefail
K=/scratch/$USER/knicks
cd "$K"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set_name=$1; shift
mkdir -p runs logs configs

for epochs in 200 400; do
  for split in val full; do
    .venv/bin/python - "$set_name" "$epochs" "$split" "$@" <<'EOF'
import hashlib, json, sys
set_name, epochs, split, *stems = sys.argv[1:]
suffix = "45k" if split == "val" else "full"
teachers = [[f"{s}-{suffix}.pt", hashlib.sha256(open(f"teachers/{s}-{suffix}.pt", "rb").read()).hexdigest()] for s in stems]
cfg = {"arch": "wrn22_4", "aug": "cutmix", "split": "val" if split == "val" else "all", "epochs": int(epochs), "seed": 1,
       "batch": 128, "lr": 0.1, "momentum": 0.9, "wd": 0.0005, "warmup": 5, "kd_alpha": 0.9, "kd_temperature": 4.0, "teachers": teachers}
open(f"configs/kd-{set_name}-e{epochs}-{split}.json", "w").write(json.dumps(cfg, indent=2) + "\n")
EOF
  done
done

queue() {
  local gpu=$1; shift
  for name in "$@"; do
    local flag=()
    [[ $name == *-val ]] && flag=(--calibrate)
    rm -rf "runs/$name"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python src/knicks_cifar.py train \
      --config "configs/$name.json" --data data --out "runs/$name" --teachers teachers \
      --device cuda --max-seconds 36000 "${flag[@]}" > "logs/$name.log" 2>&1 || echo "FAILED $name" >> logs/failures.txt
  done
}

queue 0 "kd-$set_name-e200-val" "kd-$set_name-e200-full" &
queue 1 "kd-$set_name-e400-val" "kd-$set_name-e400-full" &
wait
echo "STUDENTS $set_name DONE" >> logs/status.txt
