#!/usr/bin/env bash
# Train one distillation student on one GPU. Usage: hpc/run_student.sh GPU CONFIG_NAME
# CONFIG_NAME ending in -val runs on the 45k split and is marked ineligible; others train on all rows.
set -euo pipefail
K=/scratch/$USER/knicks
cd "$K"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
gpu=$1 name=$2
flag=()
[[ $name == *-val ]] && flag=(--calibrate)
rm -rf "runs/$name"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python src/knicks_cifar.py train \
  --config "configs/$name.json" --data data --out "runs/$name" --teachers teachers \
  --device cuda --max-seconds 36000 "${flag[@]}" > "logs/$name.log" 2>&1 \
  && echo "DONE $name" >> logs/status.txt || echo "FAILED $name" >> logs/failures.txt
