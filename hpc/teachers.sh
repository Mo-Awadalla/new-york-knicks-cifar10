#!/usr/bin/env bash
# Train the six WRN-28-10 teachers on a two-A100 node, one queue per GPU.
# GPU 0 runs four SGD recipes; GPU 1 runs the two SAM recipes, which cost about twice as much per epoch.
set -euo pipefail
K=/scratch/$USER/knicks
cd "$K"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p runs logs

queue() {
  local gpu=$1; shift
  for name in "$@"; do
    if [[ -f "runs/$name/manifest.json" ]] && grep -q '"status": "complete"' "runs/$name/manifest.json"; then
      continue
    fi
    rm -rf "runs/$name"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python src/knicks_cifar.py train \
      --config "configs/$name.json" --data data --out "runs/$name" \
      --device cuda --max-seconds 25200 --calibrate > "logs/$name.log" 2>&1 || echo "FAILED $name" >> logs/failures.txt
  done
}

queue 0 teacher-wrn28-cutmix-val teacher-wrn28-cutmix-full teacher-wrn28-cutout-val teacher-wrn28-cutout-full &
queue 1 teacher-wrn28-samcutout-val teacher-wrn28-samcutout-full &
wait
echo "ALL TEACHERS DONE" >> logs/status.txt
