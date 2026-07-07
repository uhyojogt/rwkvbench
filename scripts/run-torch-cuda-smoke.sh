#!/usr/bin/env bash
set -euo pipefail

python -m benchkit.run \
  --backend torch-cuda \
  --model torch-linear-smoke \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 128 \
  --dtype fp16 \
  --hidden-size 4096 \
  --out results/raw/torch-cuda-smoke.json
