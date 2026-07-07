#!/usr/bin/env bash
set -euo pipefail

python -m benchkit.sweep \
  --backend torch-cuda \
  --model torch-linear-smoke \
  --batch-sizes 1,2,4,8 \
  --prompt-lens 128,512 \
  --gen-lens 128 \
  --dtype fp16 \
  --hidden-size 4096 \
  --out-dir results/raw/torch-cuda-sweep
