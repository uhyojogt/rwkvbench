#!/usr/bin/env bash
set -euo pipefail

python -m benchkit.run \
  --backend hf-causal-lm \
  --model sshleifer/tiny-gpt2 \
  --batch-size 1 \
  --prompt-len 64 \
  --gen-len 32 \
  --dtype fp16 \
  --device cuda:0 \
  --out results/raw/hf-causal-lm-smoke.json
