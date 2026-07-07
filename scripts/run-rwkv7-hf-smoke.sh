#!/usr/bin/env bash
set -euo pipefail

: "${RWKV7_MODEL_DIR:?Set RWKV7_MODEL_DIR to a converted RWKV-7 Hugging Face model directory}"

python -m benchkit.run \
  --backend hf-causal-lm \
  --model "$RWKV7_MODEL_DIR" \
  --batch-size 1 \
  --prompt-len 64 \
  --gen-len 32 \
  --dtype fp16 \
  --device cuda:0 \
  --trust-remote-code \
  --out results/raw/rwkv7-hf-smoke.json
