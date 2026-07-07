#!/usr/bin/env bash
set -euo pipefail

: "${RWKV7_MODEL_DIR:?Set RWKV7_MODEL_DIR to a converted RWKV-7 Hugging Face model directory}"

python -m benchkit.sweep \
  --backend hf-causal-lm \
  --model "$RWKV7_MODEL_DIR" \
  --batch-sizes 1,2,4 \
  --prompt-lens 64,128,512 \
  --gen-lens 32 \
  --dtype fp16 \
  --device cuda:0 \
  --trust-remote-code \
  --out-dir results/raw/rwkv7-hf-sweep
