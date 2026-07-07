#!/usr/bin/env bash
set -euo pipefail

python -m benchkit.sweep \
  --backend hf-causal-lm \
  --model sshleifer/tiny-gpt2 \
  --batch-sizes 1,2 \
  --prompt-lens 64,128 \
  --gen-lens 32 \
  --dtype fp16 \
  --device cuda:0 \
  --out-dir results/raw/hf-causal-lm-sweep
