#!/usr/bin/env bash
set -euo pipefail

: "${RWKV7_MODEL_DIR:?Set RWKV7_MODEL_DIR to a converted RWKV-7 Hugging Face model directory}"

backend="${1:-native_graph}"
case "$backend" in
  fla|native_jit|native_graph) ;;
  *)
    echo "Unsupported fast-token backend: $backend" >&2
    echo "Expected one of: fla, native_jit, native_graph" >&2
    exit 2
    ;;
esac

unset RWKV7_NATIVE_MODEL
export RWKV7_FAST_TOKEN_BACKEND="$backend"

python -m benchkit.run \
  --backend hf-causal-lm \
  --model "$RWKV7_MODEL_DIR" \
  --batch-size "${BATCH_SIZE:-1}" \
  --prompt-len "${PROMPT_LEN:-128}" \
  --gen-len "${GEN_LEN:-128}" \
  --dtype fp16 \
  --device cuda:0 \
  --trust-remote-code \
  --out "results/raw/rwkv7-wrapper-${backend}.json"
