#!/usr/bin/env bash
set -euo pipefail

: "${RWKV7_MODEL_DIR:?Set RWKV7_MODEL_DIR to a converted RWKV-7 Hugging Face model directory}"

for backend in fla native_jit native_graph; do
  bash scripts/run-rwkv7-wrapper-fast-token.sh "$backend"
done

python - <<'PY'
import json
from pathlib import Path

print("backend actual prefill_tps decode_tps p50_ms p95_ms vram_mb")
for backend in ("fla", "native_jit", "native_graph"):
    result = json.loads(
        Path(f"results/raw/rwkv7-wrapper-{backend}.json").read_text(encoding="utf-8")
    )
    metrics = result["metrics"]
    metadata = result["metadata"]
    print(
        backend,
        metadata.get("rwkv7_fast_token_backend_actual"),
        metrics["prefill_tps"],
        metrics["decode_tps"],
        metrics["latency_p50_ms"],
        metrics["latency_p95_ms"],
        metrics["vram_peak_mb"],
    )
PY
