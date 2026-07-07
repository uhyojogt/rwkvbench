from __future__ import annotations

import hashlib
import random
import time
from uuid import uuid4

from benchkit.schema import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    utc_now_iso,
)
from benchkit.system_info import collect_hardware_info


def run_mock_benchmark(
    *,
    backend: str,
    model: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    seed = stable_seed(backend, model, config)
    rng = random.Random(seed)

    # Make the command feel real without slowing down local development.
    time.sleep(0.05)

    scale = max(config.batch_size, 1)
    prompt_factor = max(config.prompt_len / 128, 1)
    gen_factor = max(config.gen_len / 128, 1)
    dtype_factor = 1.15 if config.dtype in {"fp16", "bf16"} else 1.0

    prefill_tps = round((1800 * scale * dtype_factor / prompt_factor) + rng.uniform(-50, 50), 2)
    decode_tps = round((650 * scale * dtype_factor / gen_factor) + rng.uniform(-25, 25), 2)
    total_tps = round((prefill_tps + decode_tps) / 2, 2)
    latency_p50_ms = round(1000 / max(decode_tps / scale, 1), 2)
    latency_p95_ms = round(latency_p50_ms * rng.uniform(1.25, 1.6), 2)
    vram_peak_mb = int(1024 + config.batch_size * 320 + config.prompt_len * 0.8)

    return BenchmarkResult(
        schema_version="1.0",
        run_id=str(uuid4()),
        created_at=utc_now_iso(),
        backend=backend,
        model=model,
        hardware=collect_hardware_info(),
        config=config,
        metrics=BenchmarkMetrics(
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            total_tps=total_tps,
            latency_p50_ms=latency_p50_ms,
            latency_p95_ms=latency_p95_ms,
            vram_peak_mb=vram_peak_mb,
        ),
        metadata={"source": "mock"},
    )


def stable_seed(backend: str, model: str, config: BenchmarkConfig) -> int:
    raw = f"{backend}:{model}:{config.batch_size}:{config.prompt_len}:{config.gen_len}:{config.dtype}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
