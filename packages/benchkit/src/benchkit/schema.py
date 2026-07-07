from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HardwareInfo:
    os: str
    python: str
    cpu: str
    gpu: str | None = None
    driver: str | None = None
    cuda: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    prompt_len: int
    gen_len: int
    dtype: str


@dataclass(frozen=True)
class BenchmarkMetrics:
    prefill_tps: float
    decode_tps: float
    total_tps: float
    latency_p50_ms: float
    latency_p95_ms: float
    vram_peak_mb: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    run_id: str
    created_at: str
    backend: str
    model: str
    hardware: HardwareInfo
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
