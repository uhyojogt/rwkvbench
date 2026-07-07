from __future__ import annotations

from uuid import uuid4

from benchkit.schema import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    utc_now_iso,
)
from benchkit.system_info import collect_hardware_info


def run_torch_cuda_benchmark(
    *,
    backend: str,
    model: str,
    config: BenchmarkConfig,
    hidden_size: int,
    warmup_steps: int,
    device: str,
) -> BenchmarkResult:
    torch = import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available on this machine.")

    torch_device = torch.device(device)
    dtype = resolve_torch_dtype(torch, config.dtype)

    if torch_device.type != "cuda":
        raise ValueError(f"torch-cuda backend requires a CUDA device, got {device!r}.")

    torch.cuda.set_device(torch_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch_device)

    weight = torch.randn((hidden_size, hidden_size), device=torch_device, dtype=dtype)
    prefill_input = torch.randn(
        (config.batch_size * config.prompt_len, hidden_size),
        device=torch_device,
        dtype=dtype,
    )
    decode_input = torch.randn((config.batch_size, hidden_size), device=torch_device, dtype=dtype)

    for _ in range(warmup_steps):
        _ = prefill_input @ weight
        _ = decode_input @ weight
    torch.cuda.synchronize(torch_device)

    prefill_ms = time_cuda_op_ms(torch, torch_device, lambda: prefill_input @ weight)
    decode_ms = time_cuda_decode_ms(
        torch=torch,
        device=torch_device,
        steps=config.gen_len,
        op=lambda: decode_input @ weight,
    )

    prefill_tokens = config.batch_size * config.prompt_len
    decode_tokens = config.batch_size * config.gen_len
    prefill_tps = tokens_per_second(prefill_tokens, prefill_ms)
    decode_tps = tokens_per_second(decode_tokens, decode_ms)
    total_tps = tokens_per_second(prefill_tokens + decode_tokens, prefill_ms + decode_ms)
    latency_p50_ms = decode_ms / max(config.gen_len, 1)
    latency_p95_ms = latency_p50_ms
    vram_peak_mb = int(torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024))

    return BenchmarkResult(
        schema_version="1.0",
        run_id=str(uuid4()),
        created_at=utc_now_iso(),
        backend=backend,
        model=model,
        hardware=collect_hardware_info(),
        config=config,
        metrics=BenchmarkMetrics(
            prefill_tps=round(prefill_tps, 2),
            decode_tps=round(decode_tps, 2),
            total_tps=round(total_tps, 2),
            latency_p50_ms=round(latency_p50_ms, 4),
            latency_p95_ms=round(latency_p95_ms, 4),
            vram_peak_mb=vram_peak_mb,
        ),
        metadata={
            "source": "torch-cuda-smoke",
            "torch_version": torch.__version__,
            "device": str(torch_device),
            "device_name": torch.cuda.get_device_name(torch_device),
            "hidden_size": hidden_size,
            "warmup_steps": warmup_steps,
        },
    )


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The torch-cuda backend requires PyTorch. Install a CUDA-enabled PyTorch build "
            "on the GPU server, then rerun this command."
        ) from exc

    return torch


def resolve_torch_dtype(torch, dtype: str):
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of: {supported}")

    return mapping[dtype]


def time_cuda_op_ms(torch, device, op) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = op()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def time_cuda_decode_ms(torch, device, steps: int, op) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        _ = op()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def tokens_per_second(tokens: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0
    return tokens / (elapsed_ms / 1000)
