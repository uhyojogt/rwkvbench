from __future__ import annotations

import math
import os
import time
from pathlib import Path
from uuid import uuid4

from benchkit.schema import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    utc_now_iso,
)
from benchkit.system_info import collect_hardware_info
from benchkit.torch_cuda_backend import tokens_per_second


def run_hf_causal_lm_benchmark(
    *,
    backend: str,
    model: str,
    config: BenchmarkConfig,
    prompt: str,
    warmup_steps: int,
    device: str,
    trust_remote_code: bool,
    revision: str | None,
) -> BenchmarkResult:
    torch, transformers = import_runtime()
    torch_device = torch.device(device)
    dtype = resolve_torch_dtype(torch, config.dtype)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    hf_model.to(torch_device)
    hf_model.eval()

    input_ids = build_prompt_input_ids(
        torch=torch,
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_len=config.prompt_len,
        batch_size=config.batch_size,
        device=torch_device,
    )

    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        torch.cuda.empty_cache()

    with torch.inference_mode():
        warmup_outputs = None
        for _ in range(warmup_steps):
            warmup_outputs = hf_model(input_ids=input_ids, use_cache=True)

        if warmup_outputs is not None:
            time_decode_ms(
                torch=torch,
                model=hf_model,
                device=torch_device,
                prefill_outputs=warmup_outputs,
                gen_len=warmup_steps,
            )
            warmup_outputs = None

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            torch.cuda.reset_peak_memory_stats(torch_device)

        synchronize(torch, torch_device)
        prefill_ms, outputs = time_torch_op_ms(
            torch=torch,
            device=torch_device,
            op=lambda: hf_model(input_ids=input_ids, use_cache=True),
        )
        decode_ms, token_latencies_ms = time_decode_ms(
            torch=torch,
            model=hf_model,
            device=torch_device,
            prefill_outputs=outputs,
            gen_len=config.gen_len,
        )

    prefill_tokens = config.batch_size * config.prompt_len
    decode_tokens = config.batch_size * config.gen_len
    total_ms = prefill_ms + decode_ms
    vram_peak_mb = peak_memory_mb(torch, torch_device)

    runtime_metadata = collect_runtime_metadata(hf_model)

    return BenchmarkResult(
        schema_version="1.0",
        run_id=str(uuid4()),
        created_at=utc_now_iso(),
        backend=backend,
        model=public_model_name(model),
        hardware=collect_hardware_info(),
        config=config,
        metrics=BenchmarkMetrics(
            prefill_tps=round(tokens_per_second(prefill_tokens, prefill_ms), 2),
            decode_tps=round(tokens_per_second(decode_tokens, decode_ms), 2),
            total_tps=round(tokens_per_second(prefill_tokens + decode_tokens, total_ms), 2),
            latency_p50_ms=round(percentile(token_latencies_ms, 0.50), 4),
            latency_p95_ms=round(percentile(token_latencies_ms, 0.95), 4),
            vram_peak_mb=vram_peak_mb,
        ),
        metadata={
            "source": "hf-causal-lm",
            "torch_version": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "transformers_version": transformers.__version__,
            "device": str(torch_device),
            "model_class": hf_model.__class__.__name__,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            **runtime_metadata,
        },
    )


def import_runtime():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The hf-causal-lm backend requires PyTorch.") from exc

    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "The hf-causal-lm backend requires Transformers. Install it on the GPU server "
            "with a command such as: python -m pip install 'transformers<5' accelerate"
        ) from exc

    return torch, transformers


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


def build_prompt_input_ids(
    *,
    torch,
    tokenizer,
    prompt: str,
    prompt_len: int,
    batch_size: int,
    device,
):
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    token_ids = encoded["input_ids"][0]
    if token_ids.numel() == 0:
        token_ids = torch.tensor([tokenizer.eos_token_id or 0], dtype=torch.long)

    repeats = math.ceil(prompt_len / token_ids.numel())
    token_ids = token_ids.repeat(repeats)[:prompt_len]
    return token_ids.unsqueeze(0).repeat(batch_size, 1).to(device)


def time_torch_op_ms(*, torch, device, op):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = op()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)), result

    start = time.perf_counter()
    result = op()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result


def time_decode_ms(
    *, torch, model, device, prefill_outputs, gen_len: int
) -> tuple[float, list[float]]:
    logits = prefill_outputs.logits
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = getattr(prefill_outputs, "past_key_values", None)

    if device.type == "cuda":
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        token_events = []
        overall_start.record()
        for _ in range(gen_len):
            token_start = torch.cuda.Event(enable_timing=True)
            token_end = torch.cuda.Event(enable_timing=True)
            token_start.record()
            next_token, past_key_values = decode_step(
                model=model,
                next_token=next_token,
                past_key_values=past_key_values,
            )
            token_end.record()
            token_events.append((token_start, token_end))
        overall_end.record()
        torch.cuda.synchronize(device)
        return (
            float(overall_start.elapsed_time(overall_end)),
            [float(start.elapsed_time(end)) for start, end in token_events],
        )

    overall_start = time.perf_counter()
    token_latencies_ms = []
    for _ in range(gen_len):
        token_start = time.perf_counter()
        next_token, past_key_values = decode_step(
            model=model,
            next_token=next_token,
            past_key_values=past_key_values,
        )
        token_latencies_ms.append((time.perf_counter() - token_start) * 1000)
    return (time.perf_counter() - overall_start) * 1000, token_latencies_ms


def decode_step(*, model, next_token, past_key_values):
    model_kwargs = {"input_ids": next_token, "use_cache": True}
    if past_key_values is not None:
        model_kwargs["past_key_values"] = past_key_values

    outputs = model(**model_kwargs)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, getattr(outputs, "past_key_values", None)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def public_model_name(model: str) -> str:
    path = Path(model)
    return path.name if path.exists() else model


def collect_runtime_metadata(model) -> dict[str, object]:
    model_class = model.__class__.__name__
    actual_fast_backend = getattr(model, "_rwkv7_last_fast_token_backend", None)
    requested_fast_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")

    if model_class == "NativeRWKV7ForCausalLM":
        native_jit = env_enabled("RWKV7_NATIVE_MODEL_JIT", default=True)
        runtime_variant = "native-jit" if native_jit else "native-eager"
    elif actual_fast_backend:
        native_jit = None
        runtime_variant = f"wrapper-{actual_fast_backend}"
    else:
        native_jit = None
        runtime_variant = "hf-generic"

    return {
        "runtime_variant": runtime_variant,
        "rwkv7_native_model_jit": native_jit,
        "rwkv7_fast_token_backend_requested": requested_fast_backend,
        "rwkv7_fast_token_backend_actual": actual_fast_backend,
    }


def env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(torch, device) -> int | None:
    if device.type != "cuda":
        return None

    return int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
