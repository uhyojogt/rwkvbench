from __future__ import annotations

import math
import time
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
        torch.cuda.reset_peak_memory_stats(torch_device)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = hf_model(input_ids=input_ids, use_cache=True)

        synchronize(torch, torch_device)
        prefill_ms, outputs = time_torch_op_ms(
            torch=torch,
            device=torch_device,
            op=lambda: hf_model(input_ids=input_ids, use_cache=True),
        )
        decode_ms = time_decode_ms(
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

    return BenchmarkResult(
        schema_version="1.0",
        run_id=str(uuid4()),
        created_at=utc_now_iso(),
        backend=backend,
        model=model,
        hardware=collect_hardware_info(),
        config=config,
        metrics=BenchmarkMetrics(
            prefill_tps=round(tokens_per_second(prefill_tokens, prefill_ms), 2),
            decode_tps=round(tokens_per_second(decode_tokens, decode_ms), 2),
            total_tps=round(tokens_per_second(prefill_tokens + decode_tokens, total_ms), 2),
            latency_p50_ms=round(decode_ms / max(config.gen_len, 1), 4),
            latency_p95_ms=round(decode_ms / max(config.gen_len, 1), 4),
            vram_peak_mb=vram_peak_mb,
        ),
        metadata={
            "source": "hf-causal-lm",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(torch_device),
            "model_class": hf_model.__class__.__name__,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
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


def time_decode_ms(*, torch, model, device, prefill_outputs, gen_len: int) -> float:
    logits = prefill_outputs.logits
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = getattr(prefill_outputs, "past_key_values", None)

    def decode_loop():
        nonlocal next_token, past_key_values
        current_ids = next_token
        for _ in range(gen_len):
            if past_key_values is None:
                outputs = model(input_ids=current_ids, use_cache=True)
                current_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = getattr(outputs, "past_key_values", None)

    elapsed_ms, _ = time_torch_op_ms(torch=torch, device=device, op=decode_loop)
    return elapsed_ms


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(torch, device) -> int | None:
    if device.type != "cuda":
        return None

    return int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
