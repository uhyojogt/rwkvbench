from __future__ import annotations

from benchkit.mock_backend import run_mock_benchmark
from benchkit.schema import BenchmarkConfig, BenchmarkResult


SUPPORTED_BACKENDS = {"hf-causal-lm", "mock", "torch-cuda"}


def run_benchmark(
    *,
    backend: str,
    model: str,
    config: BenchmarkConfig,
    hidden_size: int = 4096,
    warmup_steps: int = 5,
    device: str = "cuda:0",
    prompt: str = "RWKVBench measures model inference throughput.",
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> BenchmarkResult:
    if backend == "mock":
        return run_mock_benchmark(backend=backend, model=model, config=config)

    if backend == "torch-cuda":
        from benchkit.torch_cuda_backend import run_torch_cuda_benchmark

        return run_torch_cuda_benchmark(
            backend=backend,
            model=model,
            config=config,
            hidden_size=hidden_size,
            warmup_steps=warmup_steps,
            device=device,
        )

    if backend == "hf-causal-lm":
        from benchkit.hf_causal_lm_backend import run_hf_causal_lm_benchmark

        return run_hf_causal_lm_benchmark(
            backend=backend,
            model=model,
            config=config,
            prompt=prompt,
            warmup_steps=warmup_steps,
            device=device,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

    supported = ", ".join(sorted(SUPPORTED_BACKENDS))
    raise ValueError(f"Unsupported backend {backend!r}; expected one of: {supported}")
