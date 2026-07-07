from __future__ import annotations

from benchkit.mock_backend import run_mock_benchmark
from benchkit.schema import BenchmarkConfig, BenchmarkResult


SUPPORTED_BACKENDS = {"mock", "torch-cuda"}


def run_benchmark(
    *,
    backend: str,
    model: str,
    config: BenchmarkConfig,
    hidden_size: int = 4096,
    warmup_steps: int = 5,
    device: str = "cuda:0",
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

    supported = ", ".join(sorted(SUPPORTED_BACKENDS))
    raise ValueError(f"Unsupported backend {backend!r}; expected one of: {supported}")
