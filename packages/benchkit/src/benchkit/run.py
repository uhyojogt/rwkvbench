from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchkit.mock_backend import run_mock_benchmark
from benchkit.schema import BenchmarkConfig


SUPPORTED_BACKENDS = {"mock"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an RWKVBench benchmark")
    parser.add_argument("--backend", required=True, choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--out", required=True, help="Path to write benchmark JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        dtype=args.dtype,
    )

    if args.backend == "mock":
        result = run_mock_benchmark(backend=args.backend, model=args.model, config=config)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark result to {out_path}")


if __name__ == "__main__":
    main()
