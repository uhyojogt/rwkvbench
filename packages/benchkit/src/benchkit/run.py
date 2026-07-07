from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchkit.backends import SUPPORTED_BACKENDS, run_benchmark
from benchkit.schema import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an RWKVBench benchmark")
    parser.add_argument("--backend", required=True, choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="RWKVBench measures model inference throughput.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision")
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

    result = run_benchmark(
        backend=args.backend,
        model=args.model,
        config=config,
        hidden_size=args.hidden_size,
        warmup_steps=args.warmup_steps,
        device=args.device,
        prompt=args.prompt,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark result to {out_path}")


if __name__ == "__main__":
    main()
