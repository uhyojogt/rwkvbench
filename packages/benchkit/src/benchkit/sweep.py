from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchkit.backends import SUPPORTED_BACKENDS, run_benchmark
from benchkit.schema import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an RWKVBench benchmark sweep")
    parser.add_argument("--backend", required=True, choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--prompt-lens", default="128")
    parser.add_argument("--gen-lens", default="128")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for batch_size in parse_int_list(args.batch_sizes):
        for prompt_len in parse_int_list(args.prompt_lens):
            for gen_len in parse_int_list(args.gen_lens):
                config = BenchmarkConfig(
                    batch_size=batch_size,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    dtype=args.dtype,
                )
                result = run_benchmark(
                    backend=args.backend,
                    model=args.model,
                    config=config,
                    hidden_size=args.hidden_size,
                    warmup_steps=args.warmup_steps,
                    device=args.device,
                )
                filename = build_result_filename(args.backend, config)
                write_json(out_dir / filename, result.to_dict())
                summary.append(result.to_dict())
                print(
                    f"{filename}: prefill={result.metrics.prefill_tps} tok/s "
                    f"decode={result.metrics.decode_tps} tok/s "
                    f"vram={result.metrics.vram_peak_mb} MB"
                )

    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {len(summary)} benchmark result(s) to {out_dir}")


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value")

    return parsed


def build_result_filename(backend: str, config: BenchmarkConfig) -> str:
    return (
        f"{backend}"
        f"-bs{config.batch_size}"
        f"-p{config.prompt_len}"
        f"-g{config.gen_len}"
        f"-{config.dtype}.json"
    )


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
