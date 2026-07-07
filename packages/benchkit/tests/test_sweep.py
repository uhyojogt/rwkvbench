from __future__ import annotations

import json
import subprocess
import sys

from benchkit.sweep import build_result_filename, parse_int_list
from benchkit.schema import BenchmarkConfig


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("1, 2,4") == [1, 2, 4]


def test_build_result_filename_includes_config():
    config = BenchmarkConfig(batch_size=2, prompt_len=128, gen_len=64, dtype="fp16")
    assert build_result_filename("mock", config) == "mock-bs2-p128-g64-fp16.json"


def test_mock_sweep_writes_summary(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchkit.sweep",
            "--backend",
            "mock",
            "--model",
            "fake-rwkv7",
            "--batch-sizes",
            "1,2",
            "--prompt-lens",
            "128",
            "--gen-lens",
            "64",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert (tmp_path / "mock-bs1-p128-g64-fp16.json").exists()
    assert (tmp_path / "mock-bs2-p128-g64-fp16.json").exists()
