from __future__ import annotations

import json
import subprocess
import sys


def test_mock_benchmark_cli_writes_result(tmp_path):
    output = tmp_path / "mock-run.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchkit.run",
            "--backend",
            "mock",
            "--model",
            "fake-rwkv7",
            "--batch-size",
            "2",
            "--prompt-len",
            "128",
            "--gen-len",
            "64",
            "--out",
            str(output),
        ],
        check=True,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["backend"] == "mock"
    assert data["model"] == "fake-rwkv7"
    assert data["config"]["batch_size"] == 2
    assert data["metrics"]["decode_tps"] > 0
    assert "hardware" in data
