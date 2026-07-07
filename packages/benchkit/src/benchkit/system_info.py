from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys

from benchkit.schema import HardwareInfo


def collect_hardware_info() -> HardwareInfo:
    gpu_info = query_nvidia_smi()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        cpu=platform.processor() or platform.machine(),
        gpu=gpu_info.get("gpu"),
        driver=gpu_info.get("driver"),
        cuda=gpu_info.get("cuda"),
    )


def query_nvidia_smi() -> dict[str, str | None]:
    if shutil.which("nvidia-smi") is None:
        return {"gpu": None, "driver": None, "cuda": None}

    summary = run_command(["nvidia-smi"])
    cuda = parse_cuda_version(summary)

    query_command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    query_output = run_command(query_command)
    parsed = parse_gpu_query_output(query_output)

    return {
        "gpu": parsed.get("gpu"),
        "driver": parsed.get("driver"),
        "cuda": cuda,
    }


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""

    return completed.stdout


def parse_gpu_query_output(output: str) -> dict[str, str | None]:
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 2:
        return {"gpu": None, "driver": None}

    return {"gpu": parts[0], "driver": parts[1]}


def parse_cuda_version(output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def hardware_info_json() -> str:
    return json.dumps(collect_hardware_info().__dict__, indent=2)
