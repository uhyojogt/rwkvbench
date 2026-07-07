from __future__ import annotations

import json
import platform
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

    query = "name,driver_version,cuda_version"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"gpu": None, "driver": None, "cuda": None}

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return {"gpu": None, "driver": None, "cuda": None}

    return {"gpu": parts[0], "driver": parts[1], "cuda": parts[2]}


def hardware_info_json() -> str:
    return json.dumps(collect_hardware_info().__dict__, indent=2)
