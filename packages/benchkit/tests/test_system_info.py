from __future__ import annotations

from benchkit.system_info import parse_cuda_version, parse_gpu_query_output


def test_parse_cuda_version_from_nvidia_smi_summary():
    output = """
    | NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4 |
    """

    assert parse_cuda_version(output) == "12.4"


def test_parse_gpu_query_output_uses_first_gpu():
    output = """
    Tesla V100-PCIE-32GB, 550.54.15
    Tesla V100-PCIE-32GB, 550.54.15
    """

    assert parse_gpu_query_output(output) == {
        "gpu": "Tesla V100-PCIE-32GB",
        "driver": "550.54.15",
    }
