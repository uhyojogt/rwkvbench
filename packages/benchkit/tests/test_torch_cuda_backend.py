from __future__ import annotations

from benchkit.torch_cuda_backend import tokens_per_second


def test_tokens_per_second_converts_milliseconds():
    assert tokens_per_second(128, 64) == 2000


def test_tokens_per_second_handles_zero_elapsed_time():
    assert tokens_per_second(128, 0) == 0
