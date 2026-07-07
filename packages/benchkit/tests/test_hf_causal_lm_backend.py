from __future__ import annotations

import pytest

from benchkit.hf_causal_lm_backend import resolve_torch_dtype


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


def test_resolve_torch_dtype_supports_common_aliases():
    assert resolve_torch_dtype(FakeTorch, "fp16") == "float16"
    assert resolve_torch_dtype(FakeTorch, "bf16") == "bfloat16"
    assert resolve_torch_dtype(FakeTorch, "fp32") == "float32"


def test_resolve_torch_dtype_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_torch_dtype(FakeTorch, "int8")
