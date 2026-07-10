from __future__ import annotations

import pytest

from benchkit.hf_causal_lm_backend import (
    collect_runtime_metadata,
    percentile,
    public_model_name,
    resolve_torch_dtype,
)


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


def test_percentile_uses_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.5) == pytest.approx(25.0)
    assert percentile(values, 0.95) == pytest.approx(38.5)


def test_percentile_handles_empty_samples():
    assert percentile([], 0.5) == 0.0


def test_public_model_name_hides_local_path(tmp_path):
    model_dir = tmp_path / "rwkv7_g1d_01b_hf"
    model_dir.mkdir()

    assert public_model_name(str(model_dir)) == "rwkv7_g1d_01b_hf"
    assert public_model_name("BlinkDL/rwkv7-g1") == "BlinkDL/rwkv7-g1"


def test_runtime_metadata_identifies_native_jit(monkeypatch):
    NativeModel = type("NativeRWKV7ForCausalLM", (), {})
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_JIT", "1")

    metadata = collect_runtime_metadata(NativeModel())

    assert metadata["runtime_variant"] == "native-jit"
    assert metadata["rwkv7_native_model_jit"] is True


def test_runtime_metadata_records_wrapper_backend(monkeypatch):
    WrapperModel = type("RWKV7ForCausalLM", (), {})
    model = WrapperModel()
    model._rwkv7_last_fast_token_backend = "native_graph"
    monkeypatch.setenv("RWKV7_FAST_TOKEN_BACKEND", "native_graph")

    metadata = collect_runtime_metadata(model)

    assert metadata["runtime_variant"] == "wrapper-native_graph"
    assert metadata["rwkv7_fast_token_backend_requested"] == "native_graph"
    assert metadata["rwkv7_fast_token_backend_actual"] == "native_graph"
