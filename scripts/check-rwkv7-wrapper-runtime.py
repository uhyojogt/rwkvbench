from __future__ import annotations

import sys

from packaging.version import Version


def main() -> None:
    try:
        import torch
        import transformers
        import triton
    except Exception as exc:
        raise RuntimeError(
            "RWKV-7 wrapper base runtime import failed. Expected the isolated V100 lane: "
            "torch 2.6.0+cu124, Triton 3.2.x, and transformers 4.57.1."
        ) from exc

    triton_version = Version(triton.__version__)
    if not Version("3.2") <= triton_version < Version("3.3"):
        raise RuntimeError(
            f"Unsupported Triton {triton.__version__}. This wrapper lane requires "
            "Triton 3.2.x: Triton 3.1 cannot resolve FLA autotune keys passed as "
            "keyword arguments, while FLA 0.4.1 is not validated with Triton 3.3. "
            "Install torch 2.6.0+cu124 to obtain Triton 3.2."
        )

    try:
        import fla
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "RWKV-7 FLA runtime import failed. Expected fla-core 0.4.1 and "
            "flash-linear-attention 0.4.1."
        ) from exc

    del RWKV7ForCausalLM
    print("RWKV-7 wrapper runtime ready")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"transformers={transformers.__version__}")
    print(f"fla={getattr(fla, '__version__', 'unknown')} triton={triton.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
