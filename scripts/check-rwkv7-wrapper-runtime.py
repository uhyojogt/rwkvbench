from __future__ import annotations

import sys


def main() -> None:
    try:
        import fla
        import torch
        import transformers
        import triton
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "RWKV-7 wrapper runtime import failed. Expected the isolated V100 lane: "
            "torch 2.5.1+cu124, transformers 4.57.1, fla-core 0.5.1, "
            "flash-linear-attention 0.5.1."
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
