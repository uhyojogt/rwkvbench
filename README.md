# RWKVBench

RWKVBench is a benchmark dashboard project for comparing RWKV-7 inference
backends across hardware, batch sizes, prompt lengths, and generation lengths.

The first milestone is intentionally small:

- run a mock benchmark on any development machine
- emit a stable JSON result
- validate the result in CI
- later run real GPU benchmarks on the V100 server

## Repository Layout

```text
apps/
  api/                 # Future FastAPI service
  web/                 # Future dashboard
  worker/              # Future GPU worker
packages/
  benchkit/            # Benchmark runner and result schema
results/
  examples/            # Small committed example outputs
scripts/               # Local helper scripts
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\packages\benchkit pytest
```

Run a mock benchmark:

```powershell
python -m benchkit.run --backend mock --model fake-rwkv7 --out results\examples\mock-run.json
```

Run a CUDA smoke benchmark on a GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchkit.run \
  --backend torch-cuda \
  --model torch-linear-smoke \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 128 \
  --dtype fp16 \
  --hidden-size 4096 \
  --out results/raw/torch-cuda-smoke.json
```

Run a small CUDA sweep:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run-torch-cuda-sweep.sh
cat results/raw/torch-cuda-sweep/summary.json
```

Run a Hugging Face causal LM smoke benchmark:

```bash
python -m pip install "numpy<2" "transformers==4.38.2" "accelerate==0.27.2"
CUDA_VISIBLE_DEVICES=0 bash scripts/run-hf-causal-lm-smoke.sh
cat results/raw/hf-causal-lm-smoke.json

CUDA_VISIBLE_DEVICES=0 bash scripts/run-hf-causal-lm-sweep.sh
cat results/raw/hf-causal-lm-sweep/summary.json
```

Run a converted RWKV-7 Hugging Face adapter benchmark:

```bash
export RWKV7_MODEL_DIR=/path/to/rwkv7_g1d_01b_hf
CUDA_VISIBLE_DEVICES=0 RWKV7_NATIVE_MODEL=1 bash scripts/run-rwkv7-hf-smoke.sh
cat results/raw/rwkv7-hf-smoke.json

CUDA_VISIBLE_DEVICES=0 RWKV7_NATIVE_MODEL=1 bash scripts/run-rwkv7-hf-sweep.sh
cat results/raw/rwkv7-hf-sweep/summary.json
```

Local adapter directories should use Python-safe names without dots because
Transformers loads trusted model code through a generated Python package.

The Hugging Face backend warms both prefill and decode paths before measurement.
Decode throughput uses one end-to-end CUDA interval, while P50 and P95 are
calculated from per-generation-step CUDA event samples. The public `model` field
contains only a local directory name; local absolute paths are not emitted.

Run tests:

```powershell
pytest
```

## Dashboard

The dashboard reads `apps/web/public/data/summary.json`.

```powershell
cd apps\web
npm install
npm run dev
```

To preview real server results, copy the server sweep file into the web public
data path:

```bash
cp results/raw/torch-cuda-sweep/summary.json apps/web/public/data/summary.json
```

## GPU Server Flow

For now, use GitHub as the code handoff:

```bash
git pull --ff-only
python -m pip install -e ./packages/benchkit
python -m benchkit.run --backend mock --model fake-rwkv7 --out results/examples/server-mock-run.json
```

Then verify CUDA timing with PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

CUDA_VISIBLE_DEVICES=0 bash scripts/run-torch-cuda-smoke.sh
cat results/raw/torch-cuda-smoke.json

CUDA_VISIBLE_DEVICES=0 bash scripts/run-torch-cuda-sweep.sh
cat results/raw/torch-cuda-sweep/summary.json

python -m pip install "numpy<2" "transformers==4.38.2" "accelerate==0.27.2"
CUDA_VISIBLE_DEVICES=0 bash scripts/run-hf-causal-lm-smoke.sh
cat results/raw/hf-causal-lm-smoke.json

export RWKV7_MODEL_DIR=/path/to/rwkv7_g1d_01b_hf
CUDA_VISIBLE_DEVICES=0 RWKV7_NATIVE_MODEL=1 bash scripts/run-rwkv7-hf-smoke.sh
cat results/raw/rwkv7-hf-smoke.json
```

The RWKV-7 adapter path expects a converted Hugging Face model directory, not a
raw `.pth` checkpoint. The community adapter documents conversion through
`convert_rwkv7_to_hf.py` in `btlqql/rwkv7-hf-adapter`.

## RWKV-7 Wrapper Fast-Token Comparison

Keep the reproducible native baseline environment unchanged. Create a separate
environment for the FLA-backed wrapper and CUDA Graph comparison. The upstream
V100 validation used PyTorch 2.5.1 with CUDA 12.4 and a source build reporting
FLA 0.5.2. That FLA version is not published on PyPI, so this lane uses the
closest public release, FLA 0.5.1. Install its two packages without dependency
resolution to preserve the validated PyTorch 2.5 / CUDA 12.4 runtime; FLA's
newer CUDA extra otherwise upgrades the environment beyond this V100 lane.

```bash
conda create -n rwkvbench-fla python=3.10 -y
conda activate rwkvbench-fla

python -m pip install --upgrade pip
python -m pip install "numpy<2"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "transformers==4.57.1" safetensors einops
python -m pip install --no-deps "fla-core==0.5.1" "flash-linear-attention==0.5.1"
python -m pip install -e ./packages/benchkit
```

Verify the runtime before loading model weights:

```bash
python - <<'PY'
import torch
import transformers
import fla

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("fla:", getattr(fla, "__version__", "unknown"))
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Warnings that Triton 3.1 or Python 3.10 are below the recommended versions are
expected in this compatibility lane. An import failure is not expected.

Run the same prompt/decode shape through the wrapper's FLA, native JIT, and
native CUDA Graph token backends:

```bash
export RWKV7_MODEL_DIR=/data/healong/models/rwkv7_g1d_01b_hf
CUDA_VISIBLE_DEVICES=0 bash scripts/run-rwkv7-wrapper-compare.sh
```

Each result records both the requested and actual fast-token backend. Treat a
requested/actual mismatch as a failed comparison rather than benchmark data.
