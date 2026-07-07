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

Run tests:

```powershell
pytest
```

## GPU Server Flow

For now, use GitHub as the code handoff:

```bash
git pull --ff-only
python -m pip install -e ./packages/benchkit
python -m benchkit.run --backend mock --model fake-rwkv7 --out results/examples/server-mock-run.json
```

Real RWKV / Transformers / vLLM / SGLang backends will be added after this
mock result pipeline is stable.
