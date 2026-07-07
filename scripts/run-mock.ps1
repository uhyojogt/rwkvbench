$ErrorActionPreference = "Stop"

python -m benchkit.run `
  --backend mock `
  --model fake-rwkv7 `
  --batch-size 1 `
  --prompt-len 128 `
  --gen-len 128 `
  --dtype fp16 `
  --out results\examples\mock-run.json
