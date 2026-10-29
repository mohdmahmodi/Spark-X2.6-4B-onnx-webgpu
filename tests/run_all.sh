#!/usr/bin/env bash
# Full validation run (about 3 h on an RTX 3090 Ti). GPU jobs run one at a time: sharing the GPU
# between transformers and the WebGPU page caused device-lost errors in Chrome.
# Needs: `npm run serve` running, hf/ checkpoints, the exported web/spark/ files.
# Each step is resumable or cheap to redo; results land in export/tests/results/, report in docs/VALIDATION.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
ORT_PY=${ORT_PY:-.venv/Scripts/python}

$PY tests/suites.py

# references: transformers bf16 (original), fp32 (noise floor), official INT4 weights
$PY tests/run_hf.py hf_bf16 parity
for s in hf_bf16 hf_fp32 hf_int4; do $PY tests/run_hf.py $s logits; done
$PY tests/run_hf.py hf_int4 parity
for s in hf_bf16 hf_int4; do
  for suite in gsm8k gsm8k_full bfcl; do $PY tests/run_hf.py $s $suite --batch 24; done
  $PY tests/run_hf.py $s niah
done

# same graph on CPU onnxruntime (separates quantization error from WebGPU kernel error)
$ORT_PY tests/run_ort.py logits
$ORT_PY tests/run_ort.py parity

# the real browser build (Chrome WebGPU via puppeteer-core)
node tests/run_browser.mjs logits
for suite in parity niah bfcl gsm8k gsm8k_full; do node tests/run_browser.mjs $suite; done

# features and stability through the HTTP bridge to the page
node tests/browser_bridge.mjs & BRIDGE=$!
trap 'kill $BRIDGE' EXIT
until curl -sf localhost:8765/info >/dev/null; do sleep 2; done
$PY tests/features.py
$PY tests/stability.py --minutes 30

$PY tests/report.py
