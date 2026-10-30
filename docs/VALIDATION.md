# Spark-X2.6-4B browser build: validation report

## Summary

| test | original (transformers) | browser build | pass |
|---|---|---|---|
| Logits in WebGPU: mean KL to bf16 (160 pos.) | INT4 weights in HF: 0.0236 | 0.0238 (adds 0.00055 vs INT4) | PASS |
| Greedy parity: first token same (50 prompts) | INT4 weights in HF: 47/50 | 46/50 | PASS |
| Greedy parity: mean tokens before divergence | INT4 weights in HF: 30.1 | 29.1 | PASS |
| GSM8K | bf16 90.8%, INT4 weights 91.2% | 90.5% | PASS |
| BFCL v3 simple function calling | bf16 89.0%, INT4 weights 89.5% | 89.5% | PASS |
| Needle in a haystack | bf16 100.0%, INT4 weights 100.0% | 100.0% | PASS |
| Features: thinking / multi-turn / tool calls | 6/6 | 8/8 | PASS |
| Stability: 30 min back-to-back | n/a | 0 errors, tok/s 78->87, GPU mem +0 MB | PASS |

Shipped build: `q4f16` on nvidia ampere (Chrome, WebGPU, static KV cache). All browser results come from the real page driven by headless Chrome (tests/run_browser.mjs); the page tokenizes with tokenizers.js + the official chat template, and its prompt ids were checked to equal transformers' on every task.

## 1. Next-token logits vs transformers

20 prompts x 8 positions (after the prompt, then teacher-forcing the bf16 greedy continuation).

| system | mean KL vs bf16 | p95 | max | top-1 vs bf16 | mean KL vs INT4 weights | top-1 vs INT4 |
|---|---|---|---|---|---|---|
| **browser q4f16 (shipped)** | 0.0238 | 0.1157 | 0.392 | 94.4% | 0.00055 | 100.0% |
| transformers bf16, int8 embedding lookup (as in the build) | 0.0003 | 0.0015 | 0.003 | 100.0% | 0.02562 | 94.4% |
| transformers fp32 | 0.0001 | 0.0005 | 0.007 | 100.0% | 0.02501 | 94.4% |
| transformers + official INT4 weights | 0.0236 | 0.1135 | 0.397 | 94.4% | 0.00000 | 100.0% |
| same ONNX graph, CPU onnxruntime | 0.0240 | 0.1104 | 0.434 | 94.4% | 0.00028 | 100.0% |
| browser q4 fp32-act, accuracy_level 4 (earlier) | 0.0246 | 0.1187 | 0.462 | 95.0% | 0.00115 | 98.1% |
| browser q4f16, accuracy_level 4 (earlier build) | 0.0287 | 0.1139 | 0.416 | 95.6% | 0.00538 | 98.8% |
| CPU onnxruntime, accuracy_level 4 (earlier) | 0.0229 | 0.1145 | 0.301 | 95.0% | 0.00064 | 99.4% |

KL vs INT4 isolates what the ONNX/WebGPU build adds on top of the official quantization.

WebGPU kernels vs CPU kernels, same graph: mean KL 0.00016, max 0.0059, top-1 agree 100.0%.

## 2. Greedy output parity (50 prompts: chat, code, math, Chinese, tool calls; 128 tokens) vs transformers bf16

| system | first token same | identical 128-token outputs | mean tokens before divergence | median |
|---|---|---|---|---|
| browser q4 (fp32 activations) | 46/50 | 8/50 | 29.4 | 22 |
| **browser q4f16 (shipped)** | 46/50 | 9/50 | 29.1 | 22 |
| transformers + official INT4 weights | 47/50 | 9/50 | 30.1 | 22 |
| same ONNX graph, CPU onnxruntime | 46/50 | 9/50 | 29.5 | 23 |
| browser q4f16, accuracy_level 4 (earlier build) | 46/50 | 10/50 | 26.0 | 18 |
| CPU onnxruntime, accuracy_level 4 (earlier) | 47/50 | 10/50 | 29.6 | 23 |

Any 4-bit model drifts from bf16 at low-margin tokens; the INT4-weights row is the bar the browser build should meet.

Browser, mean matched tokens by category: chat 20, code 37, math 39, zh 20, tool 28

### Worst 5 divergences (shipped browser build)

**p02_math** (If a train travels 180 km in 2.5 hours, what is its average speed in km/h? Show your work.): diverges after 0 tokens

| transformers bf16 | browser |
|---|---|
| To find the average speed, we use the formula:<br><br>**Average Speed = Distance ÷ Time**<br><br>**Step 1: Identify the given values**<br>* Distance = 1 | ## Finding the Average Speed<br><br>**Formula:**<br>Average speed = Total distance ÷ Total time<br><br>**Given:**<br>* Total distance = 180 km<br>* Total time |

**p04_tool** (What's the weather like in Tokyo right now?): diverges after 0 tokens

| transformers bf16 | browser |
|---|---|
| <tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call> | Here's the current weather in Tokyo:<br><br>- **Temperature:** 22°C (72°F)<br>- **Conditions:** Partly cloudy<br>- **Humidity:** |

**p31_code** (Write a Rust function that returns the n-th Fibonacci number using iteration.): diverges after 0 tokens

| transformers bf16 | browser |
|---|---|
| Here's a Rust function that calculates the n-th Fibonacci number using iteration:<br><br>```rust<br>fn fibonacci(n: u64) -> u64 {<br>    if | # Rust Function for Nth Fibonacci Number Using Iteration<br><br>Here's a Rust function that calculates the n-th Fibonacci number using an iterative approach (which is more efficient than recurs |

**p42_chat** (What is the boiling point of water at the top of Mount Everest, roughly, and why is it low): diverges after 0 tokens

| transformers bf16 | browser |
|---|---|
| ## Boiling Point of Water at the Top of Mount Everest<br><br>The boiling point of water at the summit of Mount Everest (approximately 8,849 meters / 2 | The boiling point of water at the top of Mount Everest is roughly **90°C (about 194°F)**, compared to the standard 100°C ( |

**p07_math** (Solve for x: 3x + 7 = 22.): diverges after 1 tokens

| transformers bf16 | browser |
|---|---|
| We need answer solve equation. Need think? User asks solve for x: 3x+7=22. Simple. Need maybe show steps. Since thinking_effort high but final can be | We are asked to solve for x in the equation 3x + 7 = 22. This is a simple linear equation. We need to isolate x.<br><br>Step 1: Sub |


## GSM8K, all 1319 test problems (thinking off, greedy, 768 tokens)

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| **browser q4f16 (shipped)** | 1194/1319 (90.5%) | 22 / 13, p = 0.18 |
| transformers bf16 | 1198/1319 (90.8%) | 48 / 43, p = 0.68 |
| transformers bf16, int8 embedding lookup (as in the build) | 1197/1319 (90.8%) | 48 / 42, p = 0.60 |
| transformers + official INT4 weights | 1203/1319 (91.2%) |  |
| browser q4f16, accuracy_level 4 (earlier build) | 1178/1319 (89.3%) | 44 / 19, p = 0.00 |

## GSM8K, first 200 (diagnostic runs)

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| transformers bf16 | 187/200 (93.5%) | 7 / 9, p = 0.80 |
| transformers + official INT4 weights | 185/200 (92.5%) |  |
| browser q4 fp32-act, accuracy_level 4 (earlier) | 183/200 (91.5%) | 3 / 1, p = 0.62 |
| browser q4f16, accuracy_level 4 (earlier build) | 181/200 (90.5%) | 8 / 4, p = 0.39 |
| browser q4f16 fp32-residual, acc. level 4 (earlier) | 182/200 (91.0%) | 8 / 5, p = 0.58 |

## BFCL v3 simple function calling, first 200 (Spark tool format, SGLang spark25 parsing)

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| **browser q4f16 (shipped)** | 179/200 (89.5%) | 2 / 2, p = 1.00 |
| transformers bf16 | 178/200 (89.0%) | 5 / 4, p = 1.00 |
| transformers bf16, int8 embedding lookup (as in the build) | 175/200 (87.5%) | 8 / 4, p = 0.39 |
| transformers + official INT4 weights | 179/200 (89.5%) |  |
| browser q4f16, accuracy_level 4 (earlier build) | 179/200 (89.5%) | 3 / 3, p = 1.00 |

## Needle in a haystack, 1k-16k tokens, needle at start/middle/end

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| browser q4 (fp32 activations) | 12/12 (100.0%) |  |
| | 1024_start: ok, 1024_middle: ok, 1024_end: ok, 4096_start: ok, 4096_middle: ok, 4096_end: ok, 8192_start: ok, 8192_middle: ok, 8192_end: ok, 16384_start: ok, 16384_middle: ok, 16384_end: ok | |
| **browser q4f16 (shipped)** | 12/12 (100.0%) |  |
| | 1024_start: ok, 1024_middle: ok, 1024_end: ok, 4096_start: ok, 4096_middle: ok, 4096_end: ok, 8192_start: ok, 8192_middle: ok, 8192_end: ok, 16384_start: ok, 16384_middle: ok, 16384_end: ok | |
| transformers bf16 | 12/12 (100.0%) |  |
| | 1024_start: ok, 1024_middle: ok, 1024_end: ok, 4096_start: ok, 4096_middle: ok, 4096_end: ok, 8192_start: ok, 8192_middle: ok, 8192_end: ok, 16384_start: ok, 16384_middle: ok, 16384_end: ok | |
| transformers + official INT4 weights | 12/12 (100.0%) |  |
| | 1024_start: ok, 1024_middle: ok, 1024_end: ok, 4096_start: ok, 4096_middle: ok, 4096_end: ok, 8192_start: ok, 8192_middle: ok, 8192_end: ok, 16384_start: ok, 16384_middle: ok, 16384_end: ok | |
| browser q4f16, accuracy_level 4 (earlier build) | 12/12 (100.0%) |  |
| | 1024_start: ok, 1024_middle: ok, 1024_end: ok, 4096_start: ok, 4096_middle: ok, 4096_end: ok, 8192_start: ok, 8192_middle: ok, 8192_end: ok, 16384_start: ok, 16384_middle: ok, 16384_end: ok | |

## Features (thinking, multi-turn, tool calls)

| check | transformers bf16 | browser |
|---|---|---|
| thinking on (high) | ok | ok |
| thinking on (low) | ok | ok |
| thinking off | ok | ok |
| multi-turn recall | ok | ok |
| tool call parses (spark25) | ok | ok |
| JS parser == Python parser | n/a | ok |
| answer uses tool result | ok | ok |
| KV prefix reuse == fresh prefill | n/a | ok |

Browser outputs:

- **thinking on (high)**: `We need answer simple. Need compute 15% of 80 = 0.15*80=12. Final number. Need maybe just 12. Ensure no extra? User asks give final number.</think>12`
- **thinking on (low)**: `We need answer simple. 15% of 80 = 0.15*80=12. Final number.</think>12`
- **thinking off**: `12`
- **multi-turn recall**: `Your cat is called Miso, and your favourite colour is teal.`
- **tool call parses (spark25)**: `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Berlin</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>`
- **JS parser == Python parser**: `[{"name": "get_weather", "arguments": {"city": "Berlin", "unit": "celsius"}}]`
- **answer uses tool result**: `The weather in Berlin right now is **18°C** with **light rain**.`
- **KV prefix reuse == fresh prefill**: `reused 116 of 130 prompt tokens, 48/48 tokens identical, TTFT 149 -> 92 ms`

## Smaller GPU limits

`tests/run_browser.mjs --max-binding BYTES --max-buffer BYTES` gives the page a device with lower
`maxStorageBufferBindingSize` / `maxBufferSize` on the same GPU, a stand-in for phones and other GPUs with small limits
that were not available. WebGPU's defaults are 128 MiB per binding and 256 MiB per buffer.

| device limits (binding / buffer) | variant | needle, 12 cases, 1k-16k | greedy parity, 50 prompts |
|---|---|---|---|
| 128 MiB / 1 GiB | q4f16 | 12/12 identical to the unrestricted run | 50/50 identical |
| 128 MiB / 1 GiB | q4 | 12/12 identical to the unrestricted q4 run | 50/50 identical |
| any / 256 MiB | q4f16 | does not load: "This GPU allows 268 MB per buffer; the model needs 336 MB." | |

- **Weights.** The largest tensor, the int8 embedding / LM head table (335,544,320 bytes), is one GPU buffer, so it
  needs `maxBufferSize` of at least that. A smaller binding limit is fine: onnxruntime-web binds a larger buffer in
  several pieces. The loader used to also require the binding limit to fit the table, which rejected devices that
  can run the model.
- **Activations.** With a 128 MiB binding limit and the whole prompt in one pass, the q4 build returned all-zero
  output for the 16k prompts, without an error. The cause is in onnxruntime-web: its layer-norm kernels index their
  buffers directly instead of through the accessor that spans the pieces of a split buffer, so their output is zero
  past the first 128 MiB. The fp32 hidden state `[S, 2560]` passes 128 MiB at 13,108 tokens (fp16: 26,215). One-op
  models with these shapes show it for `LayerNormalization`, `SimplifiedLayerNormalization`, `SkipLayerNormalization`
  and `SkipSimplifiedLayerNormalization` in 1.31.0-dev.20260918; in 1.31.1 the first two are fixed and the Skip
  variants are not. Reported as [microsoft/onnxruntime#33755](https://github.com/microsoft/onnxruntime/issues/33755), with a fix for
  the Skip variants in [#33760](https://github.com/microsoft/onnxruntime/pull/33760). `MatMulNBits` with the model's shapes and element-wise ops were correct with up to 5 pieces.
  The loader now prefills in pieces that keep every activation, including the largest (the MLP intermediate
  `[S, 10240]`), within one binding (3,072 tokens for q4 and 6,400 for q4f16 at 128 MiB); the outputs above are from
  that. With this GPU's own limit (2 GiB per binding) the pieces are
  larger than the context limit, so nothing is split and the published results are unchanged (`browser_q4f16`
  rerun: 12/12 and 50/50 identical).


## INT4 reference and the input embedding

The official INT4 checkpoint is untied: its input embedding stays bf16 and only the LM head is int8. `hf_int4` loads it
exactly that way. This build looks up input tokens in the int8 LM-head table instead (one tensor for both, saving a
640 MB table); `hf_emb8` measures that change alone on the bf16 model: mean KL 0.0003 to bf16, GSM8K 1197 vs 1198
(McNemar p = 1.00), BFCL 175 vs 178 (p = 0.25).

An earlier version of this report loaded the INT4 reference with the int8 table for the input lookup too, as the build
does. That reference scored 1194 on GSM8K and 179 on BFCL; loaded as published it scores 1203 and 179 (p = 0.15 between
the two on GSM8K). The browser build's 1194 is within noise of both, and of bf16's 1198.


## Several tokens per forward pass

Speculative decoding (guess a few tokens ahead, check them all in one forward pass, keep the ones the model agrees
with) only pays off when a pass over k new tokens costs about as much as a pass over one. Measured in the browser with
the static KV cache after a cached prefix, q4f16, median of 12 passes:

| context | 1 token | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| 128 tokens | 11.0 ms | 16.1 ms | | 89.1 ms | 89.3 ms | 90.1 ms |
| 2,000 tokens | 10.4 ms | 16.5 ms | 19.7 ms * | 90.0 ms | 91.6 ms | 94.3 ms |
| 8,000 tokens | 12.5 ms | 17.6 ms | | 94.4 ms | 98.3 ms | 107.4 ms |

\* separate run, in which 1 token took 11.8 ms and 2 tokens 16.3 ms.

From 4 rows on, onnxruntime-web's `MatMulNBits` switches to its prefill kernel (`kMinMForTileOptimization = 4` in
`contrib_ops/webgpu/quantization`), which is built for long prompts and has too few workgroups for a handful of rows.
36 4-bit matmuls with the MLP shapes (2560 x 10240 and back) took 3.7 ms for 1 or 2 rows, 4.7 ms for 3 and 19 ms for
anything from 4 to 64 rows.

A standalone WebGPU kernel that dequantizes each weight block once and applies it to all rows (checked against a CPU
reference) ran 2.2 GB of those weights in 4.9 ms for one row, 7.0 ms for 4 rows and 10.1 ms for 8 rows on the same
GPU. A small-batch kernel of that kind would make a 4- to 8-token check cost 1.4 to 2 times a single step instead of 8
times, but it has to live in onnxruntime.

Prompt-lookup guessing (copy the tokens that followed the last 1-3 tokens wherever they appeared earlier in the prompt
or the answer) was simulated exactly on the saved greedy outputs, since greedy decoding accepts precisely the leading
guesses that match the real continuation. It accepts few tokens on these tasks, so with today's step costs it does
not pay off:

| suite | tokens per step, 1 guess / 2 guesses | speed vs one token per pass, 1 guess / 2 guesses |
|---|---|---|
| greedy parity prompts (chat, code, math, Chinese, tools) | 1.15 / 1.20 | 0.97x / 0.92x |
| BFCL function calling | 1.34 / 1.46 | 1.07x / 1.02x |
| GSM8K | 1.32 / 1.46 | 1.03x / 0.98x |

The app decodes one token per pass.


## Speed (RTX 3090 Ti, Chrome, q4f16, greedy, headless)

| measure | value |
|---|---|
| first visit: download (2.77 GB from a local server into the Cache API) + session init | 42.6 s + 1.6 s |
| later visits: load from cache + session init | 1.3 s + 1.0 s = 2.3 s |
| decode, 28-token prompt / 2.5k-token prompt (256 new tokens) | 87-89 tok/s / 86-89 tok/s |
| time to first token, 28 / 2,518 tokens (first run after load adds ~0.7 s shader compilation) | 92 ms / 1.72 s |
| time to first token, 1k / 4k / 8k / 16k tokens (needle-in-a-haystack prompts) | 0.86 s / 2.91 s / 6.38 s / 15.19 s |
| next chat turn with KV prefix reuse (130-token prompt, 116 cached) | 92 ms vs 149 ms from scratch |

Optimization history (decode tok/s at 2.5k context): naive dynamic KV on ORT 1.30.0: 21 -> ORT 1.31-dev (local-window flash attention): 21 (host-bound) -> static in-place KV cache: 77 -> MatMulNBits accuracy_level 0: 88. accuracy_level 0 also fixed a GSM8K regression (89.3% -> 90.5%) at the cost of ~45% slower prefill.

## Stability (back-to-back generations)

- 683 generations, 122745 tokens in 30.0 min, 0 errors, page errors: 3
- decode tok/s (prompts < 1k tokens): first sixth 78.0, last sixth 86.8
- GPU memory used (whole GPU) after warm-up: 7552 MB -> 7552 MB (max 7552 MB)
- page JS heap after warm-up (run with the earlier runtime, which still referenced the 2.8 GB of downloaded weight bytes; fixed afterwards: heap is 28 MB after GC with the shipped runtime, logits and greedy outputs bit-identical): 2802 MB -> 2844 MB; KV capacity 16384 tokens
- the 3 page 'errors' are the missing favicon (404) and two onnxruntime warnings printed to console.error at session creation

## Transformers.js

The published files with the generated `config.json` (`model_type: "qwen3"`), loaded by Transformers.js 4.3.2 in
Chrome (`AutoModelForCausalLM`, `generate` with greedy decoding, `apply_chat_template` for the prompts), run through
the same suites as the app. Two runtimes: the onnxruntime-web build bundled with Transformers.js 4.3.2
(`1.31.0-dev.20260914`) and the app's `1.31.0-dev.20260918`, swapped in with an import map.

| check | runtime 0918 | bundled runtime |
|---|---|---|
| prompt ids, all suites | identical to Python transformers | identical |
| logits vs this build, 160 positions | mean KL 0.000000, top-1 1.000 | not run |
| greedy parity, 50 prompts | 50/50 identical to this build | 50/50 identical |
| BFCL, 200 cases | 179/200, 200/200 identical outputs | not run |
| GSM8K, 1,319 problems | 1194/1319, 1319/1319 identical outputs | not run |
| needle in a haystack, 12 cases | 12/12, identical outputs | 1k-8k: 9/9 identical; every 16k case fails |
| `pipeline("text-generation")` as in the README, 10 chat prompts | 10/10 identical text | 10/10 identical text |

Decode speed, 256 new tokens, mean of the second and third run (Chrome, RTX 3090 Ti):

| context | this build (static KV cache) | Transformers.js, runtime 0918 | Transformers.js, bundled runtime |
|---|---|---|---|
| short prompt | 91.6 tok/s, first token 0.09 s | 85.9 tok/s, 0.09 s | 82.4 tok/s, 0.09 s |
| 1k tokens | 90.9 tok/s, 0.68 s | 78.6 tok/s, 0.69 s | 62.4 tok/s, 0.88 s |
| 4k tokens | 87.0 tok/s, 2.87 s | 17.9 tok/s, 2.88 s | 17.2 tok/s, 7.29 s |
| 8k tokens | 79.9 tok/s, 6.29 s | 13.1 tok/s, 6.27 s | 12.7 tok/s, 24.89 s |
| 16k tokens | 70.8 tok/s, 15.03 s | 7.8 tok/s, 14.70 s | fails (integer overflow in onnxruntime) |

Transformers.js grows the KV cache at every step, while this app writes into one preallocated buffer per layer.


## Not tested

- **Other GPUs / Mac / Android**: only an RTX 3090 Ti on Windows 11 + Chrome was available. Chrome on this machine exposes only the discrete GPU to WebGPU (forcing the low-power adapter returns no adapter), so the integrated Radeon could not be used.
- **Thinking-on benchmarks**: GSM8K/BFCL were run with thinking off (thinking on/off behaviour is covered by the feature checks).
- **Two app tabs in one browser**: two WebGPU sessions in one Chrome GPU process hit `CreateDescriptorHeap E_OUTOFMEMORY` (device lost) on this machine; one tab per browser ran 1319 + 200 + 683 generations without errors.
