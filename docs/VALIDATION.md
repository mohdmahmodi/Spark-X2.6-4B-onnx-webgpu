# Spark-X2.6-4B browser build: validation report

## Summary

| test | original (transformers) | browser build | pass |
|---|---|---|---|
| Logits in WebGPU: mean KL to bf16 (160 pos.) | INT4 weights in HF: 0.0235 | 0.0238 (adds 0.00030 vs INT4) | PASS |
| Greedy parity: first token same (50 prompts) | INT4 weights in HF: 46/50 | 46/50 | PASS |
| Greedy parity: mean tokens before divergence | INT4 weights in HF: 29.4 | 29.1 | PASS |
| GSM8K | bf16 90.8%, INT4 weights 90.5% | 90.5% | PASS |
| BFCL v3 simple function calling | bf16 89.0%, INT4 weights 89.5% | 89.5% | PASS |
| Needle in a haystack | bf16 100.0%, INT4 weights 100.0% | 100.0% | PASS |
| Features: thinking / multi-turn / tool calls | 6/6 | 8/8 | PASS |
| Stability: 30 min back-to-back | n/a | 0 errors, tok/s 78->87, GPU mem +0 MB | PASS |

Shipped build: `q4f16` on nvidia ampere (Chrome, WebGPU, static KV cache). All browser results come from the real page driven by headless Chrome (tests/run_browser.mjs); the page tokenizes with tokenizers.js + the official chat template, and its prompt ids were checked to equal transformers' on every task.

## 1. Next-token logits vs transformers

20 prompts x 8 positions (after the prompt, then teacher-forcing the bf16 greedy continuation).

| system | mean KL vs bf16 | p95 | max | top-1 vs bf16 | mean KL vs INT4 weights | top-1 vs INT4 |
|---|---|---|---|---|---|---|
| **browser q4f16 (shipped)** | 0.0238 | 0.1157 | 0.392 | 94.4% | 0.00030 | 99.4% |
| transformers fp32 | 0.0001 | 0.0005 | 0.007 | 100.0% | 0.02516 | 95.0% |
| transformers + official INT4 weights | 0.0235 | 0.1122 | 0.352 | 95.0% | 0.00000 | 100.0% |
| same ONNX graph, CPU onnxruntime | 0.0240 | 0.1104 | 0.434 | 94.4% | 0.00022 | 99.4% |
| browser q4 fp32-act, accuracy_level 4 (earlier) | 0.0246 | 0.1187 | 0.462 | 95.0% | 0.00120 | 98.8% |
| browser q4f16, accuracy_level 4 (earlier build) | 0.0287 | 0.1139 | 0.416 | 95.6% | 0.00495 | 98.1% |
| CPU onnxruntime, accuracy_level 4 (earlier) | 0.0229 | 0.1145 | 0.301 | 95.0% | 0.00055 | 98.8% |

KL vs INT4 isolates what the ONNX/WebGPU build adds on top of the official quantization.

WebGPU kernels vs CPU kernels, same graph: mean KL 0.00016, max 0.0059, top-1 agree 100.0%.

## 2. Greedy output parity (50 prompts: chat, code, math, Chinese, tool calls; 128 tokens) vs transformers bf16

| system | first token same | identical 128-token outputs | mean tokens before divergence | median |
|---|---|---|---|---|
| **browser q4f16 (shipped)** | 46/50 | 9/50 | 29.1 | 22 |
| transformers + official INT4 weights | 46/50 | 8/50 | 29.4 | 22 |
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
| **browser q4f16 (shipped)** | 1194/1319 (90.5%) | 13 / 13, p = 1.00 |
| transformers bf16 | 1198/1319 (90.8%) | 37 / 41, p = 0.73 |
| transformers + official INT4 weights | 1194/1319 (90.5%) |  |
| browser q4f16, accuracy_level 4 (earlier build) | 1178/1319 (89.3%) | 35 / 19, p = 0.04 |

## GSM8K, first 200 (diagnostic runs)

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| transformers bf16 | 187/200 (93.5%) | 7 / 8, p = 1.00 |
| transformers + official INT4 weights | 186/200 (93.0%) |  |
| browser q4 fp32-act, accuracy_level 4 (earlier) | 183/200 (91.5%) | 5 / 2, p = 0.45 |
| browser q4f16, accuracy_level 4 (earlier build) | 181/200 (90.5%) | 8 / 3, p = 0.23 |
| browser q4f16 fp32-residual, acc. level 4 (earlier) | 182/200 (91.0%) | 7 / 3, p = 0.34 |

## BFCL v3 simple function calling, first 200 (Spark tool format, SGLang spark25 parsing)

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
| **browser q4f16 (shipped)** | 179/200 (89.5%) | 1 / 1, p = 1.00 |
| transformers bf16 | 178/200 (89.0%) | 5 / 4, p = 1.00 |
| transformers + official INT4 weights | 179/200 (89.5%) |  |
| browser q4f16, accuracy_level 4 (earlier build) | 179/200 (89.5%) | 4 / 4, p = 1.00 |

## Needle in a haystack, 1k-16k tokens, needle at start/middle/end

| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |
|---|---|---|
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

## Not tested

- **Other GPUs / Mac / Android**: only an RTX 3090 Ti on Windows 11 + Chrome was available. Chrome on this machine exposes only the discrete GPU to WebGPU (forcing the low-power adapter returns no adapter), so the integrated Radeon could not be used.
- **Thinking-on benchmarks**: GSM8K/BFCL were run with thinking off (thinking on/off behaviour is covered by the feature checks).
- **Two app tabs in one browser**: two WebGPU sessions in one Chrome GPU process hit `CreateDescriptorHeap E_OUTOFMEMORY` (device lost) on this machine; one tab per browser ran 1319 + 200 + 683 generations without errors.
