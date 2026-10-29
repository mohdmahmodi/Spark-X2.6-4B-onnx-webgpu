---
license: apache-2.0
base_model: XHToken/Spark-X2.6-4B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: onnxruntime
tags:
  - onnx
  - webgpu
  - onnxruntime-web
  - transformers.js
  - in-browser
  - int4
  - spark2_5
language:
  - en
  - zh
---

# Spark-X2.6-4B ONNX for WebGPU

![Spark-X2.6-4B in the browser](assets/hero.png)

[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) converted to ONNX for in-browser inference
with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on WebGPU. It uses the official INT4 weights,
repacked without loss, and shows no measurable accuracy loss against the official INT4 checkpoint run in transformers.

- [Try it in your browser](https://huggingface.co/spaces/mohdmahmodi/Spark-X2.6-WebGPU)
- [Source code and validation](https://github.com/mohdmahmodi/Spark-X2.6-4B-onnx-webgpu)

| | |
|---|---|
| Base model | [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B), 4B parameters |
| Weights | official [Spark-X2.6-4B-INT4](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) GPTQ codes, 4-bit, group size 32 |
| Runtime | onnxruntime-web on WebGPU (browser); native onnxruntime on CPU |
| Size | 2.77 GB (q4f16) or 3.05 GB (q4) |

![Chat app](assets/chat.png)

## Model quality

As reported by XHToken for the original Spark-X2.6-4B, against models of similar size, with thinking on at high
effort. This build runs the official INT4 weights of the same model; the checks below compare it with
that checkpoint.

![Model quality, reported by XHToken](assets/quality.png)

![All 30 benchmarks, reported by XHToken](assets/results.png)

## Accuracy of this build

Measured in Chrome on WebGPU against the original model in transformers, on identical prompts, chat template and
greedy decoding.

![Accuracy](assets/accuracy.png)

| Check | transformers bf16 | transformers, official INT4 | This build |
|---|---|---|---|
| GSM8K, 1,319 problems, thinking off | 90.8% | 91.2% | 90.5% |
| BFCL v3 simple function calling, 200 cases | 89.0% | 89.5% | 89.5% |
| Needle in a haystack, 1k–16k tokens, 12 cases | 12/12 | 12/12 | 12/12 |
| Next-token KL divergence to bf16, 160 positions | 0 | 0.0236 | 0.0238 |
| Greedy parity, 50 prompts: first token identical | 50/50 | 47/50 | 46/50 |

On GSM8K the browser build solves 1,194 problems to the INT4 run's 1,203 and bf16's 1,198; against the INT4 run that is
22 and 13 disagreements (McNemar p = 0.18), within noise. On BFCL they disagree on 2 and 2 cases (p = 1.00). Scores use
thinking off and are not comparable to results measured with thinking on. Method and per-case results are in the
[validation report](https://github.com/mohdmahmodi/Spark-X2.6-4B-onnx-webgpu/blob/main/docs/VALIDATION.md).

![Speed](assets/speed.png)

About 90 tokens per second with a short prompt and 69 at 16k tokens of context. The first token arrives in 92 ms for
a short prompt and in 15 s for a 16k-token prompt. Measured in Chrome on Windows 11 with an NVIDIA RTX 3090 Ti; speed
depends on the GPU.

## Files

| File | Contents |
|---|---|
| `onnx/model_q4f16.onnx` + `_data`, `_data_1` | default: 4-bit weights, fp16 activations, 2.77 GB |
| `onnx/model_q4.onnx` + `_data`, `_data_1` | GPUs without `shader-f16`: 4-bit weights, fp32 activations, 3.05 GB |
| `spark.json` | variants, file lists, build ids, cache shapes |
| `tokenizer.json`, `tokenizer_config.json` | tokenizer, with the official chat template inlined |
| `config.json` | the original model's, with `model_type` set to `qwen3` and the external-data file counts, for Transformers.js |
| `generation_config.json`, `LICENSE` | from the original model |

External data is split into files under 1.9 GB so browsers can fetch them.

## Graph

- **Weights.** Attention and MLP projections are 4-bit `MatMulNBits` with group size 32, taken directly from the
  official GPTQ checkpoint. `out_proj`, the attention gate `g_proj` and the LM head are 8-bit. Input tokens are looked
  up in the same int8 table as the LM head; the official checkpoint keeps a separate bf16 input embedding, and the
  swap measures as lossless (mean KL 0.0003 to bf16 in transformers, GSM8K 1,197 vs 1,198).
- **Operators.** Fused `GroupQueryAttention` (sliding window 512 on 27 of 36 layers, per-layer-type rotary embedding
  including partial rotary), `SkipSimplifiedLayerNormalization`, `GatherBlockQuantized`. `MatMulNBits` uses
  `accuracy_level=0`, which keeps activations in floating point.
- **Inputs.** `input_ids` `[1, S]` int64; `attention_mask` `[1, total]` int64, all ones; and
  `past_key_values.{0..35}.{key,value}` `[1, 4, P, 256]`.
- **Outputs.** `logits` `[1, 1, 131072]` float32, last position only; and `present.*`. Past and present can be bound
  to the same preallocated GPU buffer for an in-place static KV cache.
- **Runtime.** onnxruntime-web `1.31.0-dev.20260918-bc8e7ed75` or newer, imported from `onnxruntime-web/webgpu`.

### Transformers.js

The files load in Transformers.js 4.3 as they are:

```js
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "mohdmahmodi/Spark-X2.6-4B-ONNX-WebGPU", { device: "webgpu", dtype: "q4f16" });
const out = await generator([{ role: "user", content: "Hello!" }], {
  max_new_tokens: 256, do_sample: false, tokenizer_encode_kwargs: { enable_thinking: false },
});
console.log(out[0].generated_text.at(-1).content);
```

It gives the same results as the loader below: identical logits, and identical output on all 50 greedy prompts, all
200 BFCL cases and all 1,319 GSM8K problems. It grows its KV cache at every step, so decoding slows down with context.
In one run on the same machine:

| Context | Transformers.js | Loader below |
|---|---|---|
| short prompt | 86 tok/s | 92 tok/s |
| 1k tokens | 79 tok/s | 91 tok/s |
| 4k tokens | 18 tok/s | 87 tok/s |
| 16k tokens | 7.8 tok/s | 71 tok/s |

The Transformers.js column uses onnxruntime-web `1.31.0-dev.20260918`. Prompts of 16k tokens need that build or newer:
Transformers.js 4.3.2 bundles an older one that fails on them.

### Included loader

The loader in the [source repository](https://github.com/mohdmahmodi/Spark-X2.6-4B-onnx-webgpu) keeps the KV cache in
one preallocated GPU buffer and handles the chat template and streaming:

```js
import { loadSpark } from "./lib/spark.js";

const spark = await loadSpark({ base: "https://huggingface.co/mohdmahmodi/Spark-X2.6-4B-ONNX-WebGPU/resolve/main" });
let reply = "";
for await (const piece of spark.generate([{ role: "user", content: "Hello!" }], { thinking: false })) reply += piece;
```

## Requirements and limitations

- Desktop Chrome or Edge with WebGPU, about 3.5 GB of free GPU memory at 4k tokens of context, and a WebGPU
  `maxBufferSize` of at least 336 MB. A lower storage-buffer binding limit works: onnxruntime binds larger buffers in
  pieces (checked down to the 128 MB default, with identical outputs).
- Tested on Windows 11 with an RTX 3090 Ti only.
- The WASM (CPU) backend of onnxruntime-web cannot load the graph: it has no kernel for `GatherBlockQuantized`.
- Two tabs of the app in one Chrome instance can cause a "device lost" error on Windows; use one tab.

## License

Apache-2.0, the same as the base model. Spark-X2.6-4B is developed by XHToken; this repository contains a format
conversion of their released weights.

Converted and validated by Mohd Mahmodi ([GitHub](https://github.com/mohdmahmodi), [X](https://x.com/mohdmahmodi)).
