// Spark-X2.6-4B on onnxruntime-web (native WebGPU EP).
//
// The graph (convert/export_onnx.py) takes input_ids + attention_mask + past KV and returns the
// last-position logits + present KV. present.* stays on the GPU (preferredOutputLocation
// "gpu-buffer") and is fed straight back as past_key_values.* on the next step, so KV never
// round-trips through JS; only the logits row (131072 floats) is read back per token.
import * as ort from "onnxruntime-web/webgpu";
import { loadChat } from "./chat.js";

const CACHE_PREFIX = "spark-x2.6-4b-";

// Downloads are cached per build id (spark.json), so a re-exported model is never served from a stale
// cache; caches of other builds are deleted to give the disk space back.
async function openCache(build) {
  try {
    const name = CACHE_PREFIX + build;
    for (const k of await caches.keys()) if (k.startsWith(CACHE_PREFIX) && k !== name) await caches.delete(k);
    return await caches.open(name);
  } catch {
    return null; // Cache API unavailable (e.g. insecure origin): plain fetch
  }
}

async function fetchBytes(url, onBytes, cache) {
  let res = cache && (await cache.match(url));
  const hit = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  // store into the cache while reading (both tee branches drain together, nothing piles up)
  const putting = cache && !hit ? cache.put(url, res.clone()).catch((e) => console.warn("cache put failed", url, e)) : null;
  const reader = res.body.getReader();
  const buf = total ? new Uint8Array(total) : null;
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf) buf.set(value, got);
    else parts.push(value);
    got += value.length;
    onBytes?.(value.length);
  }
  await putting;
  if (buf) return buf;
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) out.set(p, (o += p.length) - p.length);
  return out;
}

function argmax(x) {
  let best = 0;
  let max = x[0];
  for (let i = 1; i < x.length; i++) if (x[i] > max) max = x[(best = i)];
  return best;
}

function sample(logits, temperature, topP, topK) {
  if (!temperature) return argmax(logits);
  const max = logits[argmax(logits)];
  // candidates within ~e^-12 of the max (rest is < 1e-5 relative mass each)
  const cut = max - 12 * temperature;
  const cand = [];
  for (let i = 0; i < logits.length; i++) if (logits[i] >= cut) cand.push(i);
  cand.sort((a, b) => logits[b] - logits[a]);
  const k = topK > 0 ? Math.min(topK, cand.length) : cand.length;
  const p = new Float64Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) sum += p[i] = Math.exp((logits[cand[i]] - max) / temperature);
  let n = 0;
  for (let acc = 0; n < k && acc < topP * sum; n++) acc += p[n];
  let r = Math.random() * p.subarray(0, n).reduce((a, b) => a + b, 0);
  for (let i = 0; i < n; i++) if ((r -= p[i]) <= 0) return cand[i];
  return cand[0];
}

export async function webgpuInfo() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const info = adapter.info || {};
  return {
    adapter,
    f16: adapter.features.has("shader-f16"),
    maxBuffer: adapter.limits.maxBufferSize,
    maxBinding: Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize),
    name: [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown",
  };
}

// Static KV cache: one [1, kv_heads, capacity, head_dim] GPU buffer per layer for K and V, bound
// as both past_key_values.* (input) and present.* (output). GroupQueryAttention sees past == present
// and switches to its in-place path (fused split + rotary + KV write + flash attention), so decode
// does no per-step allocation or past->present copy.
// Capacity follows the request (prompt + max new tokens, rounded up to 1024) and shrinks again after a
// long prompt, so GPU memory tracks the current chat (~147 KB per token). Decode speed does not depend
// on it. Resizing copies the kept prefix on the GPU, so prefix reuse survives it.
class StaticKV {
  constructor(device, names, kvHeads, headDim, dataType) {
    Object.assign(this, { device, names, kvHeads, headDim, dataType, capacity: 0, tensors: {} });
    this.elem = dataType === "float16" ? 2 : 4;
  }

  ensure(length) {
    const cap = Math.ceil(length / 1024) * 1024;
    if (cap <= this.capacity && cap > this.capacity / 2) return;
    const { device, kvHeads, headDim, elem } = this;
    const old = this.tensors;
    const keep = Math.min(this.capacity, cap) * headDim * elem; // bytes per head carried over
    const enc = device.createCommandEncoder();
    const next = {};
    for (const n of this.names) {
      const buffer = device.createBuffer({
        size: kvHeads * cap * headDim * elem,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      if (old[n] && keep)
        for (let h = 0; h < kvHeads; h++)
          enc.copyBufferToBuffer(old[n].gpuBuffer, h * this.capacity * headDim * elem, buffer, h * cap * headDim * elem, keep);
      next[n] = ort.Tensor.fromGpuBuffer(buffer, { dataType: this.dataType, dims: [1, kvHeads, cap, headDim] });
    }
    device.queue.submit([enc.finish()]);
    for (const t of Object.values(old)) t.gpuBuffer.destroy();
    this.tensors = next;
    this.capacity = cap;
  }

  feeds() {
    return Object.fromEntries(this.names.map((n) => [`past_key_values.${n}`, this.tensors[n]]));
  }

  fetches() {
    return Object.fromEntries([["logits", null], ...this.names.map((n) => [`present.${n}`, this.tensors[n]])]);
  }
}

export async function loadSpark({ base = "./spark", variant, staticKV = true, onProgress, onDeviceLost } = {}) {
  const t0 = performance.now();
  const gpu = await webgpuInfo();
  if (!gpu)
    throw new Error("WebGPU is not available. Use a recent Chrome or Edge (113+) on desktop, or Chrome 121+ on Android; on Linux enable chrome://flags/#enable-unsafe-webgpu.");
  const metaRes = await fetch(`${base}/spark.json`, { cache: "no-cache" });
  if (!metaRes.ok) throw new Error(`Model files not found at ${base} (HTTP ${metaRes.status}).`);
  const meta = await metaRes.json();
  variant ||= gpu.f16 && meta.variants.q4f16 ? "q4f16" : "q4";
  const v = meta.variants[variant];
  if (!v) throw new Error(`variant ${variant} not exported (have: ${Object.keys(meta.variants)})`);
  const kvType = v.activations ?? (variant.endsWith("16") ? "float16" : "float32");
  if (kvType === "float16" && !gpu.f16) throw new Error("This GPU has no shader-f16; use the q4 variant (?variant=q4).");
  // Every weight tensor is one GPU buffer, so the largest (the 335 MB int8 embedding / lm_head) must fit in
  // maxBufferSize. maxStorageBufferBindingSize may be smaller: onnxruntime binds a larger buffer in several
  // pieces (checked down to 128 MiB, the WebGPU default, with identical outputs).
  if (v.max_tensor_bytes && gpu.maxBuffer < v.max_tensor_bytes)
    throw new Error(`This GPU allows ${(gpu.maxBuffer / 1e6).toFixed(0)} MB per buffer; the model needs ${(v.max_tensor_bytes / 1e6).toFixed(0)} MB.`);

  const [chat, gen] = await Promise.all([loadChat(base), fetch(`${base}/generation_config.json`).then((r) => r.json())]);

  let done = 0;
  const total = v.bytes;
  const tick = (n) => onProgress?.({ loaded: (done += n), total });
  const cache = await openCache(v.build || String(v.bytes));
  const kvNames = [];
  for (let i = 0; i < meta.layers; i++) kvNames.push(`${i}.key`, `${i}.value`);
  // The downloaded bytes (2.8 GB) live only inside this function: once ORT has uploaded the weights to
  // the GPU nothing references them, so the page does not keep a second copy of the model in RAM.
  const { session, tDownload } = await (async () => {
    const [graph, ...chunks] = await Promise.all([
      fetchBytes(`${base}/${v.file}`, null, cache),
      ...v.external_data.map((f) => fetchBytes(`${base}/${f}`, tick, cache)),
    ]);
    const tDownload = performance.now();
    ort.env.webgpu.powerPreference = "high-performance";
    ort.env.webgpu.adapter = gpu.adapter;
    const preferredOutputLocation = { logits: "cpu" };
    for (const n of kvNames) preferredOutputLocation[`present.${n}`] = "gpu-buffer";
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
      externalData: v.external_data.map((f, i) => ({ path: f.split("/").pop(), data: chunks[i] })),
      preferredOutputLocation,
    });
    return { session, tDownload };
  })();
  const tReady = performance.now();

  const emptyKV = () =>
    new ort.Tensor(kvType, kvType === "float16" ? new Uint16Array(0) : new Float32Array(0), [1, meta.kv_heads, 0, meta.head_dim]);
  const eos = new Set([gen.eos_token_id ?? meta.eos_token_id].flat());
  // A long prompt goes in pieces small enough that every prefill activation fits one storage binding (phones report
  // 128 MiB); the largest is the MLP intermediate, [S, intermediate_size]. onnxruntime-web's (Skip)LayerNorm kernels
  // only reach the first binding of a split buffer, so without pieces the fp32 hidden state of a 16k prompt (168 MB)
  // came out as zeros, without an error. With a 2 GB binding limit (Windows desktops) a prompt is never split.
  const actBytes = (meta.intermediate_size ?? 0) * (kvType === "float16" ? 2 : 4);
  const prefillChunk = actBytes ? Math.max(256, Math.floor(gpu.maxBinding / actBytes / 256) * 256) : Infinity;
  const device = await ort.env.webgpu.device;
  const kv = staticKV ? new StaticKV(device, kvNames, meta.kv_heads, meta.head_dim, kvType) : null;
  // Token ids whose K/V currently sit at positions 0..n-1 of the static cache. A new prompt that
  // starts with them (the next turn of a chat) only prefills the new suffix.
  let cached = [];
  // A lost device (driver reset, GPU memory exhausted, tab backgrounded on some mobiles) cannot be
  // recovered in place: surface it so the page can tell the user to reload.
  let lost = null;
  device.lost.then((info) => {
    lost = info;
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
    onDeviceLost?.(info);
  });

  // Token-level loop: yields each new token id. `forced` teacher-forces the given tokens (the
  // sampled one is ignored) and `onLogits(step, Float32Array)` sees every next-token distribution;
  // both are for the test harness. Stops at EOS, maxNewTokens or signal.
  async function* generateIds(prompt, opts = {}) {
    const {
      maxNewTokens = 2048,
      temperature = gen.temperature ?? 1.0,
      topP = gen.top_p ?? 0.95,
      topK = gen.top_k ?? -1,
      signal,
      stats = {},
      forced,
      onLogits,
      reuse = true,
      chunk = prefillChunk,
    } = opts;
    const start = performance.now();
    if (lost) throw new Error(`WebGPU device was lost (${lost.reason}: ${lost.message}); reload the page.`);
    if (prompt.length + maxNewTokens > meta.max_position) throw new Error(`context limit ${meta.max_position} tokens`);
    Object.assign(stats, { promptTokens: prompt.length, newTokens: 0, ttftMs: 0, prefillMs: 0, decodeTokS: 0, done: false });

    kv?.ensure(prompt.length + maxNewTokens);
    let past = kv ? null : Object.fromEntries(kvNames.map((n) => [`past_key_values.${n}`, emptyKV()]));
    let pos = 0;
    if (kv && reuse) {
      const limit = Math.min(cached.length, prompt.length - 1); // feed at least one token for logits
      while (pos < limit && cached[pos] === prompt[pos]) pos++;
    }
    cached = prompt.slice(0, pos);
    stats.reusedTokens = pos;
    let input = prompt.slice(pos);
    let tFirst = 0;
    // one forward pass over `ids`, appending their K/V to the cache
    const forward = async (ids) => {
      const total = pos + ids.length;
      const feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]),
      };
      let out;
      if (kv) {
        out = await session.run({ ...feeds, ...kv.feeds() }, kv.fetches());
      } else {
        out = await session.run({ ...feeds, ...past });
        for (const tsr of Object.values(past)) if (tsr.location === "gpu-buffer") tsr.dispose();
        past = Object.fromEntries(kvNames.map((n) => [`past_key_values.${n}`, out[`present.${n}`]]));
      }
      pos = total;
      if (kv) cached.push(...ids);
      return out;
    };
    try {
      for (let step = 0; step < maxNewTokens; step++) {
        if (signal?.aborted) break;
        const t = performance.now();
        for (; input.length > chunk; input = input.slice(chunk)) (await forward(input.slice(0, chunk))).logits.dispose?.();
        const out = await forward(input);
        onLogits?.(step, out.logits.data);
        let next = sample(out.logits.data, temperature, topP, topK);
        out.logits.dispose?.();
        if (forced) {
          if (step >= forced.length) break;
          next = forced[step];
        }
        if (step === 0) {
          tFirst = performance.now();
          stats.prefillMs = tFirst - t;
          stats.ttftMs = tFirst - start;
        }
        stats.newTokens = step + 1;
        if (step > 0) stats.decodeTokS = (step * 1000) / (performance.now() - tFirst);
        if (eos.has(next) && !forced) break;
        yield next;
        input = [next];
      }
    } finally {
      if (past) for (const tsr of Object.values(past)) if (tsr.location === "gpu-buffer") tsr.dispose();
      stats.totalMs = performance.now() - start;
      stats.done = true;
    }
  }

  // Chat-level streaming: yields text pieces, holding back an incomplete UTF-8 character.
  async function* generate(messages, opts = {}) {
    const { thinking = true, effort = "high", tools } = opts;
    let pending = [];
    for await (const id of generateIds(chat.encodeChat(messages, { thinking, effort, tools }), opts)) {
      pending.push(id);
      const text = chat.decode(pending);
      if (!text.endsWith("�")) {
        pending = [];
        yield text;
      }
    }
  }

  return {
    model: meta.model ?? null,
    session,
    chat,
    generate,
    generateIds,
    variant,
    staticKV: !!kv,
    kvCapacity: () => kv?.capacity ?? 0,
    gpu: gpu.name,
    loadMs: { download: tDownload - t0, init: tReady - tDownload, total: tReady - t0 },
  };
}
