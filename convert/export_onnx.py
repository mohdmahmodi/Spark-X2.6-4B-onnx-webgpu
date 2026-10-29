"""Build ONNX graphs for XHToken/Spark-X2.6-4B directly from the safetensors checkpoint.

The graph is written op by op (the way onnxruntime-genai's model builder does it) instead
of tracing torch, so it uses the fused ops ORT's WebGPU EP has fast kernels for:

  GatherBlockQuantized          int8 tied embedding lookup
  SimplifiedLayerNormalization  RMSNorm (fp32 statistics inside the kernel)
  SkipSimplifiedLayerNormalization  residual add + RMSNorm
  MatMulNBits                   4-bit / 8-bit weight-only matmuls
  GroupQueryAttention           RoPE + KV cache append + causal / sliding-window attention

Spark specifics (from modeling_spark.py / config.json):
  * 36 layers, layer_types = 3 x sliding_attention : 1 x full_attention, sliding_window 512
    (HF mask kv_idx > q_idx - 512  ==  GQA local_window_size=512, checked in tests/test_gqa_op.py)
  * sliding: rope theta 1e4, full rotary (256 dims); full: theta 5e6, partial rotary 0.25 (64 dims)
  * rotate_half (non-interleaved) RoPE computed from the absolute cache position
  * 16 query heads, 4 KV heads, head_dim 256, fused q_k_v_proj [q | k | v] -> packed GQA input
  * headwise output gate: attn_out[:, :, h, :] *= sigmoid(g_proj(x))[h]  (g_proj: 2560 -> 16)
  * MLP: down(gelu_erf(gate(x)) * up(x)), 2560 -> 10240
  * tied embeddings: logits = norm(h) @ embedding.T ; no embedding scaling
  * residual stream kept in fp32 by the reference; peak |h| ~700, so fp16 is in range

Variants:
  fp32    unquantized fp32 weights, for checking the graph against transformers
  fp16    unquantized fp16 weights and activations (reference for fp16 range checks)
  q4      4-bit weights, fp32 activations (GPUs without shader-f16)
  q4f16   4-bit weights, fp16 activations (default for WebGPU)

usage: python convert/export_onnx.py q4f16 [--src hf/Spark-X2.6-4B] [--int4 hf/Spark-X2.6-4B-INT4/model.safetensors]
                                          [--weights official|mse|rtn] [--block 32] [--fp32-residual]
                                          [--accuracy-level 0] [--out web/spark/onnx]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from safetensors import safe_open

SRC = Path("hf/Spark-X2.6-4B")
OFFICIAL = Path("hf/Spark-X2.6-4B-INT4/model.safetensors")
CHUNK = 1_900_000_000  # external data file size cap (browsers choke on >2 GB ArrayBuffers)
MAX_POS = 32768        # rotary cache length = max context the web build supports


class Weights:
    def __init__(self, src):
        index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}
        self.where = {k: src / f for k, f in index.items()}

    def __call__(self, name):
        f = self.where[name]
        if f not in self.files:
            self.files[f] = safe_open(str(f), "pt")
        return self.files[f].get_tensor(name).float()


class DataWriter:
    """Streams initializer bytes into <= CHUNK sized external data files."""

    def __init__(self, out, stem):
        self.out, self.stem = out, stem
        self.idx, self.used, self.fh = 0, 0, None
        self.files = []
        for f in out.glob(f"{stem}_data*"):
            f.unlink()

    def write(self, t, raw):
        n = len(raw)
        if self.fh is None or (self.used and self.used + n > CHUNK):
            if self.fh:
                self.fh.close()
                self.idx += 1
            name = f"{self.stem}_data" + (f"_{self.idx}" if self.idx else "")
            self.fh = open(self.out / name, "wb")
            self.files.append(name)
            self.used = 0
        pad = (-self.used) % 64
        self.fh.write(b"\0" * pad)
        self.used += pad
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", self.files[-1]), ("offset", str(self.used)), ("length", str(n))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self.fh.write(raw)
        self.used += n

    def close(self):
        if self.fh:
            self.fh.close()


def _qparams(wb, lo, hi, qmax):
    scale = (hi - lo) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zp = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.round(wb / scale + zp).clamp(0, qmax)
    return q, scale, zp


def quantize(w, bits, block, mode="mse"):
    """Asymmetric block quantization of an HF Linear weight w [N, K] along K.

    mode "rtn": min/max range.  mode "mse": per block, shrink the range over a grid and keep
    the clip that minimizes squared reconstruction error (cheap, no calibration data).
    Returns (q [N, nb, block] uint8 codes, scale [N, nb], zp [N, nb], dequantized [N, K]).
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.to(dev, torch.float32)
    n, k = w.shape
    qmax = 2 ** bits - 1
    wb = w.reshape(n, k // block, block)
    lo0 = wb.amin(-1, keepdim=True).clamp(max=0)
    hi0 = wb.amax(-1, keepdim=True).clamp(min=0)
    q, scale, zp = _qparams(wb, lo0, hi0, qmax)
    if mode == "mse":
        best = ((q - zp) * scale - wb).pow(2).sum(-1, keepdim=True)
        for r in torch.linspace(0.95, 0.55, 9).tolist():
            q2, s2, z2 = _qparams(wb, lo0 * r, hi0 * r, qmax)
            err = ((q2 - z2) * s2 - wb).pow(2).sum(-1, keepdim=True)
            better = err < best
            best = torch.where(better, err, best)
            q, scale, zp = torch.where(better, q2, q), torch.where(better, s2, scale), torch.where(better, z2, zp)
    deq = ((q - zp) * scale).reshape(n, k)
    return q.to(torch.uint8), scale.squeeze(-1), zp.squeeze(-1).to(torch.uint8), deq


class OfficialQuant:
    """XHToken/Spark-X2.6-4B-INT4 (compressed-tensors, llm-compressor GPTQ recipe).

    q_k_v_proj / gate / up / down: 4-bit asymmetric, group 32, int32-packed nibbles along K
    (zero points packed along N); out_proj and lm_head: int8 symmetric per output channel.
    Both map losslessly onto MatMulNBits (int8 -> codes q+128, zero point 128).
    """

    def __init__(self, path):
        self.f = safe_open(str(path), "pt")
        self.keys = set(self.f.keys())

    @staticmethod
    def _nibbles(p, dim):
        p = p.to(torch.int64) & 0xFFFFFFFF
        parts = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim + 1)
        shape = list(p.shape)
        shape[dim] *= 8
        return parts.reshape(shape).to(torch.uint8)

    def get(self, name, block):
        """Returns (codes [N, nb, block], scale [N, nb], zp [N, nb], bits) or None."""
        if name + ".weight_packed" in self.keys:
            q = self._nibbles(self.f.get_tensor(name + ".weight_packed"), 1)
            n, k = q.shape
            scale = self.f.get_tensor(name + ".weight_scale").float()
            zp = self._nibbles(self.f.get_tensor(name + ".weight_zero_point"), 0)[:n]
            assert scale.shape[1] * 32 == k and block == 32, "official INT4 uses group 32"
            return q.reshape(n, k // 32, 32), scale, zp, 4
        if name + ".weight" in self.keys and self.f.get_slice(name + ".weight").get_dtype() == "I8":
            q = self.f.get_tensor(name + ".weight").to(torch.int16)
            n, k = q.shape
            s = self.f.get_tensor(name + ".weight_scale").float().reshape(n, 1).expand(n, k // block)
            codes = (q + 128).to(torch.uint8).reshape(n, k // block, block)
            return codes, s.contiguous(), torch.full((n, k // block), 128, dtype=torch.uint8), 8
        return None

    @staticmethod
    def dequant(codes, scale, zp):
        n, nb, blk = codes.shape
        return ((codes.float() - zp.float()[..., None]) * scale[..., None]).reshape(n, nb * blk)


def fake_quant(w, bits, block, mode):
    return quantize(w, bits, block, mode)[3].cpu()


def pack(q, scale, zp, bits, np_dtype):
    """MatMulNBits / GatherBlockQuantized byte layout: codes [N, nb, block*bits/8],
    scales [N, nb], zero points [N, ceil(nb*bits/8)] (4-bit: two per byte, low nibble first)."""
    n, nb, _ = q.shape
    if bits == 4:
        codes = q[..., 0::2] | (q[..., 1::2] << 4)
        if nb % 2:
            zp = torch.cat([zp, torch.zeros(n, 1, dtype=torch.uint8, device=zp.device)], 1)
        zp = zp[:, 0::2] | (zp[:, 1::2] << 4)
    else:
        codes = q
    return (codes.contiguous().cpu().numpy(), scale.cpu().numpy().astype(np_dtype), zp.contiguous().cpu().numpy())


class Builder:
    def __init__(self, cfg, W, variant, block, weights, fp32_residual, out, key=None, official=OFFICIAL):
        self.cfg, self.W = cfg, W
        self.quant = variant.startswith("q4")
        self.f16 = variant.endswith("16")
        self.adt = TensorProto.FLOAT16 if self.f16 else TensorProto.FLOAT
        self.anp = np.float16 if self.f16 else np.float32
        self.rdt = TensorProto.FLOAT if fp32_residual else self.adt
        self.block = block
        self.weights = weights
        # MatMulNBits accuracy_level: 4 lets kernels quantize activations to int8 (WebGPU DP4A prefill, CPU int8).
        # On WebGPU that raised logit KL vs the official INT4 model 16x (0.0050 vs 0.0003) and cost ~1.2 GSM8K
        # points; level 0 is also faster at decode (88 vs 75 tok/s on an RTX 3090 Ti), slower at prefill.
        self.acc_level = 0
        self.official = OfficialQuant(official) if self.quant and weights == "official" else None
        self.nodes, self.inits = [], []
        self.names = set()
        self.data = DataWriter(out, f"model_{key or variant}.onnx")
        self.consts = {}

    # ---- graph helpers -------------------------------------------------------------
    def uniq(self, name):
        base, i = name, 1
        while name in self.names:
            name = f"{base}_{i}"
            i += 1
        self.names.add(name)
        return name

    def init(self, name, arr):
        arr = np.ascontiguousarray(arr)
        t = TensorProto(name=name, data_type=helper.np_dtype_to_tensor_dtype(arr.dtype), dims=arr.shape)
        raw = arr.tobytes()
        if len(raw) >= 1024:
            self.data.write(t, raw)
        else:
            t.raw_data = raw
        self.inits.append(t)
        return name

    def const(self, value, dtype=np.int64):
        arr = np.array(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self.consts:
            self.consts[key] = self.init(self.uniq(f"const_{arr.dtype.name}_{len(self.consts)}"), arr)
        return self.consts[key]

    def op(self, op_type, inputs, name, n_out=1, domain="", unused=(), **attrs):
        name = self.uniq(name)
        outs = ["" if i in unused else f"{name}/out{i}" if i else f"{name}/out" for i in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, name=name, domain=domain, **attrs))
        return outs[0] if n_out == 1 else outs

    def cast(self, x, to, name):
        return self.op("Cast", [x], name, to=to)

    # ---- layers ----------------------------------------------------------------------
    def qweight(self, hf_name):
        """Block-quantized weight for an HF Linear: (codes, scale, zp, bits)."""
        if self.official:
            got = self.official.get(hf_name, self.block)
            if got:
                return got
        bits = 8 if hf_name == "lm_head" or hf_name.endswith(("out_proj", "g_proj")) else 4
        w = self.W("model.embedding.weight" if hf_name == "lm_head" else hf_name + ".weight")
        q, scale, zp, _ = quantize(w, bits, self.block, self.weights)
        return q, scale, zp, bits

    def linear(self, x, hf_name, name):
        """x @ W.T for HF Linear `hf_name`.

        g_proj (2560 -> 16) is unquantized in the official INT4 recipe (4-bit costs ~0.06 mean KL);
        here it is 8-bit MatMulNBits: the WebGPU fp16 MatMul kernel takes ~70 us for N=16."""
        if self.quant:
            codes, scale, zp, bits = self.qweight(hf_name)
            n, nb, blk = codes.shape
            c, sc, z = pack(codes, scale, zp, bits, self.anp)
            ins = [x, self.init(f"{name}.q{bits}", c), self.init(f"{name}.scales", sc), self.init(f"{name}.zp", z)]
            return self.op("MatMulNBits", ins, name, domain="com.microsoft",
                           K=nb * blk, N=n, bits=bits, block_size=blk, accuracy_level=self.acc_level)
        w = self.W(hf_name + ".weight")
        return self.op("MatMul", [x, self.init(f"{name}.weight_t", w.T.numpy().astype(self.anp))], name)

    def norm(self, x, w, name, skip=None):
        """RMSNorm (optionally fused with the residual add). Returns (normed act, residual)."""
        eps = self.cfg["rms_norm_eps"]
        gamma = self.init(f"{name}.weight", w.numpy().astype(np.float32 if self.rdt == TensorProto.FLOAT else self.anp))
        if skip is None:
            y = self.op("SimplifiedLayerNormalization", [x, gamma], name, axis=-1, epsilon=eps, stash_type=1)
            res = x
        else:
            if self.rdt != self.adt:
                skip = self.cast(skip, self.rdt, f"{name}/skip_cast")
            y, _, _, res = self.op("SkipSimplifiedLayerNormalization", [x, skip, gamma], name, n_out=4, unused=(1, 2),
                                   domain="com.microsoft", epsilon=eps)
        if self.rdt != self.adt:
            y = self.cast(y, self.adt, f"{name}/out_cast")
        return y, res

    def rope_caches(self, kind):
        p = self.cfg["rope_parameters"][kind]
        rot = int(self.cfg["head_dim"] * p["partial_rotary_factor"])
        inv = 1.0 / (p["rope_theta"] ** (np.arange(0, rot, 2, dtype=np.float32) / rot))
        f = np.arange(MAX_POS, dtype=np.float32)[:, None] * inv[None]
        return (self.init(f"cos_cache_{kind}", np.cos(f).astype(self.anp)),
                self.init(f"sin_cache_{kind}", np.sin(f).astype(self.anp)))

    def build(self):
        c, W = self.cfg, self.W
        L, nh, kvh, hd, hid = c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["hidden_size"]
        V = c["vocab_size"]
        kv_t = self.adt
        inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
                  helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"])]
        outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, V])]
        for i in range(L):
            for n in ("key", "value"):
                inputs.append(helper.make_tensor_value_info(f"past_key_values.{i}.{n}", kv_t, ["batch_size", kvh, "past_sequence_length", hd]))
                # own symbol: with a static KV cache present is the max-length buffer, not total_sequence_length
                outputs.append(helper.make_tensor_value_info(f"present.{i}.{n}", kv_t, ["batch_size", kvh, "present_sequence_length", hd]))

        # seqlens_k = total_len - 1 per batch row, total_sequence_length = attention_mask.shape[1]
        # (batch 1 / no left padding: the web runtime never pads)
        shp = self.op("Shape", ["attention_mask"], "/mask/shape")
        total = self.op("Gather", [shp, self.const(1)], "/mask/total", axis=0)
        batch = self.op("Gather", [shp, self.const([0])], "/mask/batch", axis=0)
        last = self.op("Sub", [total, self.const(1)], "/mask/last")
        seqlens = self.cast(self.op("Expand", [self.op("Unsqueeze", [last, self.const([0])], "/mask/unsq"), batch], "/mask/expand"),
                            TensorProto.INT32, "/mask/seqlens_k")
        total32 = self.cast(total, TensorProto.INT32, "/mask/total_i32")

        caches = {k: self.rope_caches(k) for k in ("sliding_attention", "full_attention")}

        if self.quant:
            # Tied embedding: one quantized tensor serves the lm_head MatMulNBits ([V, nb, blk*bits/8])
            # and, through a Reshape to [V, K*bits/8], the GatherBlockQuantized lookup.
            codes, scale, zp, bits = self.qweight("lm_head")
            head_blk = codes.shape[2]
            c_, sc, z = pack(codes, scale, zp, bits, self.anp)
            hq = self.init(f"lm_head.q{bits}", c_)
            hs, hz = self.init("lm_head.scales", sc), self.init("lm_head.zp", z)
            flat = self.op("Reshape", [hq, self.const([V, hid * bits // 8])], "/embedding/view")
            x = self.op("GatherBlockQuantized", [flat, "input_ids", hs, hz], "/embedding", domain="com.microsoft",
                        bits=bits, block_size=head_blk, gather_axis=0, quantize_axis=1)
        else:
            emb = W("model.embedding.weight")
            x = self.op("Gather", [self.init("embedding.weight", emb.numpy().astype(self.anp)), "input_ids"], "/embedding", axis=0)
        if self.rdt != self.adt:
            x = self.cast(x, self.rdt, "/embedding/cast")

        residual, pending = x, None
        for i in range(L):
            p = f"model.layers.{i}."
            kind = c["layer_types"][i]
            xn, residual = self.norm(residual, W(p + "input_layernorm.weight"), f"/l{i}/input_norm", skip=pending)
            qkv = self.linear(xn, p + "self_attn.q_k_v_proj", f"/l{i}/qkv_proj")
            gate = self.linear(xn, p + "self_attn.g_proj", f"/l{i}/g_proj")
            cos, sin = caches[kind]
            attn, pk, pv = self.op(
                "GroupQueryAttention",
                [qkv, "", "", f"past_key_values.{i}.key", f"past_key_values.{i}.value", seqlens, total32, cos, sin],
                f"/l{i}/attn", n_out=3, domain="com.microsoft",
                num_heads=nh, kv_num_heads=kvh, scale=float(hd ** -0.5), do_rotary=1, rotary_interleaved=0,
                local_window_size=c["sliding_window"] if kind == "sliding_attention" else -1,
            )
            self.nodes[-1].output[1] = f"present.{i}.key"
            self.nodes[-1].output[2] = f"present.{i}.value"
            heads = self.op("Reshape", [attn, self.const([0, 0, nh, hd])], f"/l{i}/attn/split_heads")
            g = self.op("Unsqueeze", [self.op("Sigmoid", [gate], f"/l{i}/gate/sigmoid"), self.const([-1])], f"/l{i}/gate/unsq")
            gated = self.op("Reshape", [self.op("Mul", [heads, g], f"/l{i}/gate/mul"), self.const([0, 0, nh * hd])], f"/l{i}/attn/merge_heads")
            o = self.linear(gated, p + "self_attn.out_proj", f"/l{i}/out_proj")

            xn, residual = self.norm(residual, W(p + "post_attention_layernorm.weight"), f"/l{i}/post_norm", skip=o)
            up = self.linear(xn, p + "mlp.up_proj", f"/l{i}/up_proj")
            act = self.op("Gelu", [self.linear(xn, p + "mlp.gate_proj", f"/l{i}/gate_proj")], f"/l{i}/gelu", approximate="none")
            pending = self.linear(self.op("Mul", [act, up], f"/l{i}/mlp_mul"), p + "mlp.down_proj", f"/l{i}/down_proj")
            print(f"layer {i} ({kind})", flush=True)

        # only the last position feeds the LM head: prefill never materializes [S, 131072]
        res_last = self.op("Slice", [residual, self.const([-1]), self.const([2**62]), self.const([1])], "/final/slice_res")
        mlp_last = self.op("Slice", [pending, self.const([-1]), self.const([2**62]), self.const([1])], "/final/slice_mlp")
        xn, _ = self.norm(res_last, W("model.norm.weight"), "/final/norm", skip=mlp_last)
        if self.quant:
            logits = self.op("MatMulNBits", [xn, hq, hs, hz], "/lm_head", domain="com.microsoft",
                             K=hid, N=V, bits=bits, block_size=head_blk, accuracy_level=self.acc_level)
        else:
            logits = self.op("MatMul", [xn, self.init("lm_head.weight_t", emb.T.numpy().astype(self.anp))], "/lm_head")
        if self.adt != TensorProto.FLOAT:
            logits = self.cast(logits, TensorProto.FLOAT, "/lm_head/cast")
        self.nodes.append(helper.make_node("Identity", [logits], ["logits"], name="/logits"))
        self.data.close()

        graph = helper.make_graph(self.nodes, "spark_x2_6", inputs, outputs, self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                                  producer_name="spark-web-model")
        model.ir_version = 10
        return model, self.data.files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["fp32", "fp16", "q4", "q4f16"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--weights", choices=["official", "mse", "rtn"], default="official",
                    help="official: XHToken INT4 (GPTQ) codes; mse/rtn: quantize the bf16 checkpoint here")
    ap.add_argument("--fp32-residual", action="store_true")
    ap.add_argument("--src", type=Path, default=SRC, help="original checkpoint folder")
    ap.add_argument("--int4", type=Path, default=OFFICIAL, help="official INT4 checkpoint (model.safetensors)")
    ap.add_argument("--out", default="web/spark/onnx")
    ap.add_argument("--accuracy-level", type=int, default=0, help="MatMulNBits accuracy_level (4 allows int8 activations: faster prefill, less accurate)")
    ap.add_argument("--tag", default="", help="suffix for the variant name, e.g. _r32 for an --fp32-residual build")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.src / "config.json").read_text())
    key = a.variant + a.tag
    b = Builder(cfg, Weights(a.src), a.variant, a.block, a.weights, a.fp32_residual, out, key, official=a.int4)
    b.acc_level = a.accuracy_level
    with torch.no_grad():
        model, files = b.build()
    name = f"model_{key}.onnx"
    onnx.save_model(model, str(out / name))
    sizes = {f: (out / f).stat().st_size for f in files}
    print("saved", out / name, {k: f"{v / 1e9:.2f} GB" for k, v in sizes.items()})
    meta_path = out.parent / "spark.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ext_len = lambda t: int(next(e.value for e in t.external_data if e.key == "length")) if t.external_data else t.ByteSize()  # noqa: E731
    meta.setdefault("variants", {})[key] = {
        "file": f"onnx/{name}",
        "external_data": [f"onnx/{f}" for f in files],
        "block_size": a.block,
        "weights": a.weights if a.variant.startswith("q") else a.variant,
        "activations": "float16" if a.variant.endswith("16") else "float32",
        "fp32_residual": a.fp32_residual,
        "max_tensor_bytes": max(map(ext_len, model.graph.initializer)),  # must fit one WebGPU buffer (maxBufferSize)
        "bytes": sum(sizes.values()),
        # content id of this build: the page versions its download cache with it
        "build": hashlib.sha256((out / name).read_bytes() + b"".join(str(sizes[f]).encode() for f in files)).hexdigest()[:16],
    }
    meta.update(model=a.src.name, layers=cfg["num_hidden_layers"], kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                max_position=MAX_POS, eos_token_id=cfg["eos_token_id"],
                intermediate_size=cfg["intermediate_size"])  # sizes the prefill pieces on small-binding GPUs
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
