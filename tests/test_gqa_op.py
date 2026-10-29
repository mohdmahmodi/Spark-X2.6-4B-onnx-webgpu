"""Check com.microsoft GroupQueryAttention semantics against the Spark reference math.

Covers what the export relies on: packed QKV input, do_rotary with full rotary (sliding
layers, theta 1e4) and partial rotary (full layers, 64 of 256 dims, theta 5e6),
local_window_size=512 vs the HF sliding mask (kv_idx > q_idx - 512), prefill then decode.
Saves the graphs so the same cases can be replayed on WebGPU.

usage: python tests/test_gqa_op.py
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

NH, KVH, HD, WIN = 16, 4, 256, 512
OUT = Path("export/optest")


def rope_cache(theta, rot_dim, n):
    inv = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    f = np.arange(n, dtype=np.float32)[:, None] * inv[None]
    return np.cos(f), np.sin(f)


def rope_ref(x, pos, theta, rot_dim):
    # x [S, H, D]; mirrors apply_rotary_pos_emb + rotate_half in modeling_spark.py
    c, s = rope_cache(theta, rot_dim, int(pos.max()) + 1)
    c = np.concatenate([c, c], -1)[pos][:, None]
    s = np.concatenate([s, s], -1)[pos][:, None]
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    h = rot_dim // 2
    rot = np.concatenate([-xr[..., h:], xr[..., :h]], -1)
    return np.concatenate([xr * c + rot * s, xp], -1)


def attn_ref(qkv, theta, rot_dim, window, pos0, k_past, v_past):
    s = qkv.shape[0]
    q = qkv[:, : NH * HD].reshape(s, NH, HD)
    k = qkv[:, NH * HD: (NH + KVH) * HD].reshape(s, KVH, HD)
    v = qkv[:, (NH + KVH) * HD:].reshape(s, KVH, HD)
    pos = np.arange(pos0, pos0 + s)
    q, k = rope_ref(q, pos, theta, rot_dim), rope_ref(k, pos, theta, rot_dim)
    k = np.concatenate([k_past, k], 0)
    v = np.concatenate([v_past, v], 0)
    kpos = np.arange(k.shape[0])
    diff = pos[:, None] - kpos[None]
    ok = diff >= 0
    if window:
        ok &= diff < window
    rep = NH // KVH
    kk, vv = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    w = np.einsum("shd,thd->hst", q, kk) / np.sqrt(HD)
    w = np.where(ok[None], w, -np.inf)
    w = np.exp(w - w.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    o = np.einsum("hst,thd->shd", w, vv).reshape(s, NH * HD)
    return o, k, v


def build(theta, rot_dim, window, dtype, max_pos=4096):
    et = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    c, s = rope_cache(theta, rot_dim, max_pos)
    inits = [helper.make_tensor("cos", et, c.shape, c.astype(dtype).tobytes(), raw=True),
             helper.make_tensor("sin", et, s.shape, s.astype(dtype).tobytes(), raw=True)]
    node = helper.make_node(
        "GroupQueryAttention",
        ["qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq", "cos", "sin"],
        ["out", "present_key", "present_value"], domain="com.microsoft",
        num_heads=NH, kv_num_heads=KVH, scale=float(HD ** -0.5), do_rotary=1,
        rotary_interleaved=0, local_window_size=window if window else -1,
    )
    g = helper.make_graph(
        [node], "gqa",
        [helper.make_tensor_value_info("qkv", et, [1, "S", (NH + 2 * KVH) * HD]),
         helper.make_tensor_value_info("past_key", et, [1, KVH, "P", HD]),
         helper.make_tensor_value_info("past_value", et, [1, KVH, "P", HD]),
         helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
         helper.make_tensor_value_info("total_seq", TensorProto.INT32, [])],
        [helper.make_tensor_value_info("out", et, [1, "S", NH * HD]),
         helper.make_tensor_value_info("present_key", et, [1, KVH, "T", HD]),
         helper.make_tensor_value_info("present_value", et, [1, KVH, "T", HD])],
        inits,
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])


CASES = {
    "sliding": (10000.0, 256, WIN),
    "full_partial": (5_000_000.0, 64, 0),
}


def run(dtype=np.float32, provider="CPUExecutionProvider"):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tol = 2e-3 if dtype == np.float32 else 3e-2
    ok_all = True
    for name, (theta, rot, win) in CASES.items():
        model = build(theta, rot, win, dtype)
        path = OUT / f"gqa_{name}_{np.dtype(dtype).name}.onnx"
        onnx.save(model, path)
        sess = ort.InferenceSession(str(path), providers=[provider])
        kp = np.zeros((0, KVH, HD), np.float32)
        vp = np.zeros((0, KVH, HD), np.float32)
        past_k = np.zeros((1, KVH, 0, HD), dtype)
        past_v = np.zeros((1, KVH, 0, HD), dtype)
        pos = 0
        # prefill longer than the window, then a few single-token decode steps
        for step, s in enumerate([700, 1, 1, 37, 1]):
            qkv = rng.standard_normal((s, (NH + 2 * KVH) * HD)).astype(np.float32)
            ref, kp, vp = attn_ref(qkv, theta, rot, win, pos, kp, vp)
            t = pos + s
            out, past_k, past_v = sess.run(None, {
                "qkv": qkv[None].astype(dtype), "past_key": past_k, "past_value": past_v,
                "seqlens_k": np.array([t - 1], np.int32), "total_seq": np.array(t, np.int32),
            })
            err = np.abs(out[0].astype(np.float32) - ref).max()
            kerr = np.abs(past_k[0].transpose(1, 0, 2).astype(np.float32) - kp).max()
            good = err < tol and kerr < tol
            ok_all &= good
            print(f"{name:13s} step {step} s={s:3d} total={t:4d} out_err={err:.2e} present_k_err={kerr:.2e} {'OK' if good else 'FAIL'}")
            pos = t
    return ok_all


if __name__ == "__main__":
    dt = np.float16 if "fp16" in sys.argv else np.float32
    sys.exit(0 if run(dt) else 1)
