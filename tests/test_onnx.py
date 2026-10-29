"""Check an exported graph against the transformers reference (run tests/reference.py first).

For each reference prompt:
  * one-shot prefill of the whole prompt -> last-token logits
  * chunked prefill + single-token decode through the KV cache, comparing logits at every
    saved position (the long prompt crosses the 512 sliding window many times)
  * greedy generation, compared token by token with transformers' greedy output

usage: python tests/test_onnx.py q4f16 [--provider cpu|cuda]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

WEB = Path(os.environ.get("SPARK_WEB", "web/spark"))  # SPARK_WEB=export/onnx_ref for the fp32/fp16 graphs
REF = Path("export/reference")


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def compare(a, b):
    """a: reference logits, b: onnx logits (1D)."""
    pa, pb = softmax(a.astype(np.float64)), softmax(b.astype(np.float64))
    kl = float((pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum())
    return float(np.abs(a - b).max()), kl, int(a.argmax() == b.argmax())


class Runner:
    def __init__(self, variant, provider):
        meta = json.loads((WEB / "spark.json").read_text())
        self.meta = meta
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        t = time.time()
        self.sess = ort.InferenceSession(str(WEB / meta["variants"][variant]["file"]), so, providers=providers)
        print(f"session load {time.time() - t:.1f}s on {self.sess.get_providers()[0]}")
        self.names = [o.name for o in self.sess.get_outputs()]
        self.kv_dtype = np.float16 if self.sess.get_inputs()[2].type == "tensor(float16)" else np.float32
        self.reset()

    def reset(self):
        m = self.meta
        self.past = {f"past_key_values.{i}.{n}": np.zeros((1, m["kv_heads"], 0, m["head_dim"]), self.kv_dtype)
                     for i in range(m["layers"]) for n in ("key", "value")}
        self.pos = 0

    def step(self, ids):
        t = self.pos + len(ids)
        res = dict(zip(self.names, self.sess.run(None, {
            "input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, t), np.int64), **self.past})))
        self.past = {"past_key_values." + k[8:]: v for k, v in res.items() if k.startswith("present.")}
        self.pos = t
        return res["logits"][0, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--provider", default="cpu")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("hf/Spark-X2.6-4B")
    summary = json.loads((REF / "summary.json").read_text(encoding="utf-8"))
    r = Runner(a.variant, a.provider)
    report = {}
    for name in ("short", "long"):
        ref = np.load(REF / f"{name}.npz")
        ids, pos, ref_logits = ref["ids"].tolist(), ref["pos"].tolist(), ref["logits"]

        r.reset()
        t0 = time.time()
        one_shot = compare(ref_logits[-1], r.step(ids))
        prefill_s = time.time() - t0

        r.reset()
        stats, prev = [], 0
        for p, rl in zip(pos, ref_logits):
            stats.append(compare(rl, r.step(ids[prev:p + 1])))
            prev = p + 1
        st = np.array(stats)

        gen = []
        r.reset()
        logits = r.step(ids)
        t1 = time.time()
        for _ in range(len(summary[name]["greedy"])):
            nxt = int(logits.argmax())
            gen.append(nxt)
            if nxt == r.meta["eos_token_id"]:
                break
            logits = r.step([nxt])
        dec_s = (time.time() - t1) / max(1, len(gen) - 1)
        want = summary[name]["greedy"]
        same = next((i for i, (x, y) in enumerate(zip(gen, want)) if x != y), min(len(gen), len(want)))
        report[name] = {
            "tokens": len(ids),
            "one_shot_last": {"max_abs": round(one_shot[0], 4), "kl": round(one_shot[1], 6), "top1": one_shot[2]},
            "incremental": {"positions": len(pos), "max_abs": round(float(st[:, 0].max()), 4),
                            "mean_kl": round(float(st[:, 1].mean()), 6), "max_kl": round(float(st[:, 1].max()), 6),
                            "top1_agree": f"{int(st[:, 2].sum())}/{len(pos)}"},
            "greedy_match": f"{same}/{len(want)} tokens identical",
            "prefill_s": round(prefill_s, 2), "decode_tok_s": round(1 / dec_s, 2),
            "text": tok.decode(gen),
        }
        print(json.dumps({name: report[name]}, indent=1, ensure_ascii=False), flush=True)
    out = REF.parent / f"check_{a.variant}_{a.provider}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
