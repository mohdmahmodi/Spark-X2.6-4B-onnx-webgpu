"""Measure how each weight-quantization choice moves the logits (fake-quant in torch, GPU).

Loads the fp32 transformers model, swaps in dequantized weights, and reports mean KL to
the unquantized model over every position of the reference prompts. Used to pick the
export settings; the ONNX graphs are then re-checked with tests/test_onnx.py.

usage: python convert/quant_eval.py
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, "convert")
from export_onnx import SRC, OfficialQuant, fake_quant  # noqa: E402

REF = "export/reference"


@torch.no_grad()
def main():
    model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    ids = [torch.tensor(np.load(f"{REF}/{n}.npz")["ids"])[None].cuda() for n in ("short", "long")]
    base = [F.log_softmax(model(x, use_cache=False).logits[0].float(), -1) for x in ids]
    orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    def kl():
        out = []
        for x, b in zip(ids, base):
            lp = F.log_softmax(model(x, use_cache=False).logits[0].float(), -1)
            out.append((b.exp() * (b - lp)).sum(-1))
        k = torch.cat(out)
        return f"mean KL {k.mean():.4f}  p95 {k.quantile(0.95):.4f}  max {k.max():.3f}"

    official = OfficialQuant("hf/Spark-X2.6-4B-INT4/model.safetensors")

    def apply(rule):
        for n, p in model.named_parameters():
            w = orig[n]
            if w.dim() == 2:
                bits, mode, block = rule(n)
                if mode == "official":
                    src = "lm_head" if "embedding" in n else n[: -len(".weight")]
                    got = official.get(src, 32)
                    w = w if got is None else OfficialQuant.dequant(*got[:3])
                elif bits is not None:
                    w = fake_quant(w, bits, block, mode)
            p.copy_(w.to(p.device))

    linear = ("q_k_v_proj", "g_proj", "out_proj", "gate_proj", "up_proj", "down_proj")

    def layer_of(n):
        return int(n.split(".")[2]) if ".layers." in n else -1

    configs = {
        "q4 rtn b32, embed q4": lambda n: (4, "rtn", 32),
        "q4 mse b32, embed q4": lambda n: (4, "mse", 32),
        "q4 mse b32, embed fp": lambda n: (None, None, None) if "embedding" in n else (4, "mse", 32),
        "q4 mse b32, embed q8": lambda n: (8, "rtn", 32) if "embedding" in n else (4, "mse", 32),
        "q4 mse b64, embed q8": lambda n: (8, "rtn", 32) if "embedding" in n else (4, "mse", 64),
        "q4 mse b128, embed q8": lambda n: (8, "rtn", 32) if "embedding" in n else (4, "mse", 128),
        # XHToken INT4 recipe shape: out_proj 8-bit, g_proj unquantized
        "mixed: o q8, g fp, embed q8": lambda n: (None, None, None) if "g_proj" in n else
            (8, "rtn", 32) if ("embedding" in n or "out_proj" in n) else (4, "mse", 32),
        "mixed: o q8, g fp, embed q4": lambda n: (None, None, None) if "g_proj" in n else
            (8, "rtn", 32) if "out_proj" in n else (4, "mse", 32),
        # the published XHToken INT4 weights (embedding := its int8 lm_head, tied)
        "official int4/int8": lambda n: (0, "official", 32),
        "official, embed mse q4": lambda n: (4, "mse", 32) if "embedding" in n else (0, "official", 32),
        "mixed + down q8": lambda n: (None, None, None) if "g_proj" in n else
            (8, "rtn", 32) if ("embedding" in n or "out_proj" in n or "down_proj" in n) else (4, "mse", 32),
    }
    for name in sys.argv[1:] or configs:
        apply(configs[name])
        print(f"{name:28s} {kl()}", flush=True)

    # sensitivity: quantize one matrix type at a time (mse b32), everything else fp32
    if "--sens" in sys.argv or not sys.argv[1:]:
        for t in linear:
            apply(lambda n, t=t: (4, "mse", 32) if t in n else (None, None, None))
            print(f"only {t:22s} {kl()}", flush=True)
        for grp in ((0, 1, 2, 3), (32, 33, 34, 35)):
            apply(lambda n, g=grp: (8, "rtn", 32) if layer_of(n) in g or "embedding" in n else (4, "mse", 32))
            print(f"q8 layers {grp} + embed  {kl()}", flush=True)


if __name__ == "__main__":
    main()
