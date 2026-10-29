"""Run the original transformers model (trust_remote_code) and save reference outputs.

Saves, per test prompt: input ids, logits at a set of positions, the max |hidden| per
layer (to decide whether an fp16 residual stream is safe), and a greedy continuation.

usage: python tests/reference.py   (writes export/reference/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = Path("hf/Spark-X2.6-4B")
OUT = Path("export/reference")
GEN_TOKENS = 48


def prompts(tok):
    short = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        add_generation_prompt=True, enable_thinking=False,
    )
    # > 512 tokens so sliding layers actually drop keys, plus thinking on.
    text = (SRC / "README.md").read_text(encoding="utf-8")
    long_ids = tok.apply_chat_template(
        [{"role": "user", "content": "Summarize this model card in three bullet points.\n\n" + text}],
        add_generation_prompt=True, enable_thinking=True,
    )
    return {"short": short, "long": long_ids[:1400] if len(long_ids) > 1400 else long_ids}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=torch.float32).to(device).eval()

    peaks = {}

    def hook(i):
        def f(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            peaks[i] = max(peaks.get(i, 0.0), h.abs().max().item())
        return f

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook(i))

    summary = {}
    for name, ids in prompts(tok).items():
        peaks.clear()
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(x, use_cache=False).logits[0].float().cpu().numpy()
        t = len(ids)
        pos = sorted(set(list(range(0, t, 64)) + list(range(max(0, t - 16), t))))
        np.savez(OUT / f"{name}.npz", ids=np.array(ids, np.int64), pos=np.array(pos), logits=logits[pos])
        with torch.no_grad():
            gen = model.generate(x, max_new_tokens=GEN_TOKENS, do_sample=False, use_cache=True)[0, t:].tolist()
        summary[name] = {
            "tokens": t,
            "max_abs_hidden_per_layer": [round(peaks[i], 1) for i in range(len(peaks))],
            "greedy": gen,
            "greedy_text": tok.decode(gen),
        }
        print(name, t, "tokens; peak |hidden|", max(peaks.values()), "\n ", repr(tok.decode(gen)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
