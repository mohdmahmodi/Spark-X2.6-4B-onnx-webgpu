"""Run a suite through the original transformers model (trust_remote_code), greedy.

Systems:
  hf_bf16   original weights, bf16 compute (the reference)
  hf_fp32   original weights, fp32 compute (noise floor of the reference itself)
  hf_int4   XHToken/Spark-X2.6-4B-INT4 weights (dequantized into the model), bf16 compute:
            what the official INT4 checkpoint computes without activation quantization

usage: python tests/run_hf.py SYSTEM SUITE [--batch 24]
  SUITE: parity | gsm8k | bfcl | niah | logits   (logits needs hf_bf16 parity results first)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, "convert")
from export_onnx import OFFICIAL, SRC, OfficialQuant  # noqa: E402

SUITES = Path("export/tests/suites")
RESULTS = Path("export/tests/results")
EOS, PAD = 1, 2
CHUNK = 2048  # prefill chunk for long prompts (eager attention materializes [heads, S, T])


def forced_tokens(ref_ids, n=8):
    """Teacher-forced continuation for the logits test: the bf16 greedy output (+ EOS), at most n."""
    return (ref_ids + [EOS])[:n]


def load(system):
    dtype = torch.float32 if system == "hf_fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=dtype).cuda().eval()
    if system == "hf_int4":
        q = OfficialQuant(OFFICIAL)
        with torch.no_grad():
            for name, p in model.named_parameters():
                src = "lm_head" if name == "model.embedding.weight" else name[: -len(".weight")]
                got = q.get(src, 32) if p.dim() == 2 else None
                if got:
                    p.copy_(OfficialQuant.dequant(*got[:3]).to(p.device, p.dtype))
    return model


@torch.no_grad()
def greedy(model, ids, max_new):
    """Single sequence, chunked prefill + cached greedy decode."""
    cache = DynamicCache(config=model.config)
    x = torch.tensor([ids], device="cuda")
    for s in range(0, x.shape[1], CHUNK):
        out = model(x[:, s:s + CHUNK], past_key_values=cache, use_cache=True)
    gen = []
    nxt = out.logits[0, -1].argmax().item()
    while len(gen) < max_new and nxt != EOS:
        gen.append(nxt)
        if len(gen) == max_new:
            break
        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
        nxt = out.logits[0, -1].argmax().item()
    return gen


@torch.no_grad()
def greedy_batch(model, prompts, max_new):
    """Left-padded batched greedy (RoPE is relative, padded keys are masked)."""
    n = max(map(len, prompts))
    x = torch.tensor([[PAD] * (n - len(p)) + p for p in prompts], device="cuda")
    mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    out = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=EOS, pad_token_id=PAD, top_p=None, top_k=None, temperature=None)
    res = []
    for row in out[:, n:].tolist():
        res.append(row[: row.index(EOS)] if EOS in row else row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["hf_bf16", "hf_fp32", "hf_int4"])
    ap.add_argument("suite")
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = RESULTS / a.system
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load(a.system)

    if a.suite == "logits":
        ref = {r["id"]: r for r in map(json.loads, open(RESULTS / "hf_bf16/parity.jsonl", encoding="utf-8"))}
        tasks = json.loads((SUITES / "parity.json").read_text(encoding="utf-8"))[:20]
        allv = []
        with torch.no_grad():
            for t in tasks:
                forced = forced_tokens(ref[t["id"]]["ids"])
                # rows: after the prompt, then after each forced token but the last -> len(forced) rows
                x = torch.tensor([t["prompt_ids"] + forced[:-1]], device="cuda")
                lg = model(x, use_cache=False).logits[0, len(t["prompt_ids"]) - 1:].float().cpu().numpy()
                allv.append(lg)
        np.save(out_dir / "logits.npy", np.concatenate(allv).astype(np.float32))  # [sum(len(forced)+1), V]
        json.dump([{"id": t["id"], "forced": forced_tokens(ref[t["id"]]["ids"])} for t in tasks], open(out_dir / "logits_forced.json", "w"))
        print("saved", out_dir / "logits.npy")
        return

    tasks = json.loads((SUITES / f"{a.suite}.json").read_text(encoding="utf-8"))
    t0 = time.time()
    rows = []
    batched = a.suite in ("gsm8k", "gsm8k_full", "bfcl")
    step = a.batch if batched else 1
    for i in range(0, len(tasks), step):
        chunk = tasks[i:i + step]
        if batched:
            outs = greedy_batch(model, [t["prompt_ids"] for t in chunk], chunk[0]["max_new_tokens"])
        else:
            outs = [greedy(model, t["prompt_ids"], t["max_new_tokens"]) for t in chunk]
        for t, ids in zip(chunk, outs):
            rows.append({"id": t["id"], "ids": ids})
        print(f"{a.system} {a.suite}: {len(rows)}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    with open(out_dir / f"{a.suite}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
