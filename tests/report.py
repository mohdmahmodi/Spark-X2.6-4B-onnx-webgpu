"""Score every system's results against the transformers bf16 reference; write docs/VALIDATION.md.

usage: python tests/report.py
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "convert")
from export_onnx import SRC  # noqa: E402

SUITES = Path("export/tests/suites")
RESULTS = Path("export/tests/results")
RESULTS_ACC4 = Path("export/tests/results_acc4")  # earlier browser build (MatMulNBits accuracy_level 4)
REPORT = Path("docs/VALIDATION.md")
V = 131072
tok = AutoTokenizer.from_pretrained(SRC)


def sysdir(system):
    return RESULTS_ACC4 / system[5:] if system.startswith("acc4/") else RESULTS / system


LABELS = {"hf_bf16": "transformers bf16", "hf_fp32": "transformers fp32", "hf_int4": "transformers + official INT4 weights",
          "hf_emb8": "transformers bf16, int8 embedding lookup (as in the build)",
          "ort_cpu_q4f16": "same ONNX graph, CPU onnxruntime", "browser_q4f16": "**browser q4f16 (shipped)**",
          "browser_q4": "browser q4 (fp32 activations)", "acc4/browser_q4f16": "browser q4f16, accuracy_level 4 (earlier build)",
          "acc4/browser_q4": "browser q4 fp32-act, accuracy_level 4 (earlier)",
          "acc4/browser_q4f16_r32": "browser q4f16 fp32-residual, acc. level 4 (earlier)",
          "acc4/ort_cpu_q4f16": "CPU onnxruntime, accuracy_level 4 (earlier)"}


def label(s):
    return LABELS.get(s, s)


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for paired correct/incorrect lists: (only-a-right, only-b-right, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return n01, n10, p


def load_rows(system, suite):
    f = sysdir(system) / f"{suite}.jsonl"
    if not f.exists():
        return None
    rows = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def text_of(r):
    return r.get("text") if r.get("text") is not None else tok.decode(r["ids"])


# ---------------------------------------------------------------- 1. logits
def logits_of(system):
    d = sysdir(system)
    if (d / "logits.npy").exists():
        return np.load(d / "logits.npy")
    if (d / "logits.f32").exists():
        return np.fromfile(d / "logits.f32", dtype=np.float32).reshape(-1, V)
    return None


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def logit_stats(ref, test):
    lp, lq = log_softmax(ref), log_softmax(test)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    top1 = (ref.argmax(-1) == test.argmax(-1))
    return {"n": len(kl), "mean_kl": float(kl.mean()), "p95_kl": float(np.quantile(kl, 0.95)), "max_kl": float(kl.max()),
            "top1": float(top1.mean())}


# ---------------------------------------------------------------- 2. greedy parity
def parity_stats(ref_rows, rows):
    out = []
    for k, r in ref_rows.items():
        a, b = r["ids"], rows[k]["ids"]
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        full = len(a) == len(b) == n
        out.append({"id": k, "match": n, "ref_len": len(a), "identical": full, "first_token": bool(a and b and a[0] == b[0]) or (not a and not b)})
    return out


# ---------------------------------------------------------------- 3. GSM8K / BFCL
def gsm8k_answer(text):
    text = text.split("</think>")[-1]
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    cand = m[-1] if m else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", cand.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def num_eq(a, b):
    try:
        return a is not None and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except ValueError:
        return False


ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)


def parse_calls(text, tools):
    """Python port of SGLang's spark25 detector (python/sglang/srt/function_call/spark25_detector.py)."""
    types = {}
    for t in tools or []:
        f = t["function"]
        for k, d in (f.get("parameters", {}).get("properties") or {}).items():
            types[(f["name"], k)] = d.get("type", "string") if isinstance(d, dict) and isinstance(d.get("type"), str) else "string"

    def conv(v, typ):
        if v.lower() == "null":
            return None
        typ = typ.lower()
        try:
            if typ in ("string", "str", "text"):
                return v
            if typ in ("integer", "int"):
                return int(v)
            if typ in ("number", "float"):
                x = float(v)
                return int(x) if x.is_integer() else x
            if typ in ("boolean", "bool"):
                if v.strip().lower() not in ("true", "1", "false", "0"):
                    raise ValueError
                return v.strip().lower() in ("true", "1")
            return json.loads(v)
        except (ValueError, TypeError):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return v

    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        i = body.find("<arg_key>")
        name = (body if i < 0 else body[:i]).strip()
        if name:
            calls.append({"name": name, "arguments": {k: conv(v, types.get((name, k), "string")) for k, v in ARG.findall(body) if k}})
    return calls


def norm_str(s):
    return re.sub(r"[ ,./\-_*^]", "", s).lower()


def value_ok(v, allowed):
    for a in allowed:
        if isinstance(a, str) and isinstance(v, str) and norm_str(a) == norm_str(v):
            return True
        if isinstance(a, bool) or isinstance(v, bool):
            if a is v:
                return True
            continue
        if isinstance(a, (int, float)) and isinstance(v, (int, float)) and math.isclose(a, v, rel_tol=1e-6):
            return True
        if isinstance(a, list) and isinstance(v, list) and len(a) == len(v) and all(value_ok(x, [y]) for x, y in zip(v, a)):
            return True
        if isinstance(a, dict) and isinstance(v, dict) and set(a) <= set(v) | {k for k, x in a.items() if "" in x} and \
                all(value_ok(v[k], x) for k, x in a.items() if k in v):
            return True
        if a == v:
            return True
    return False


def bfcl_ok(calls, truth, tools):
    """BFCL simple AST check: one call, right name, required args present, every value allowed."""
    if len(calls) != 1:
        return False
    (fname, allowed), = truth[0].items()
    c = calls[0]
    if c["name"] != fname:
        return False
    fdef = next(t["function"] for t in tools if t["function"]["name"] == fname)
    required = fdef.get("parameters", {}).get("required", [])
    if any(r not in c["arguments"] for r in required):
        return False
    for k, v in c["arguments"].items():
        if k not in allowed or not value_ok(v, allowed[k]):
            return False
    return all(k in c["arguments"] or "" in vals for k, vals in allowed.items())


def score(system, suite, tasks):
    rows = load_rows(system, suite)
    if not rows or len(rows) < len(tasks):
        return None
    ok = []
    for t in tasks:
        text = text_of(rows[t["id"]])
        if suite.startswith("gsm8k"):
            ok.append(num_eq(gsm8k_answer(text), t["answer"]))
        elif suite == "bfcl":
            ok.append(bfcl_ok(parse_calls(text, t["tools"]), t["ground_truth"], t["tools"]))
        elif suite == "niah":
            ok.append(t["needle"] in text)
    return ok


def main():
    REPORT.parent.mkdir(exist_ok=True)
    systems = sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
    systems += sorted("acc4/" + p.name for p in RESULTS_ACC4.iterdir() if p.is_dir()) if RESULTS_ACC4.exists() else []
    out = [f"# {SRC.name} browser build: validation report", ""]
    info = RESULTS / "browser_q4f16/info.json"
    if info.exists():
        i = json.loads(info.read_text())
        out += [f"Shipped build: `{i['variant']}` on {i['gpu']} (Chrome, WebGPU, static KV cache). "
                "All browser results come from the real page driven by headless Chrome (tests/run_browser.mjs); the page tokenizes "
                "with tokenizers.js + the official chat template, and its prompt ids were checked to equal transformers' on every task.", ""]
    summary = []

    # 1. logits
    ref, i4 = logits_of("hf_bf16"), logits_of("hf_int4")
    out += ["## 1. Next-token logits vs transformers", "",
            "20 prompts x 8 positions (after the prompt, then teacher-forcing the bf16 greedy continuation).", "",
            "| system | mean KL vs bf16 | p95 | max | top-1 vs bf16 | mean KL vs INT4 weights | top-1 vs INT4 |",
            "|---|---|---|---|---|---|---|"]
    lstats = {}
    for s in systems:
        x = logits_of(s)
        if x is None or s == "hf_bf16" or x.shape != ref.shape:
            continue
        st = lstats[s] = logit_stats(ref, x)
        si = logit_stats(i4, x) if s != "hf_int4" else {"mean_kl": 0.0, "top1": 1.0}
        st["kl_int4"] = si["mean_kl"]
        out.append(f"| {label(s)} | {st['mean_kl']:.4f} | {st['p95_kl']:.4f} | {st['max_kl']:.3f} | {st['top1']:.1%} | "
                   f"{si['mean_kl']:.5f} | {si['top1']:.1%} |")
    out += ["", "KL vs INT4 isolates what the ONNX/WebGPU build adds on top of the official quantization."]
    if "browser_q4f16" in lstats and "ort_cpu_q4f16" in lstats:
        k = logit_stats(logits_of("ort_cpu_q4f16"), logits_of("browser_q4f16"))
        out += ["", f"WebGPU kernels vs CPU kernels, same graph: mean KL {k['mean_kl']:.5f}, max {k['max_kl']:.4f}, top-1 agree {k['top1']:.1%}."]
        b = lstats["browser_q4f16"]
        summary.append(("Logits in WebGPU: mean KL to bf16 (160 pos.)", f"INT4 weights in HF: {lstats['hf_int4']['mean_kl']:.4f}",
                        f"{b['mean_kl']:.4f} (adds {b['kl_int4']:.5f} vs INT4)", b["mean_kl"] <= 1.2 * lstats["hf_int4"]["mean_kl"]))

    # 2. greedy parity
    tasks = {t["id"]: t for t in json.loads((SUITES / "parity.json").read_text(encoding="utf-8"))}
    ref_rows = load_rows("hf_bf16", "parity")
    out += ["", "## 2. Greedy output parity (50 prompts: chat, code, math, Chinese, tool calls; 128 tokens) vs transformers bf16", "",
            "| system | first token same | identical 128-token outputs | mean tokens before divergence | median |", "|---|---|---|---|---|"]
    pstats = {}
    for s in systems:
        rows = load_rows(s, "parity")
        if s == "hf_bf16" or not rows or len(rows) < len(ref_rows):
            continue
        st = pstats[s] = parity_stats(ref_rows, rows)
        m = [x["match"] for x in st]
        out.append(f"| {label(s)} | {sum(x['first_token'] for x in st)}/50 | {sum(x['identical'] for x in st)}/50 | "
                   f"{np.mean(m):.1f} | {np.median(m):.0f} |")
    out += ["", "Any 4-bit model drifts from bf16 at low-margin tokens; the INT4-weights row is the bar the browser build should meet."]
    if "browser_q4f16" in pstats:
        s = "browser_q4f16"
        rows = load_rows(s, "parity")
        if not all(r.get("prompt_ids_match", True) for r in rows.values()):
            out.append("\n**Browser tokenization differed from transformers on some prompts.**")
        cats = {}
        for x in pstats[s]:
            cats.setdefault(tasks[x["id"]]["category"], []).append(x["match"])
        out += ["", "Browser, mean matched tokens by category: " + ", ".join(f"{c} {np.mean(v):.0f}" for c, v in cats.items())]
        out += ["", "### Worst 5 divergences (shipped browser build)", ""]
        for x in sorted(pstats[s], key=lambda x: x["match"])[:5]:
            a, b = ref_rows[x["id"]]["ids"], rows[x["id"]]["ids"]
            n = x["match"]
            q = tasks[x["id"]]["messages"][-1]["content"][:90].replace("\n", " ")
            cell = lambda ids: tok.decode(ids[: n + 40]).replace("\n", "<br>").replace("|", "\\|")  # noqa: E731
            out += [f"**{x['id']}** ({q}): diverges after {n} tokens", "",
                    "| transformers bf16 | browser |", "|---|---|", f"| {cell(a)} | {cell(b)} |", ""]
        ft = sum(x["first_token"] for x in pstats[s])
        fi = sum(x["first_token"] for x in pstats["hf_int4"])
        mi = np.mean([x["match"] for x in pstats["hf_int4"]])
        mb = np.mean([x["match"] for x in pstats[s]])
        summary.append(("Greedy parity: first token same (50 prompts)", f"INT4 weights in HF: {fi}/50", f"{ft}/50", ft >= fi - 2))
        summary.append(("Greedy parity: mean tokens before divergence", f"INT4 weights in HF: {mi:.1f}", f"{mb:.1f}", mb >= 0.8 * mi))

    # 3-4. benchmarks
    for suite, title in (("gsm8k_full", "GSM8K, all 1319 test problems (thinking off, greedy, 768 tokens)"),
                         ("gsm8k", "GSM8K, first 200 (diagnostic runs)"),
                         ("bfcl", "BFCL v3 simple function calling, first 200 (Spark tool format, SGLang spark25 parsing)"),
                         ("niah", "Needle in a haystack, 1k-16k tokens, needle at start/middle/end")):
        f = SUITES / f"{suite}.json"
        if not f.exists():
            continue
        ts = json.loads(f.read_text(encoding="utf-8"))
        out += ["", f"## {title}", "", "| system | score | vs INT4 weights: only-INT4-right / only-this-right, McNemar p |", "|---|---|---|"]
        oks = {}
        for s in systems:
            ok = score(s, suite, ts)
            if ok is not None:
                oks[s] = ok
        for s, ok in oks.items():
            cmp = ""
            if "hf_int4" in oks and s != "hf_int4" and suite != "niah":
                n01, n10, p = mcnemar(oks["hf_int4"], ok)
                cmp = f"{n01} / {n10}, p = {p:.2f}"
            out.append(f"| {label(s)} | {sum(ok)}/{len(ok)} ({sum(ok) / len(ok):.1%}) | {cmp} |")
            if suite == "niah":
                cells = {t["id"]: ("ok" if o else "MISS") for t, o in zip(ts, ok)}
                out.append("| | " + ", ".join(f"{k[5:]}: {v}" for k, v in cells.items()) + " | |")
        if suite != "gsm8k" and all(k in oks for k in ("hf_bf16", "hf_int4", "browser_q4f16")):
            sb, si, sw = (sum(oks[k]) / len(oks[k]) for k in ("hf_bf16", "hf_int4", "browser_q4f16"))
            if suite == "niah":
                ok = sw >= sb
            else:
                _, _, p = mcnemar(oks["hf_int4"], oks["browser_q4f16"])
                ok = sw >= si or p >= 0.05  # not significantly worse than the official INT4 model
            summary.append((title.split(",")[0], f"bf16 {sb:.1%}, INT4 weights {si:.1%}", f"{sw:.1%}", ok))

    for extra in sorted(RESULTS.glob("extra_*.json")):  # features.py, stability.py, speed
        for row in json.loads(extra.read_text(encoding="utf-8")):
            summary.append(tuple(row))
    out = out[:2] + ["## Summary", "", "| test | original (transformers) | browser build | pass |", "|---|---|---|---|"] + \
        [f"| {a} | {b} | {c} | {'PASS' if d else 'FAIL'} |" for a, b, c, d in summary] + [""] + out[2:]
    for extra_md in sorted(RESULTS.glob("extra_*.md")):
        out += ["", extra_md.read_text(encoding="utf-8")]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:30]))


if __name__ == "__main__":
    main()
