"""Feature checks, run identically on transformers bf16 and on the browser build (via tests/browser_bridge.mjs):

  thinking on   output has a reasoning block closed by </think>, then a correct answer
  thinking low  same with reasoning_effort=low
  thinking off  no reasoning block, correct answer
  multi-turn    a fact from turn 1 is recalled in turn 2 (turn 1 run with thinking; its reasoning is
                passed back as reasoning_content, as the chat template expects)
  tool call     the model emits a Spark2_5 <tool_call>, it parses (SGLang spark25 rules) to the right
                function/arguments, and after a tool response the final answer uses the result
  tool parsers  the page's JS parser (web/lib/toolcall.js) and the Python port agree

usage: python tests/features.py [--browser http://localhost:8765]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, "convert")
sys.path.insert(0, "tests")
from report import parse_calls  # noqa: E402
from run_hf import greedy, load  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("hf/Spark-X2.6-4B")
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city"]}}}


class HF:
    name = "transformers bf16"

    def __init__(self):
        self.model = load("hf_bf16")

    def gen(self, messages, thinking=False, effort="high", tools=None, max_new=512):
        ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=thinking, reasoning_effort=effort)
        out = greedy(self.model, ids, max_new)
        return {"text": tok.decode(out), "tool_calls": None}


class Browser:
    name = "browser"

    def __init__(self, url):
        self.url = url

    def gen(self, messages, thinking=False, effort="high", tools=None, max_new=512, reuse=False):
        task = {"type": "generate", "messages": messages, "thinking": thinking, "effort": effort, "tools": tools, "max_new_tokens": max_new,
                "reuse": reuse}
        req = urllib.request.Request(self.url + "/run", json.dumps(task).encode(), {"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def split_think(text):
    return (text.split("</think>", 1) + [""])[:2] if "</think>" in text else ("", text)


def run(backend):
    res = []

    def check(name, ok, detail):
        res.append({"test": name, "ok": bool(ok), "detail": detail[:300]})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail[:160]!r}", flush=True)

    q = [{"role": "user", "content": "What is 15% of 80? Give the final number."}]
    for label, kw in (("thinking on (high)", {"thinking": True}), ("thinking on (low)", {"thinking": True, "effort": "low"})):
        t = backend.gen(q, max_new=1024, **kw)["text"]
        think, ans = split_think(t)
        check(label, think.strip() and "12" in ans and "<think>" not in ans, t)
    t = backend.gen(q, thinking=False)["text"]
    check("thinking off", "</think>" not in t and "<think>" not in t and "12" in t, t)

    m = [{"role": "user", "content": "My favourite colour is teal and my cat is called Miso. Just acknowledge briefly."}]
    t1 = backend.gen(m, thinking=True, max_new=768)["text"]
    think, ans = split_think(t1)
    m += [{"role": "assistant", "content": ans.strip(), "reasoning_content": think.strip()},
          {"role": "user", "content": "What is my cat called, and what is my favourite colour?"}]
    t2 = backend.gen(m, thinking=False)["text"]
    check("multi-turn recall", "miso" in t2.lower() and "teal" in t2.lower(), t2)

    m = [{"role": "user", "content": "What's the weather in Berlin right now, in celsius?"}]
    r = backend.gen(m, tools=[WEATHER], max_new=128)
    calls = parse_calls(r["text"], [WEATHER])
    good = len(calls) == 1 and calls[0]["name"] == "get_weather" and calls[0]["arguments"].get("city", "").lower() == "berlin"
    check("tool call parses (spark25)", good, r["text"])
    if r.get("tool_calls") is not None:
        check("JS parser == Python parser", r["tool_calls"] == calls, json.dumps(r["tool_calls"]))
    if calls:
        m += [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": calls[0]}]},
              {"role": "tool", "content": json.dumps({"city": "Berlin", "temp_c": 18, "condition": "light rain"})}]
        t = backend.gen(m, tools=[WEATHER], max_new=128)["text"]
        check("answer uses tool result", "18" in t and "<tool_call>" not in t, t)

    if isinstance(backend, Browser):
        # KV prefix reuse (on by default in the page): a chat turn that extends the cached sequence
        # only prefills the new tokens; outputs must equal a from-scratch run.
        conv = [{"role": "user", "content": "Name three rivers in Africa."}]
        a1 = backend.gen(conv, max_new=96)
        conv += [{"role": "assistant", "content": a1["text"].strip()}, {"role": "user", "content": "Which of them is the longest?"}]
        fresh = backend.gen(conv, max_new=96)
        backend.gen(conv[:1], max_new=96, reuse=True)            # cache now holds turn 1
        reused = backend.gen(conv, max_new=96, reuse=True)
        n = next((i for i, (x, y) in enumerate(zip(fresh["ids"], reused["ids"])) if x != y), min(len(fresh["ids"]), len(reused["ids"])))
        check("KV prefix reuse == fresh prefill", reused["stats"]["reusedTokens"] > 0 and n >= min(48, len(fresh["ids"])),
              f"reused {reused['stats']['reusedTokens']} of {reused['stats']['promptTokens']} prompt tokens, {n}/{len(fresh['ids'])} tokens identical, "
              f"TTFT {fresh['stats']['ttftMs']:.0f} -> {reused['stats']['ttftMs']:.0f} ms")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default="http://localhost:8765")
    a = ap.parse_args()
    out = {}
    for b in (HF(), Browser(a.browser)):
        print(b.name, flush=True)
        out[b.name] = run(b)
        if isinstance(b, HF):
            del b.model
            torch.cuda.empty_cache()
    rows, md = [], ["## Features (thinking, multi-turn, tool calls)", "", "| check | transformers bf16 | browser |", "|---|---|---|"]
    hf_rows = {x["test"]: x for x in out["transformers bf16"]}
    for y in out["browser"]:
        x = hf_rows.get(y["test"])
        md.append(f"| {y['test']} | {('ok' if x['ok'] else 'FAIL') if x else 'n/a'} | {'ok' if y['ok'] else 'FAIL'} |")
    n_hf, n_br = sum(x["ok"] for x in out["transformers bf16"]), sum(y["ok"] for y in out["browser"])
    rows.append(["Features: thinking / multi-turn / tool calls", f"{n_hf}/{len(out['transformers bf16'])}", f"{n_br}/{len(out['browser'])}", n_br >= n_hf])
    md += ["", "Browser outputs:", ""] + [f"- **{y['test']}**: `{y['detail'][:200]}`".replace("\n", " ") for y in out["browser"]]
    Path("export/tests/results/extra_features.json").write_text(json.dumps(rows), encoding="utf-8")
    Path("export/tests/results/extra_features.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
