"""Back-to-back generations on the browser build for N minutes (default 30), via tests/browser_bridge.mjs.

Mixes short chat, GSM8K-length answers and an 8k-token prompt (run first, so the static KV cache
reaches its working size before memory is tracked). Samples GPU memory (nvidia-smi), page JS heap
and KV capacity; reports errors, decode speed drift and memory growth.

usage: python tests/stability.py [--minutes 30] [--bridge http://localhost:8765]
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

SUITES = Path("export/tests/suites")


def gpu_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return float(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bridge", default="http://localhost:8765")
    a = ap.parse_args()

    def call(path, task=None):
        req = urllib.request.Request(a.bridge + path, json.dumps(task).encode() if task else None, {"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=3600).read())

    parity = json.loads((SUITES / "parity.json").read_text(encoding="utf-8"))
    gsm = json.loads((SUITES / "gsm8k.json").read_text(encoding="utf-8"))
    long_task = next(t for t in json.loads((SUITES / "niah.json").read_text(encoding="utf-8")) if t["length"] == 8192)
    plan = [long_task]
    for i in range(10000):
        plan += [parity[i % 50], gsm[(i * 7) % 200], parity[(i * 3 + 1) % 50]]
        if i % 10 == 9:
            plan.append(long_task)

    t0, log, errors = time.time(), [], []
    for i, t in enumerate(plan):
        if time.time() - t0 > a.minutes * 60:
            break
        task = {k: v for k, v in t.items() if k != "prompt_ids"}
        task["max_new_tokens"] = max(t["max_new_tokens"], 256)
        try:
            r = call("/run", task)
            if "error" in r:
                raise RuntimeError(r["error"])
            mem = call("/memory")
            row = {"i": i, "t": time.time() - t0, "id": t["id"], "prompt": r["stats"]["promptTokens"], "new": r["stats"]["newTokens"],
                   "tok_s": r["stats"]["decodeTokS"], "ttft_ms": r["stats"]["ttftMs"], "gpu_mb": gpu_mem_mb(),
                   "js_heap_mb": mem["jsHeapMB"], "kv_capacity": mem["kvCapacity"], "page_errors": mem["errors"]}
            log.append(row)
            if i % 10 == 0:
                print(json.dumps(row), flush=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{time.time() - t0:.0f}s {t['id']}: {e}")
            print("ERROR", errors[-1], flush=True)
            if len(errors) > 5:
                break

    Path("export/tests/results/stability.json").write_text(json.dumps({"log": log, "errors": errors}), encoding="utf-8")
    short = [r for r in log if r["prompt"] < 1000 and r["new"] > 32]
    k = max(1, len(short) // 6)
    first, last = short[:k], short[-k:]
    tps = lambda rows: sum(r["tok_s"] for r in rows) / max(1, len(rows))  # noqa: E731
    warm = [r for r in log if r["t"] > 120]
    mem0, mem1 = (warm[0]["gpu_mb"], warm[-1]["gpu_mb"]) if warm else (0, 0)
    heap0, heap1 = (warm[0]["js_heap_mb"] or 0, warm[-1]["js_heap_mb"] or 0) if warm else (0, 0)
    minutes = log[-1]["t"] / 60 if log else 0
    tokens = sum(r["new"] for r in log)
    md = ["## Stability (back-to-back generations)", "",
          f"- {len(log)} generations, {tokens} tokens in {minutes:.1f} min, {len(errors)} errors, page errors: {log[-1]['page_errors'] if log else '?'}",
          f"- decode tok/s (prompts < 1k tokens): first sixth {tps(first):.1f}, last sixth {tps(last):.1f}",
          f"- GPU memory used (whole GPU) after warm-up: {mem0:.0f} MB -> {mem1:.0f} MB (max {max((r['gpu_mb'] for r in log), default=0):.0f} MB)",
          f"- page JS heap after warm-up: {heap0:.0f} MB -> {heap1:.0f} MB; KV capacity {log[-1]['kv_capacity'] if log else '?'} tokens"]
    md += [f"- error: {e}" for e in errors]
    ok = not errors and minutes >= a.minutes * 0.95 and tps(last) >= 0.9 * tps(first) and mem1 - mem0 < 512
    Path("export/tests/results/extra_stability.json").write_text(json.dumps([[
        f"Stability: {minutes:.0f} min back-to-back", "n/a", f"{len(errors)} errors, tok/s {tps(first):.0f}->{tps(last):.0f}, GPU mem {mem1 - mem0:+.0f} MB", ok]]),
        encoding="utf-8")
    Path("export/tests/results/extra_stability.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
