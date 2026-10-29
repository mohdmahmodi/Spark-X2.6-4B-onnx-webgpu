"""Run a suite through the exported ONNX graph with onnxruntime (CPU by default), greedy.

Same graph as the browser, different kernels: comparing ort_cpu with the browser separates
quantization error (shared) from WebGPU kernel error (browser only).

usage: python tests/run_ort.py SUITE [--variant q4f16] [--provider cpu]   (SUITE: parity | logits | ...)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
from test_onnx import Runner  # noqa: E402

SUITES = Path("export/tests/suites")
RESULTS = Path("export/tests/results")
EOS = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--variant", default="q4f16")
    ap.add_argument("--provider", default="cpu")
    a = ap.parse_args()
    out_dir = RESULTS / f"ort_{a.provider}_{a.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Runner(a.variant, a.provider)

    if a.suite == "logits":
        tasks = json.loads((SUITES / "parity.json").read_text(encoding="utf-8"))[:20]
        forced = {f["id"]: f["forced"] for f in json.load(open(RESULTS / "hf_bf16/logits_forced.json"))}
        allv = []
        for t in tasks:
            r.reset()
            rows = [r.step(t["prompt_ids"])]
            for tok in forced[t["id"]][:-1]:
                rows.append(r.step([tok]))
            allv.append(np.stack(rows))
        np.save(out_dir / "logits.npy", np.concatenate(allv).astype(np.float32))
        print("saved", out_dir / "logits.npy")
        return

    tasks = json.loads((SUITES / f"{a.suite}.json").read_text(encoding="utf-8"))
    t0 = time.time()
    with open(out_dir / f"{a.suite}.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            r.reset()
            logits, ids = r.step(t["prompt_ids"]), []
            while len(ids) < t["max_new_tokens"]:
                nxt = int(logits.argmax())
                if nxt == EOS:
                    break
                ids.append(nxt)
                if len(ids) < t["max_new_tokens"]:
                    logits = r.step([nxt])
            f.write(json.dumps({"id": t["id"], "ids": ids}, ensure_ascii=False) + "\n")
            print(f"ort {a.suite}: {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
