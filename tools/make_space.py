"""Build the Hugging Face Space (static, no build step) for the web app into release/space/.

The Space holds the app only (HTML, CSS, JavaScript, icons). The page loads the model from the model repository
(https://huggingface.co/<model>/resolve/main) and the browser keeps it in its Cache API.

usage: python tools/make_space.py [--model mohdmahmodi/Spark-X2.6-4B-ONNX-WebGPU] [--out release/space]
then:  hf upload <user>/<space> release/space . --repo-type space
"""
import argparse
import shutil
from pathlib import Path

WEB = Path("web")
SKIP = {"spark"}  # local model build; the Space loads the model from the Hub
TEXT = {".html", ".js", ".css", ".svg", ".txt"}
DEFAULT_MODEL = "mohdmahmodi/Spark-X2.6-4B-ONNX-WebGPU"
SOURCE = "https://github.com/mohdmahmodi/Spark-X2.6-4B-onnx-webgpu"

README = """---
title: Spark-X2.6 WebGPU
emoji: ⚡
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Spark-X2.6-4B chat running in your browser on WebGPU
models:
  - {model}
  - XHToken/Spark-X2.6-4B
tags:
  - webgpu
  - onnxruntime-web
  - in-browser
thumbnail: https://huggingface.co/{model}/resolve/main/assets/hero.png
---

# Spark-X2.6 in the browser

Chat with [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) running entirely on your GPU
through WebGPU. Nothing you type leaves your device.

Requires desktop Chrome or Edge and about 4 GB of free GPU memory. The 2.77 GB model is downloaded from
[{model}](https://huggingface.co/{model}) on the first visit and cached by the browser.

Benchmarks: `benchmarks.html` (linked in the sidebar). Source and validation: {source}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("release/space"))
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    base = f"https://huggingface.co/{a.model}/resolve/main"
    for src in sorted(WEB.rglob("*")):
        rel = src.relative_to(WEB)
        if src.is_dir() or rel.parts[0] in SKIP:
            continue
        dst = a.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix not in TEXT:
            shutil.copy(src, dst)
            continue
        text = src.read_text(encoding="utf-8")
        if rel.as_posix() == "index.html":  # load straight from the Hub, no local-folder probe
            old = f'<meta name="spark-model" content="./spark" data-fallback="https://huggingface.co/{DEFAULT_MODEL}/resolve/main" />'
            assert text.count(old) == 1, "model meta tag not found in index.html"
            text = text.replace(old, f'<meta name="spark-model" content="{base}" />')
        text = text.replace(f"huggingface.co/{DEFAULT_MODEL}", f"huggingface.co/{a.model}")
        dst.write_text(text, encoding="utf-8", newline="\n")
    (a.out / "README.md").write_text(README.format(model=a.model, source=SOURCE), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} (model: {base})")


if __name__ == "__main__":
    main()
