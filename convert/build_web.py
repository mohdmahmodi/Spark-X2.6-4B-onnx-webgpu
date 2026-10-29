"""Assemble the model folder web/spark/ next to the exported ONNX graphs (convert/export_onnx.py).

Copies the tokenizer, config and license from the original checkpoint, inlines the chat template into
tokenizer_config.json (tokenizers.js and jinja read it from there), and adds the model card from model-card/.
web/spark/ is then exactly what gets uploaded as the Hugging Face model repo.

config.json is the original with two changes so Transformers.js can load the graph: model_type "qwen3" (it has no
class for Spark; its Qwen3 decoder feeds the same inputs and KV cache names and shapes as this graph) and the
external-data chunk count of each variant. The app itself does not read config.json.

usage: python convert/build_web.py [--src hf/Spark-X2.6-4B] [--out web/spark]
"""
import argparse
import json
import shutil
from pathlib import Path

SRC = Path("hf/Spark-X2.6-4B")
CARD = Path("model-card")
WEB = Path("web/spark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC, help="original checkpoint folder")
    ap.add_argument("--out", type=Path, default=WEB)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ["tokenizer.json", "generation_config.json", "special_tokens_map.json", "LICENSE"]:
        shutil.copy(a.src / f, a.out / f)
    cfg = json.loads((a.src / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "qwen3"
    cfg.pop("auto_map", None)  # points at Python code that is not in this repo
    variants = json.loads((a.out / "spark.json").read_text())["variants"]
    cfg["transformers.js_config"] = {"use_external_data_format": {Path(v["file"]).name: len(v["external_data"]) for v in variants.values()}}
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tc = json.loads((a.src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tc["chat_template"] = (a.src / "chat_template.jinja").read_text(encoding="utf-8")
    (a.out / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(CARD / "README.md", a.out / "README.md")
    shutil.copy(CARD / "hf.gitattributes", a.out / ".gitattributes")  # LFS rules for the Hugging Face repo only
    shutil.copytree(CARD / "assets", a.out / "assets", dirs_exist_ok=True)
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
