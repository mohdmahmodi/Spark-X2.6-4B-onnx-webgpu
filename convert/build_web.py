"""Assemble the model folder web/spark/ next to the exported ONNX graphs (convert/export_onnx.py).

Copies the tokenizer, config and license from the original checkpoint, inlines the chat template into
tokenizer_config.json (tokenizers.js and jinja read it from there), and adds the model card from model-card/.
web/spark/ is then exactly what gets uploaded as the Hugging Face model repo.

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
    for f in ["tokenizer.json", "config.json", "generation_config.json", "special_tokens_map.json", "LICENSE"]:
        shutil.copy(a.src / f, a.out / f)
    tc = json.loads((a.src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tc["chat_template"] = (a.src / "chat_template.jinja").read_text(encoding="utf-8")
    (a.out / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(CARD / "README.md", a.out / "README.md")
    shutil.copy(CARD / ".gitattributes", a.out / ".gitattributes")
    shutil.copytree(CARD / "assets", a.out / "assets", dirs_exist_ok=True)
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
