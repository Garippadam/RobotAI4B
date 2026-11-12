#!/usr/bin/env python3
"""LoRA adaptörünü temel modelle birleştirir ve tam bir Hugging Face modeli kaydeder.
Çıktı, llama.cpp ile GGUF'a çevrilmeye hazırdır.

Not: Gemma 4 E4B'yi bf16 olarak birleştirmek ~20 GB RAM ister. Raspberry Pi'de değil,
eğitimi yaptığınız makinede (ör. Colab) çalıştırın.

Kullanım:
    python scripts/merge_lora.py --adapter outputs/robotai4b-lora/final --out outputs/robotai4b-merged
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", help="Varsayılan: adapter_config.json içindeki temel model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = args.base_model or json.load(open(Path(args.adapter) / "adapter_config.json"))["base_model_name_or_path"]
    print(f"Temel model: {base}")
    try:
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)

    # Tokenizer + işlemci (görüntü/ses) dosyalarını da kopyala; llama.cpp dönüştürücüsü
    # ve görüntü projektörü (mmproj) için gerekli.
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base).save_pretrained(args.out)
    except Exception as e:  # metin-tek modellerde işlemci olmayabilir
        print(f"(İşlemci kaydedilmedi: {e})")
    for name in ("chat_template.jinja",):
        src = Path(args.adapter) / name
        if src.exists():
            shutil.copy(src, Path(args.out) / name)
    print(f"Birleştirilmiş model: {args.out}")


if __name__ == "__main__":
    main()
