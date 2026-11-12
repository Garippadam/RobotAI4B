#!/usr/bin/env python3
"""Türkçe eğitim verisini hazırlar.

Yapılandırmadaki kaynakları okur, hepsini tek bir sohbet formatına çevirir:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
filtreler, tekrarları atar, karıştırır ve train/val olarak yazar.

Kullanım:
    python scripts/prepare_data.py --config configs/robotai4b.yaml
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml


def clean(text):
    if text is None:
        return ""
    return re.sub(r"[ \t]+", " ", str(text)).strip()


# --- Kaynak formatlarını ortak formata çeviren fonksiyonlar -----------------

def from_conversations(row):
    msgs = []
    for m in row["conversations"]:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(role, role)
        if role == "system":  # kaynağın kendi sistem mesajını atıyoruz (ör. "Gemma 3")
            continue
        msgs.append({"role": role, "content": clean(content)})
    return msgs


def from_soru_cevap(row):
    return [{"role": "user", "content": clean(row["soru"])},
            {"role": "assistant", "content": clean(row["cevap"])}]


def from_talimat(row):
    # merve/turkish_instructions sütun adlarının başında boşluk var: " giriş", " çıktı"
    get = lambda k: next((row[c] for c in row if c.strip() == k), None)
    instr, inp, out = clean(get("talimat")), clean(get("giriş")), clean(get("çıktı"))
    user = f"{instr}\n\n{inp}" if inp else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


def from_alpaca(row):
    instr, inp, out = clean(row["instruction"]), clean(row.get("input")), clean(row["output"])
    user = f"{instr}\n\n{inp}" if inp else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


def from_messages(row):
    return [{"role": m["role"], "content": clean(m["content"])} for m in row["messages"]
            if m["role"] != "system"]


CONVERTERS = {
    "conversations": from_conversations,
    "soru_cevap": from_soru_cevap,
    "talimat": from_talimat,
    "alpaca": from_alpaca,
    "messages": from_messages,
}


def is_good(msgs, min_chars, max_chars, strict):
    if not msgs or msgs[-1]["role"] != "assistant":
        return False
    roles = [m["role"] for m in msgs]
    # user/assistant sırayla gelmeli
    if any(r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles)):
        return False
    for m in msgs:
        if not m["content"]:
            return False
    if not strict:  # kendi elle yazılmış verimize uzunluk filtresi uygulamıyoruz
        return True
    for m in msgs:
        if m["role"] == "assistant":
            a = m["content"]
            if len(a) < min_chars or len(a) > max_chars:
                return False
            # Çeviri artığı / yarım kalmış cevaplar: "...listesi burada:" gibi
            if a.rstrip().endswith(":"):
                return False
    return True


def load_source(src, cfg):
    data_cfg = cfg["data"]
    if src["type"] == "jsonl":
        rows = [json.loads(l) for l in Path(src["path"]).read_text(encoding="utf-8").splitlines() if l.strip()]
        conv = from_messages
        strict = False
    else:
        from datasets import load_dataset
        rows = load_dataset(src["hf_id"], split=src.get("split", "train"))
        conv = CONVERTERS[src["format"]]
        strict = True

    out = []
    for row in rows:
        try:
            msgs = conv(row)
        except (KeyError, TypeError):
            continue
        if is_good(msgs, data_cfg["min_answer_chars"], data_cfg["max_answer_chars"], strict):
            out.append(msgs)

    rng = random.Random(data_cfg["seed"])
    rng.shuffle(out)
    if src.get("max_samples"):
        out = out[: src["max_samples"]]
    return out * int(src.get("repeat", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robotai4b.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = cfg["data"]
    system_prompt = cfg["system_prompt"].strip()

    examples, seen = [], set()
    for src in data_cfg["sources"]:
        if not src.get("enabled", True):
            continue
        convs = load_source(src, cfg)
        kept = 0
        for msgs in convs:
            key = hashlib.md5(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()
            # tekrarları at; ama "repeat" ile bilerek çoğaltılan kaynaklara izin ver
            if key in seen and int(src.get("repeat", 1)) == 1:
                continue
            seen.add(key)
            examples.append({"source": src["name"],
                             "messages": [{"role": "system", "content": system_prompt}] + msgs})
            kept += 1
        print(f"{src['name']:<24} {kept:>6} örnek")

    rng = random.Random(data_cfg["seed"])
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * data_cfg["val_ratio"]))
    val, train = examples[:n_val], examples[n_val:]

    out_dir = Path(data_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Toplam: train={len(train)} val={len(val)} -> {out_dir}/")


if __name__ == "__main__":
    main()
