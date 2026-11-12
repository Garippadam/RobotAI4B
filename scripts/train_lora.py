#!/usr/bin/env python3
"""Gemma 4 E4B (veya başka bir sohbet modeli) üzerinde LoRA / QLoRA ince ayarı.

Yalnızca asistan cevaplarındaki token'lar üzerinden kayıp (loss) hesaplanır;
sistem mesajı ve kullanıcı soruları maskelenir.

Kullanım (GPU önerilir):
    python scripts/train_lora.py --config configs/robotai4b.yaml
    # 16 GB GPU (ör. Colab T4) için:
    python scripts/train_lora.py --config configs/robotai4b.yaml --load-in-4bit
"""
import argparse
import inspect
import json
import math
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer, Trainer, TrainingArguments)


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #

def encode_conversation(tokenizer, messages, max_len):
    """Sohbeti token'lara çevirir; yalnızca asistan cevaplarını etiketler."""
    full = tokenizer.apply_chat_template(messages, tokenize=False)

    # Her asistan cevabının metin içindeki [başlangıç, bitiş) aralığını bul.
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False,
                                               add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        if not full.startswith(prefix) or not full.startswith(upto):
            raise ValueError("Sohbet şablonu önek-tutarlı değil; maskeleme yapılamıyor.")
        spans.append((len(prefix), len(upto)))

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)
    for k, (start, _end) in enumerate(offsets):
        if any(s <= start < e for s, e in spans):
            labels[k] = ids[k]

    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.items, dropped = [], 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = encode_conversation(tokenizer, json.loads(line)["messages"], max_len)
            if ex is None:
                dropped += 1
            else:
                self.items.append(ex)
        n_tok = sum(len(x["input_ids"]) for x in self.items)
        print(f"{path}: {len(self.items)} örnek, {n_tok} token (atlanan: {dropped})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, pad_id, extra_zero_keys=()):
        self.pad_id = pad_id
        self.extra_zero_keys = extra_zero_keys

    def __call__(self, batch):
        n = max(len(x["input_ids"]) for x in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for x in batch:
            pad = n - len(x["input_ids"])
            out["input_ids"].append(x["input_ids"] + [self.pad_id] * pad)
            out["labels"].append(x["labels"] + [-100] * pad)
            out["attention_mask"].append([1] * len(x["input_ids"]) + [0] * pad)
        out = {k: torch.tensor(v) for k, v in out.items()}
        # Çok kipli (multimodal) Gemma modelleri eğitimde token_type_ids bekleyebilir:
        # hepsi 0 = "sadece metin".
        for k in self.extra_zero_keys:
            out[k] = torch.zeros_like(out["input_ids"])
        return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def pick_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("UYARI: GPU bf16 desteklemiyor, fp16 kullanılıyor (Gemma için bf16 önerilir).")
        return torch.float16
    return torch.float32


def load_model(name, dtype, load_in_4bit):
    kwargs = {"dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            # Görüntü/ses kuleleri eğitilmiyor; onları nicemlemeye gerek yok.
            llm_int8_skip_modules=["vision_tower", "audio_tower", "embed_vision",
                                   "embed_audio", "lm_head"],
        )
    try:  # Gemma 4 çok kipli bir model; ağırlık adlarını korumak için tam sınıfı yükle
        return AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robotai4b.yaml")
    ap.add_argument("--base-model", help="Yapılandırmadaki temel modeli geçersiz kıl")
    ap.add_argument("--train-file")
    ap.add_argument("--val-file")
    ap.add_argument("--output-dir")
    ap.add_argument("--max-steps", type=int, default=-1, help="Hızlı deneme için adım sınırı")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tcfg, lcfg = cfg["train"], cfg["lora"]
    base = args.base_model or cfg["base_model"]
    data_dir = Path(cfg["data"]["output_dir"])
    train_file = args.train_file or data_dir / "train.jsonl"
    val_file = args.val_file or data_dir / "val.jsonl"
    out_dir = Path(args.output_dir or tcfg["output_dir"])
    load_in_4bit = args.load_in_4bit or tcfg.get("load_in_4bit", False)

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = ChatDataset(train_file, tokenizer, tcfg["max_seq_len"])
    val_ds = ChatDataset(val_file, tokenizer, tcfg["max_seq_len"]) if Path(val_file).exists() else None

    dtype = pick_dtype()
    model = load_model(base, dtype, load_in_4bit)
    model.config.use_cache = False
    if tcfg["gradient_checkpointing"]:
        # Not: peft'in prepare_model_for_kbit_training'i kullanılmıyor; o fonksiyon nicemlenmemiş
        # tüm ağırlıkları fp32'ye çevirir. Gemma 4 E4B'de bu, ~2.8B parametrelik katman-başı
        # gömmeleri (~11 GB) kapsar ve 16 GB GPU'ya sığmaz.
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    fwd_params = inspect.signature(model.get_base_model().forward).parameters
    extra = tuple(k for k in ("token_type_ids", "mm_token_type_ids") if k in fwd_params)

    steps_per_epoch = math.ceil(len(train_ds) / (tcfg["per_device_batch_size"] *
                                                 tcfg["gradient_accumulation_steps"]))
    print(f"Epoch başına optimizer adımı: {steps_per_epoch}")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        max_steps=args.max_steps,
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_steps=max(1, int(tcfg["warmup_ratio"] * steps_per_epoch * tcfg["epochs"])),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps" if val_ds else "no",
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=cfg["data"]["seed"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tokenizer.pad_token_id, extra))
    trainer.train()

    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"LoRA adaptörü kaydedildi: {final}")


if __name__ == "__main__":
    main()
