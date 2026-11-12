#!/usr/bin/env bash
# Seçenek B: Birleştirme YAPMADAN sadece LoRA adaptörünü GGUF'a çevirir (~50-100 MB).
# Az RAM'li makinelerde (ör. ücretsiz Colab) birleştirme yerine bunu kullanın.
# Pi'de hazır temel modelle (unsloth/gemma-4-E4B-it-GGUF) birlikte --lora ile yüklenir.
#
# Kullanım:
#   bash scripts/export_lora_gguf.sh outputs/robotai4b-lora/final outputs/gguf
set -euo pipefail
ADAPTER=${1:?lora adaptor klasoru}
OUT=${2:-outputs/gguf}
BASE=${BASE:-google/gemma-4-E4B-it}   # yalnızca config.json okunur, ağırlıklar indirilmez
LLAMA_DIR=${LLAMA_DIR:-third_party/llama.cpp}
NAME=${NAME:-robotai4b}

[ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
python3 -m pip install -q sentencepiece protobuf numpy safetensors
mkdir -p "$OUT"

if [ -d "$BASE" ]; then
  BASE_ARGS=(--base "$BASE")
else
  BASE_ARGS=(--base-model-id "$BASE")
fi
python3 "$LLAMA_DIR/convert_lora_to_gguf.py" "$ADAPTER" "${BASE_ARGS[@]}" \
  --outtype f16 --outfile "$OUT/$NAME-lora-F16.gguf"
ls -lh "$OUT/$NAME-lora-F16.gguf"
