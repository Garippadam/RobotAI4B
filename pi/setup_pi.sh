#!/usr/bin/env bash
# Raspberry Pi 5 (64-bit Raspberry Pi OS / Debian Bookworm) için kurulum:
#  1) gerekli paketler  2) llama.cpp'yi Pi'nin işlemcisine özel derleme  3) modeli indirme
#
# Kullanım:  bash pi/setup_pi.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/robotai4b.env"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "UYARI: 64-bit ARM (aarch64) bekleniyordu, bulunan: $(uname -m). 64-bit Raspberry Pi OS kullanın."
fi

echo "==> Paketler kuruluyor"
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3 curl

echo "==> llama.cpp derleniyor ($LLAMA_DIR)"
mkdir -p "$(dirname "$LLAMA_DIR")"
if [ ! -d "$LLAMA_DIR" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" pull --ff-only || true
fi
# GGML_NATIVE: Cortex-A76'nın dotprod/fp16 komutlarını kullanır.
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build "$LLAMA_DIR/build" -j4 --target llama-server llama-cli llama-bench

echo "==> Model indiriliyor: $MODEL_REPO / $MODEL_FILE"
mkdir -p "$MODEL_DIR"
# Özel (private) bir Hugging Face deposu için:  HF_TOKEN=hf_xxx bash pi/setup_pi.sh
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
download() {  # $1 = depo, $2 = dosya adı
  [ -f "$MODEL_DIR/$2" ] && return 0
  curl -fL --retry 5 -C - "${AUTH[@]}" -o "$MODEL_DIR/$2.part" \
    "https://huggingface.co/$1/resolve/main/$2"
  mv "$MODEL_DIR/$2.part" "$MODEL_DIR/$2"
}
download "$MODEL_REPO" "$MODEL_FILE"
if [ -n "$LORA_FILE" ]; then download "$LORA_REPO" "$LORA_FILE"; fi
ls -lh "$MODEL_DIR"

echo
echo "Kurulum tamam. Başlatmak için:"
echo "  bash $HERE/run_server.sh        # sunucuyu başlatır"
echo "  python3 $HERE/chat.py           # başka bir terminalde sohbet"
