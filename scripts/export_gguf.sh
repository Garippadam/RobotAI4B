#!/usr/bin/env bash
# Birleştirilmiş Hugging Face modelini GGUF'a çevirir ve Raspberry Pi için nicemler (quantize).
#
# Kullanım:
#   bash scripts/export_gguf.sh outputs/robotai4b-merged outputs/gguf [Q4_K_M Q4_0 ...]
#
# Çıktılar:
#   outputs/gguf/robotai4b-BF16.gguf      (ara dosya, büyük)
#   outputs/gguf/robotai4b-Q4_K_M.gguf    (Pi için önerilen: kalite/hız dengesi)
#   outputs/gguf/robotai4b-Q4_0.gguf      (ARM'de genelde en hızlısı)
#   outputs/gguf/robotai4b-mmproj.gguf    (isteğe bağlı: kamera/görüntü desteği)
set -euo pipefail

MERGED=${1:?birlesik model klasoru}
OUT=${2:-outputs/gguf}
shift $(( $# < 2 ? $# : 2 ))
QUANTS=("$@")
[ ${#QUANTS[@]} -eq 0 ] && QUANTS=(Q4_K_M)

LLAMA_DIR=${LLAMA_DIR:-third_party/llama.cpp}
NAME=${NAME:-robotai4b}

if [ ! -d "$LLAMA_DIR" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-quantize
fi
# Not: llama.cpp'nin kendi requirements dosyası transformers'ı 4.x'e sabitliyor; bu sürüm
# Gemma 4 tokenizer'ını okuyamıyor. Eğitim ortamındaki transformers 5.x ile dönüştürme çalışıyor,
# bu yüzden sadece eksik küçük paketleri kuruyoruz (gguf-py, llama.cpp klasöründen otomatik yüklenir).
python3 -m pip install -q sentencepiece protobuf numpy safetensors

mkdir -p "$OUT"
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" --outtype bf16 --outfile "$OUT/$NAME-BF16.gguf"

# Görüntü projektörü (Gemma 4 görebilir; robot kamerası için). Başarısız olursa atla.
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" --mmproj --outtype f16 \
  --outfile "$OUT/$NAME-mmproj.gguf" || echo "(mmproj oluşturulamadı, atlanıyor)"

for Q in "${QUANTS[@]}"; do
  "$LLAMA_DIR/build/bin/llama-quantize" "$OUT/$NAME-BF16.gguf" "$OUT/$NAME-$Q.gguf" "$Q"
done
ls -lh "$OUT"
