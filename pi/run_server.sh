#!/usr/bin/env bash
# RobotAI4B'yi OpenAI uyumlu bir HTTP sunucusu olarak başlatır (llama-server).
#   Sohbet API'si:  http://$HOST:$PORT/v1/chat/completions
#   Tarayıcı arayüzü: http://$HOST:$PORT/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/robotai4b.env"

MODEL="${MODEL_PATH:-$MODEL_DIR/$MODEL_FILE}"
[ -f "$MODEL" ] || { echo "Model bulunamadı: $MODEL (önce pi/setup_pi.sh çalıştırın)"; exit 1; }

EXTRA=()
if [ -n "$LORA_FILE" ]; then
  LORA="$MODEL_DIR/$LORA_FILE"
  [ -f "$LORA" ] || { echo "LoRA bulunamadı: $LORA"; exit 1; }
  EXTRA+=(--lora "$LORA")
fi

exec "$LLAMA_DIR/build/bin/llama-server" \
  "${EXTRA[@]}" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -t "$THREADS" \
  --jinja \
  --temp 0.7 --top-p 0.95 --top-k 64 \
  --reasoning off \
  "$@"
