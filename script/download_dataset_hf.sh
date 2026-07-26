#!/usr/bin/env bash
# Full LIBERO dataset via hf-mirror (ModelScope snapshot was incomplete).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/logs/dataset_hf_download.log"
OUT="${ROOT}/data/libero-long-lerobot"
mkdir -p "$ROOT/logs"
export PATH="${HOME}/.local/bin:${PATH}"
source "${ROOT}/.venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') hf-mirror dataset download =====" | tee -a "$LOG"
huggingface-cli download robbyant/libero-long-lerobot \
  --repo-type dataset \
  --local-dir "$OUT" \
  --resume-download 2>&1 | tee -a "$LOG"
echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') dataset download done =====" | tee -a "$LOG"
