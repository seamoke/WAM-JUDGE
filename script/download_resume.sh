#!/usr/bin/env bash
# Resume LIBERO training asset downloads (survives SSH disconnect via tmux/nohup).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/logs/download_resume.log"
PID_FILE="${ROOT}/logs/download_resume.pid"

mkdir -p "${ROOT}/logs"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') download resume start =====" | tee -a "$LOG"

# Step 2: base model (ModelScope -> hf-mirror)
if [[ ! -f "${ROOT}/checkpoints/lingbot-va-base/transformer/config.json" ]] || \
   [[ ! -f "${ROOT}/checkpoints/lingbot-va-base/transformer/diffusion_pytorch_model-00001-of-00003.safetensors" ]]; then
  echo "[model] downloading lingbot-va-base..." | tee -a "$LOG"
  bash script/download_cn.sh lingbot-va-base "${ROOT}/checkpoints/lingbot-va-base" 2>&1 | tee -a "$LOG"
else
  echo "[model] lingbot-va-base looks complete, skip" | tee -a "$LOG"
fi

# attn_mode for training
CONFIG_JSON="${ROOT}/checkpoints/lingbot-va-base/transformer/config.json"
if [[ -f "$CONFIG_JSON" ]]; then
  python - <<PY | tee -a "$LOG"
import json
p = "${CONFIG_JSON}"
with open(p) as f:
    cfg = json.load(f)
cfg["attn_mode"] = "flex"
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print("attn_mode -> flex")
PY
fi

# Step 3: dataset
if ! find "${ROOT}/data/libero-long-lerobot" -name 'info.json' 2>/dev/null | grep -q .; then
  echo "[dataset] downloading libero-long-lerobot..." | tee -a "$LOG"
  bash script/download_cn.sh libero-long-lerobot "${ROOT}/data/libero-long-lerobot" 2>&1 | tee -a "$LOG"
else
  echo "[dataset] libero-long-lerobot looks complete, skip" | tee -a "$LOG"
fi

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') download resume done =====" | tee -a "$LOG"
rm -f "$PID_FILE"
