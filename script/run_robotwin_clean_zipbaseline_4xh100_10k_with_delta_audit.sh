#!/usr/bin/env bash
set -euo pipefail

ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
RUN_ROOT="${ROOT}/train_out/robotwin/${RUN_ID}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
AUDIT_ROOT="${RUN_ROOT}/checkpoint_delta_vs_base"
BASE_MODEL="${ROOT}/models/lingbot-va-base"
DATASET="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
EMPTY_EMB="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt"
TRAIN_SCRIPT="${ROOT}/code/script/run_robotwin_clean_zipbaseline_4xh100_10k.sh"
AUDIT_SCRIPT="${ROOT}/code/script/compare_checkpoint_to_base.py"
PYTHON="${ROOT}/code/.venv/bin/python"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_ROOT}" >&2
  exit 2
fi
test -f "${TRAIN_SCRIPT}"
test -f "${AUDIT_SCRIPT}"
test -x "${PYTHON}"
test -s "${BASE_MODEL}/transformer/config.json"
test -d "${DATASET}"
test -s "${EMPTY_EMB}"
if ! compgen -G "${BASE_MODEL}/transformer/*.safetensors" >/dev/null; then
  echo "No base transformer safetensors found under ${BASE_MODEL}/transformer" >&2
  exit 2
fi
"${PYTHON}" -c "import safetensors, torch"

mkdir -p "${ROOT}/logs"
cd "${ROOT}/code"

(
  for _ in $(seq 1 120); do
    if [[ -d "${RUN_ROOT}" ]]; then
      break
    fi
    sleep 1
  done
  if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "Training did not create its run directory within 120 seconds." >&2
    exit 3
  fi

  mkdir -p "${AUDIT_ROOT}"
  printf '%s\n' "$$" > "${AUDIT_ROOT}/watcher.pid"
  exec "${PYTHON}" "${AUDIT_SCRIPT}" watch \
    --base "${BASE_MODEL}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --steps 2000,4000,6000,8000,10000 \
    --output "${AUDIT_ROOT}" \
    --poll-seconds 60 \
    --stable-polls 2
) > "${ROOT}/logs/${RUN_ID}.watcher-bootstrap.log" 2>&1 &

# Keep the reference launcher's process structure intact. Making the training
# script a background child can stall its multiprocessing dataset Pool.
exec env RUN_ID="${RUN_ID}" bash "${TRAIN_SCRIPT}"
