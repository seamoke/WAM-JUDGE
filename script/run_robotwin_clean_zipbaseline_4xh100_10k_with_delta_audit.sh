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
command -v setsid >/dev/null
if ! compgen -G "${BASE_MODEL}/transformer/*.safetensors" >/dev/null; then
  echo "No base transformer safetensors found under ${BASE_MODEL}/transformer" >&2
  exit 2
fi
"${PYTHON}" -c "import safetensors, torch"

mkdir -p "${ROOT}/logs"
cd "${ROOT}/code"

setsid env RUN_ID="${RUN_ID}" bash "${TRAIN_SCRIPT}" &
train_pid=$!
printf '%s\n' "${train_pid}" > "${ROOT}/logs/${RUN_ID}.train.pid"

for _ in $(seq 1 120); do
  if [[ -d "${RUN_ROOT}" ]]; then
    break
  fi
  if ! kill -0 "${train_pid}" 2>/dev/null; then
    wait "${train_pid}"
    exit $?
  fi
  sleep 1
done
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Training did not create its run directory within 120 seconds." >&2
  exit 3
fi

mkdir -p "${AUDIT_ROOT}"
nohup "${PYTHON}" "${AUDIT_SCRIPT}" watch \
  --base "${BASE_MODEL}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --steps 2000,4000,6000,8000,10000 \
  --output "${AUDIT_ROOT}" \
  --poll-seconds 60 \
  --stable-polls 2 \
  > "${AUDIT_ROOT}/watcher.log" 2>&1 &
watcher_pid=$!
printf '%s\n' "${watcher_pid}" > "${AUDIT_ROOT}/watcher.pid"
sleep 2
if ! kill -0 "${watcher_pid}" 2>/dev/null; then
  wait "${watcher_pid}" || true
  echo "Checkpoint delta watcher failed during startup." >&2
  tail -n 100 "${AUDIT_ROOT}/watcher.log" >&2
  kill -TERM -- "-${train_pid}" 2>/dev/null || true
  wait "${train_pid}" 2>/dev/null || true
  exit 4
fi

set +e
wait "${train_pid}"
train_rc=$?
set -e
printf '%s\n' "${train_rc}" > "${RUN_ROOT}/train_exit_code"

if [[ "${train_rc}" -ne 0 ]]; then
  kill -TERM "${watcher_pid}" 2>/dev/null || true
  wait "${watcher_pid}" 2>/dev/null || true
  printf '%s\n' "terminated_after_training_failure" > "${AUDIT_ROOT}/watcher_exit_code"
  echo "Training failed with rc=${train_rc}; stopped this run's delta watcher." >&2
  exit "${train_rc}"
fi

set +e
wait "${watcher_pid}"
watcher_rc=$?
set -e
printf '%s\n' "${watcher_rc}" > "${AUDIT_ROOT}/watcher_exit_code"
if [[ "${watcher_rc}" -ne 0 ]]; then
  echo "Training completed, but checkpoint delta watcher failed with rc=${watcher_rc}." >&2
  exit "${watcher_rc}"
fi

echo "Training and all five checkpoint delta audits completed."
