#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "${CODE_ROOT}")}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to checkpoint_step_XXXX}"
CALIBRATION_SUMMARY="${CALIBRATION_SUMMARY:-}"

if [[ -z "${CALIBRATION_SUMMARY}" ]] && \
   [[ -s "${LINGBOT_ROOT}/train_out/robotwin-clean-calibration/LATEST_PASSED" ]]; then
  CALIBRATION_SUMMARY="$(cat "${LINGBOT_ROOT}/train_out/robotwin-clean-calibration/LATEST_PASSED")"
fi
if [[ -z "${CALIBRATION_SUMMARY}" || ! -s "${CALIBRATION_SUMMARY}" ]]; then
  echo "Missing passed official calibration summary. Run official calibration first." >&2
  exit 2
fi

python - "${CALIBRATION_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if int(summary["tasks"]) != 50 or int(summary["episodes_per_task"]) != 20:
    raise SystemExit("Calibration protocol is not 50 tasks x 20 episodes")
if float(summary["sr"]) < 0.85:
    raise SystemExit(f"Calibration SR {summary['sr']:.4f} is below 0.85")
print(f"Calibration gate accepted: SR={summary['sr']:.4f}")
PY

RESULT_LABEL="${RESULT_LABEL:-$(basename "${CHECKPOINT_PATH}")}"
MODEL_KIND=checkpoint \
MODEL_PATH="${CHECKPOINT_PATH}" \
RESULT_LABEL="${RESULT_LABEL}" \
TEST_NUM=20 \
  bash "${CODE_ROOT}/script/run_robotwin_clean_eval_portable.sh"
