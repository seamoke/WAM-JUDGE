#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "${CODE_ROOT}")}"
OFFICIAL_MODEL="${OFFICIAL_MODEL:-${LINGBOT_ROOT}/models/lingbot-va-posttrain-robotwin}"
RUN_ID="${RUN_ID:-robotwin_clean_official_calibration_n20_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LINGBOT_ROOT}/train_out/robotwin-clean-calibration/${RUN_ID}}"

MODEL_KIND=official \
MODEL_PATH="${OFFICIAL_MODEL}" \
RESULT_LABEL=lingbot-va-posttrain-robotwin \
TEST_NUM=20 \
RUN_ID="${RUN_ID}" \
RUN_ROOT="${RUN_ROOT}" \
  bash "${CODE_ROOT}/script/run_robotwin_clean_eval_portable.sh"

python - "${RUN_ROOT}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
sr = float(summary["sr"])
if sr < 0.85:
    raise SystemExit(
        f"OFFICIAL_CALIBRATION_FAILED sr={sr:.4f} < 0.8500. "
        "Do not evaluate custom checkpoints until the environment/rendering is fixed."
    )
print(f"OFFICIAL_CALIBRATION_PASSED sr={sr:.4f}")
PY

printf '%s\n' "${RUN_ROOT}/summary.json" > "${LINGBOT_ROOT}/train_out/robotwin-clean-calibration/LATEST_PASSED"
