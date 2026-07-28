#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "${CODE_ROOT}")}"
BASE_MODEL="${BASE_MODEL:-${LINGBOT_ROOT}/models/lingbot-va-base}"
OFFICIAL_MODEL="${OFFICIAL_MODEL:-${LINGBOT_ROOT}/models/lingbot-va-posttrain-robotwin}"
MODEL_KIND="${MODEL_KIND:-official}"
MODEL_PATH="${MODEL_PATH:-${OFFICIAL_MODEL}}"
RESULT_LABEL="${RESULT_LABEL:-$(basename "${MODEL_PATH}")}"
TEST_NUM="${TEST_NUM:-10}"
SEED_CACHE="${SEED_CACHE:-${CODE_ROOT}/evaluation/robotwin/seed_cache/demo_clean_seed0_n100.json}"
RT_DENOISER="${ROBOTWIN_RT_DENOISER:-optix}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
PROMPT_SERVICE_GPU="${PROMPT_SERVICE_GPU:-0}"
PROMPT_SERVICE_PORT="${PROMPT_SERVICE_PORT:-31056}"
RUN_ID="${RUN_ID:-robotwin_clean_${RESULT_LABEL}_t50_n${TEST_NUM}_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LINGBOT_ROOT}/train_out/robotwin-clean-eval/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${LINGBOT_ROOT}/logs/robotwin-clean-eval/${RUN_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-${RUN_ROOT}/results}"
PROMPT_CACHE="${PROMPT_CACHE:-${LINGBOT_ROOT}/train_out/robotwin/prompt_embed_cache/demo_clean_n${TEST_NUM}_all50}"
EVAL_MODEL_CACHE="${EVAL_MODEL_CACHE:-${RUN_ROOT}/eval_models}"
START_PORT="${START_PORT:-35056}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-35161}"

TASKS="$(
  python3 - "${SEED_CACHE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
tasks = list(payload.get("tasks", {}))
if payload.get("task_config") != "demo_clean" or len(tasks) != 50:
    raise SystemExit(
        f"Expected the aligned 50-task demo_clean seed cache, got "
        f"task_config={payload.get('task_config')!r}, tasks={len(tasks)}"
    )
print(",".join(tasks))
PY
)"

if [[ "${TEST_NUM}" -ne 10 ]]; then
  echo "This aligned formal entry point requires TEST_NUM=10." >&2
  exit 2
fi
if [[ "${NGPU}" -lt 1 ]]; then
  echo "No visible GPU found." >&2
  exit 2
fi
if [[ "${RESULT_LABEL}" != "$(basename "${MODEL_PATH}")" ]]; then
  echo "RESULT_LABEL must match the model/checkpoint directory basename." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_ROOT}" >&2
  exit 2
fi
test -f "${CODE_ROOT}/.venv/bin/activate"
test -s "${BASE_MODEL}/transformer/config.json"
test -s "${OFFICIAL_MODEL}/transformer/config.json"
test -s "${SEED_CACHE}"
test -d "${MODEL_PATH}"

if pgrep -af "[r]un_robotwin_eval.sh|[r]un_server_ckpt.py|[e]val_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin evaluation is active." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${RESULTS_ROOT}" "${EVAL_MODEL_CACHE}"
source "${CODE_ROOT}/.venv/bin/activate"
cd "${CODE_ROOT}"

python - "${OFFICIAL_MODEL}/transformer/config.json" "${SEED_CACHE}" "${TASKS}" <<'PY'
import json
import sys

config_path, seed_path, task_csv = sys.argv[1:]
config = json.load(open(config_path, encoding="utf-8"))
if config.get("attn_mode") not in {"torch", "flashattn"}:
    raise SystemExit(f"Official eval model has invalid attn_mode={config.get('attn_mode')!r}")
payload = json.load(open(seed_path, encoding="utf-8"))
tasks = task_csv.split(",")
if (
    payload.get("task_config") != "demo_clean"
    or len(tasks) != 50
    or len(set(tasks)) != 50
    or set(payload["tasks"]) != set(tasks)
):
    raise SystemExit("Seed cache does not match the aligned 50-task demo_clean protocol")
for task in tasks:
    rows = payload["tasks"][task]
    seeds = [int(row["seed"]) for row in rows[:10]]
    if len(rows) < 10 or len(seeds) != len(set(seeds)):
        raise SystemExit(f"Invalid seed cache rows for {task}")
print("MODEL_AND_SEED_PREFLIGHT_OK")
PY

if [[ ! -s "${PROMPT_CACHE}/manifest.json" ]] || \
   ! python - "${PROMPT_CACHE}/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("complete")
assert payload.get("task_config") == "demo_clean"
assert int(payload.get("test_num", 0)) >= 10
assert int(payload.get("tasks", 0)) == 50
PY
then
  mkdir -p "${PROMPT_CACHE}"
  CUDA_VISIBLE_DEVICES="${PROMPT_SERVICE_GPU}" \
    python -u script/precompute_robotwin_prompt_embeddings.py \
      --model-path "${OFFICIAL_MODEL}" \
      --seed-cache "${SEED_CACHE}" \
      --robotwin-root "${CODE_ROOT}/third_party/RoboTwin" \
      --output-dir "${PROMPT_CACHE}" \
      --test-num 10 \
      --enumerate-all-seen \
      --device cuda:0 \
      | tee "${LOG_DIR}/prompt_precompute.log"
fi

PROMPT_SERVICE_PID=""
cleanup() {
  if [[ -n "${PROMPT_SERVICE_PID}" ]] && kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
    kill -TERM "${PROMPT_SERVICE_PID}" 2>/dev/null || true
    wait "${PROMPT_SERVICE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${PROMPT_SERVICE_GPU}" \
PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" \
  python -u script/robotwin_prompt_embedding_service.py \
    --model-path "${OFFICIAL_MODEL}" \
    --cache-dir "${PROMPT_CACHE}" \
    --host 127.0.0.1 \
    --port "${PROMPT_SERVICE_PORT}" \
    --device cuda:0 \
    > "${LOG_DIR}/prompt_service.log" 2>&1 &
PROMPT_SERVICE_PID=$!

prompt_service_ready=0
for _ in $(seq 1 150); do
  if ! kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
    tail -n 100 "${LOG_DIR}/prompt_service.log" >&2
    exit 1
  fi
  if python - "${PROMPT_SERVICE_PORT}" <<'PY'
import json
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1) as sock:
    sock.sendall(b'{"op":"ping"}\n')
    response = json.loads(sock.makefile("rb").readline())
assert response.get("ok") and response.get("status") == "ready"
PY
  then
    prompt_service_ready=1
    break
  fi
  sleep 2
done
if [[ "${prompt_service_ready}" -ne 1 ]]; then
  echo "Prompt embedding service did not become ready." >&2
  tail -n 100 "${LOG_DIR}/prompt_service.log" >&2
  exit 1
fi

FULL_MODEL_PATH=""
CHECKPOINT_PATH="${MODEL_PATH}"
if [[ "${MODEL_KIND}" == "official" || "${MODEL_KIND}" == "full" ]]; then
  FULL_MODEL_PATH="${MODEL_PATH}"
  python - "${MODEL_PATH}/transformer/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
if config.get("attn_mode") not in {"torch", "flashattn"}:
    raise SystemExit(
        f"Full eval model requires attn_mode=torch or flashattn, "
        f"got {config.get('attn_mode')!r}"
    )
PY
elif [[ "${MODEL_KIND}" != "checkpoint" ]]; then
  echo "MODEL_KIND must be official, full, or checkpoint." >&2
  exit 2
fi

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'model_kind=%s\n' "${MODEL_KIND}"
  printf 'model_path=%s\n' "${MODEL_PATH}"
  printf 'result_label=%s\n' "${RESULT_LABEL}"
  printf 'task_config=demo_clean\n'
  printf 'tasks=50\n'
  printf 'episodes_per_task=10\n'
  printf 'ngpu=%s\n' "${NGPU}"
  printf 'rt_denoiser=%s\n' "${RT_DENOISER}"
  printf 'seed_cache_sha256=%s\n' "$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
  printf 'render=fast0_low_render0_policy0_defer0_recreate0\n'
} > "${RUN_ROOT}/manifest.txt"

set +e
FULL_MODEL_PATH="${FULL_MODEL_PATH}" \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
BASE_MODEL="${BASE_MODEL}" \
EVAL_MODEL_CACHE="${EVAL_MODEL_CACHE}" \
EVAL_CONFIG_NAME=robotwin \
TASK_CONFIG=demo_clean \
RUN_HARD=0 \
NGPU="${NGPU}" \
CLIENTS_PER_GPU=1 \
ROBOTWIN_DYNAMIC_SHARDS=1 \
ROBOTWIN_SIM_FOLLOWS_SERVER_GPU=1 \
ROBOTWIN_TASKS="${TASKS}" \
TEST_NUM=10 \
SEED=0 \
ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
ROBOTWIN_EXPERT_CHECK=1 \
ROBOTWIN_FAST=0 \
ROBOTWIN_EVAL_LOW_RENDER=0 \
ROBOTWIN_POLICY_CAMERAS_ONLY=0 \
ROBOTWIN_DEFER_RENDER_UPDATES=0 \
ROBOTWIN_RECREATE_CAMERAS_EVERY=0 \
ROBOTWIN_RT_DENOISER="${RT_DENOISER}" \
ROBOTWIN_EVAL_TIMING=1 \
ROBOTWIN_RESUME_PARTIAL=1 \
ROBOTWIN_CLIENT_EPISODE_CHUNK=1 \
ROBOTWIN_CLIENT_TIMEOUT_SEC=900 \
ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES=3 \
ROBOTWIN_EVAL_VIDEO_LOG=0 \
ROBOTWIN_SAVE_COMPARISON_VIDEO=0 \
ROBOTWIN_SAVE_VISUALIZATION=0 \
WAN_VA_ENABLE_OFFLOAD=0 \
WAN_VA_OFFLOAD_VAE=0 \
WAN_VA_OFFLOAD_TEXT_ENCODER=1 \
WAN_VA_SWAP_TEXT_ENCODER_FOR_PROMPT=0 \
WAN_VA_PROMPT_EMBED_CACHE_DIR="${PROMPT_CACHE}" \
WAN_VA_PROMPT_EMBED_CACHE_STRICT=1 \
WAN_VA_PROMPT_EMBED_SERVICE="127.0.0.1:${PROMPT_SERVICE_PORT}" \
WAN_VA_PROMPT_EMBED_SERVICE_TIMEOUT=300 \
WAN_VA_SAVE_INFER_DEBUG=0 \
STRICT_CUROBO_COMPAT=1 \
SKIP_EXISTING=1 \
MAX_CHECKPOINTS=1 \
START_PORT="${START_PORT}" \
MASTER_PORT_BASE="${MASTER_PORT_BASE}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
LOG_DIR="${LOG_DIR}" \
VIS_ROOT="${RUN_ROOT}/visualization" \
  bash script/run_robotwin_eval.sh
rc=$?
set -e
printf '%s\n' "${rc}" > "${RUN_ROOT}/exit_code"
if [[ "${rc}" -ne 0 ]]; then
  exit "${rc}"
fi

python script/audit_robotwin_clean_eval.py \
  --results-root "${RESULTS_ROOT}" \
  --label "${RESULT_LABEL}" \
  --seed-cache "${SEED_CACHE}" \
  --episodes 10 \
  --output "${RUN_ROOT}/summary.json"
echo "EVAL_DONE ${RUN_ROOT}"
