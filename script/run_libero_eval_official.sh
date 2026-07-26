#!/usr/bin/bash
# Official lingbot-va-posttrain-libero-long: 4 LIBERO suites in parallel.
# GPU0 spatial | GPU1 object | GPU2 goal | GPU3 libero_10 (Long)
#
# Usage:
#   bash script/run_libero_eval_official.sh
#   TEST_NUM=1 bash script/run_libero_eval_official.sh   # quick sanity (default)
#   # Fast sanity (~5–10 min): fewer denoise steps + 1 task per suite
#   EVAL_CONFIG_NAME=libero_eval TASK_RANGE="0 1" TEST_NUM=1 bash script/run_libero_eval_official.sh
#   TEST_NUM=5 bash script/run_libero_eval_official.sh
#   TEST_NUM=50 bash script/run_libero_eval_official.sh  # full eval
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

OFFICIAL_MODEL=${OFFICIAL_MODEL:-"/data/lingbot-va/models/lingbot-va-posttrain-libero-long"}
TEST_NUM=${TEST_NUM:-1}
EVAL_CONFIG_NAME=${EVAL_CONFIG_NAME:-libero}
TASK_RANGE=${TASK_RANGE:-0 10}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-800}
START_PORT=${START_PORT:-29056}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29561}
SERVER_WARMUP_SEC=${SERVER_WARMUP_SEC:-8}

LOG_DIR="${ROOT}/logs/libero_eval"
RESULTS_ROOT="${ROOT}/train_out/libero/eval_results/official_posttrain"
VIS_ROOT="${ROOT}/train_out/libero/eval_visualization/official_posttrain"
BATCH_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/official_posttrain_${BATCH_TIME}.log"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${VIS_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export WAN_VA_MODEL_PATH="${OFFICIAL_MODEL}"
[[ -f "${ROOT}/script/.libero_eval_env" ]] && source "${ROOT}/script/.libero_eval_env"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

if [[ ! -f "${OFFICIAL_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json" ]]; then
  echo "Official model not found: ${OFFICIAL_MODEL}" >&2
  exit 1
fi

BENCHMARKS=(libero_spatial libero_object libero_goal libero_10)

SERVER_PIDS=""
CLIENT_PIDS=""
cleanup() {
  for pid in ${CLIENT_PIDS} ${SERVER_PIDS}; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

{
  echo "Official model : ${OFFICIAL_MODEL}"
  echo "Benchmarks     : ${BENCHMARKS[*]}"
  echo "TEST_NUM       : ${TEST_NUM} per task"
  echo "EVAL_CONFIG    : ${EVAL_CONFIG_NAME}"
  echo "TASK_RANGE     : ${TASK_RANGE}"
  echo "MAX_ENV_STEPS  : ${MAX_ENV_STEPS}"
  echo "Results        : ${RESULTS_ROOT}"
} | tee -a "${LOG_FILE}"

for gpu_id in 0 1 2 3; do
  benchmark="${BENCHMARKS[$gpu_id]}"
  port=$((START_PORT + gpu_id))
  master_port=$((MASTER_PORT_BASE + gpu_id))
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  nohup python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${master_port}" \
    wan_va/wan_va_server.py \
    --config-name "${EVAL_CONFIG_NAME}" \
    --port "${port}" \
    --save_root "${VIS_ROOT}/${benchmark}/" \
    >> "${LOG_FILE}" 2>&1 &
  SERVER_PIDS="${SERVER_PIDS} $!"
  echo "Server GPU${gpu_id} benchmark=${benchmark} port=${port}" | tee -a "${LOG_FILE}"
  sleep 3
done

for gpu_id in 0 1 2 3; do
  port=$((START_PORT + gpu_id))
  elapsed=0
  while (( elapsed < 600 )); do
    if python -c "import socket,sys; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', int(sys.argv[1]))); s.close()" "${port}" 2>/dev/null; then
      echo "Port ${port} ready (${elapsed}s)" | tee -a "${LOG_FILE}"
      break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
done
sleep "${SERVER_WARMUP_SEC}"

for gpu_id in 0 1 2 3; do
  benchmark="${BENCHMARKS[$gpu_id]}"
  port=$((START_PORT + gpu_id))
  client_log="${LOG_DIR}/client_official_${benchmark}_${BATCH_TIME}.log"
  env -u PYOPENGL_PLATFORM MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  nohup python evaluation/libero/client.py \
    --libero-benchmark "${benchmark}" \
    --port "${port}" \
    --test-num "${TEST_NUM}" \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --task-range ${TASK_RANGE} \
    --out-dir "${RESULTS_ROOT}" \
    > "${client_log}" 2>&1 &
  CLIENT_PIDS="${CLIENT_PIDS} $!"
  echo "Client GPU${gpu_id} benchmark=${benchmark} port=${port} log=${client_log}" | tee -a "${LOG_FILE}"
done

for pid in ${CLIENT_PIDS}; do
  wait "${pid}" || true
done

cleanup
trap - EXIT INT TERM

python evaluation/libero/collect_results.py \
  --results-root "${RESULTS_ROOT}" \
  --out-csv "${RESULTS_ROOT}/results.csv" \
  --out-md "${RESULTS_ROOT}/results.md"

echo "Official eval done: ${RESULTS_ROOT}/results.md" | tee -a "${LOG_FILE}"
