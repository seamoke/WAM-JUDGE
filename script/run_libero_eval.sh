#!/usr/bin/bash
# Evaluate all LIBERO training checkpoints on libero_10 (LIBERO-Long).
# Standard eval: 50 rollouts/task, 800 max env steps, multi-GPU server + client sharding.
#
# Usage:
#   bash script/setup_libero_eval.sh          # first time only
#   bash script/run_libero_eval.sh
#   TEST_NUM=50 bash script/run_libero_eval.sh
#   MAX_CHECKPOINTS=1 bash script/run_libero_eval.sh
#   SKIP_EXISTING=1 bash script/run_libero_eval.sh
#   CHECKPOINT_PATH=train_out/libero/checkpoints/checkpoint_step_10000 bash script/run_libero_eval.sh
#   LIBERO_FAST=1 bash script/run_libero_eval.sh   # libero_eval config, no video
#   LIBERO_EVAL_TIMING=1 bash script/run_libero_eval.sh  # per-episode infer vs sim timing
#   CLIENTS_PER_GPU=3 bash script/run_libero_eval.sh  # 12-way shards, 3 servers+clients/GPU
#
# Before running, stop GPU guard if active:
#   kill $(cat logs/gpu_occupy.pid) 2>/dev/null; pkill -f gpu_guard.sh 2>/dev/null
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

NGPU=${NGPU:-4}
TEST_NUM=${TEST_NUM:-50}
EVAL_CONFIG_NAME=${EVAL_CONFIG_NAME:-libero}
LIBERO_BENCHMARK=${LIBERO_BENCHMARK:-libero_10}
SAVE_VIDEO=${SAVE_VIDEO:-0}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-800}
LIBERO_RENDER_GL=${LIBERO_RENDER_GL:-egl}
START_PORT=${START_PORT:-29056}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29561}
SERVER_WARMUP_SEC=${SERVER_WARMUP_SEC:-8}
SKIP_EXISTING=${SKIP_EXISTING:-0}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
LIBERO_FAST=${LIBERO_FAST:-0}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-1}
CLIENT_STAGGER_SEC=${CLIENT_STAGGER_SEC:-0}

if [[ "${LIBERO_FAST}" == "1" ]]; then
  if [[ "${EVAL_CONFIG_NAME}" == "libero" ]]; then
    EVAL_CONFIG_NAME=libero_eval
  fi
  SAVE_VIDEO=0
fi

export LIBERO_EVAL_TIMING

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${ROOT}/train_out/libero/checkpoints"}
BASE_MODEL=${BASE_MODEL:-"${ROOT}/checkpoints/lingbot-va-base"}
EVAL_MODEL_CACHE=${EVAL_MODEL_CACHE:-"${TMPDIR:-/tmp}/lingbot_eval_symlinks"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT}/train_out/libero/eval_results"}
LOG_DIR=${LOG_DIR:-"${ROOT}/logs/libero_eval"}
VIS_ROOT=${VIS_ROOT:-"${ROOT}/train_out/libero/eval_visualization"}

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${EVAL_MODEL_CACHE}" "${VIS_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export EVAL_MODEL_CACHE
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.libero_eval_env" ]] && source "${ROOT}/script/.libero_eval_env"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

NUM_GPUS_SHARD=${NGPU}
NUM_SHARDS=$((NUM_GPUS_SHARD * CLIENTS_PER_GPU))
NUM_TASKS=10

if [[ "${CLIENTS_PER_GPU}" -lt 1 ]]; then
  echo "CLIENTS_PER_GPU must be >= 1, got ${CLIENTS_PER_GPU}" >&2
  exit 1
fi

shard_tasks() {
  local shard_id=$1
  local -a out=()
  local i
  for ((i = shard_id; i < NUM_TASKS; i += NUM_SHARDS)); do
    out+=("${i}")
  done
  printf '%s\n' "${out[@]}"
}

gpu_for_shard() {
  echo $(( $1 % NUM_GPUS_SHARD ))
}

client_slot_for_shard() {
  echo $(( $1 / NUM_GPUS_SHARD ))
}

port_for_shard() {
  local shard_id=$1
  local gpu_id client_slot
  gpu_id="$(gpu_for_shard "${shard_id}")"
  client_slot="$(client_slot_for_shard "${shard_id}")"
  echo $((START_PORT + gpu_id * CLIENTS_PER_GPU + client_slot))
}

master_port_for_shard() {
  echo $((MASTER_PORT_BASE + $1))
}

task_result_complete() {
  local result_file=$1
  local expected=${2:-${TEST_NUM}}
  python - <<PY "${result_file}" "${expected}"
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    with open(path) as f:
        data = json.load(f)
    total = int(data.get("total_num", 0))
except (OSError, ValueError, TypeError):
    sys.exit(1)
sys.exit(0 if total >= expected else 1)
PY
}

preflight() {
  if ! python -c "from libero.libero import benchmark" 2>/dev/null; then
    echo "LIBERO not installed. Run: bash script/setup_libero_eval.sh" >&2
    exit 1
  fi
  if ! python -c '
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.pop("PYOPENGL_PLATFORM", None)
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
' 2>/dev/null && ! python -c '
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
' 2>/dev/null; then
    echo "MuJoCo offscreen render unavailable. Run: bash script/setup_libero_eval.sh" >&2
    exit 1
  fi
  if [[ ! -d "${BASE_MODEL}/vae" || ! -d "${BASE_MODEL}/text_encoder" ]]; then
    echo "Base model incomplete: ${BASE_MODEL}" >&2
    exit 1
  fi
  local gpu_count
  gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "${gpu_count}" -lt "${NUM_GPUS_SHARD}" ]]; then
    echo "Need >= ${NUM_GPUS_SHARD} GPUs, found ${gpu_count}" >&2
    exit 1
  fi
  if [[ "${NGPU}" -lt "${NUM_GPUS_SHARD}" ]]; then
    echo "NGPU=${NGPU} < ${NUM_GPUS_SHARD}: set NGPU=${NUM_GPUS_SHARD}" >&2
    exit 1
  fi
}

checkpoint_done() {
  local result_dir=$1
  local task_idx
  local found=0
  for task_idx in $(seq 0 $((NUM_TASKS - 1))); do
    local res_json="${result_dir}/${LIBERO_BENCHMARK}_${task_idx}.json"
    if [[ -f "${res_json}" ]] && task_result_complete "${res_json}"; then
      found=$((found + 1))
    fi
  done
  [[ "${found}" -ge "${NUM_TASKS}" ]]
}

wait_for_port() {
  local port=$1
  local timeout=${2:-600}
  local server_pids=${3:-}
  local elapsed=0
  while (( elapsed < timeout )); do
    if [[ -n "${server_pids}" ]]; then
      local pid
      for pid in ${server_pids}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          echo "Server pid ${pid} exited before port ${port} was ready. Check ${LOG_DIR}" >&2
          return 1
        fi
      done
    fi
    if python - <<PY "${port}"
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
    then
      echo "Port ${port} ready (${elapsed}s)"
      return 0
    fi
    if (( elapsed > 0 && elapsed % 30 == 0 )); then
      echo "  Waiting for port ${port} ... ${elapsed}s"
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "Timed out waiting for port ${port}. Check ${LOG_DIR}" >&2
  return 1
}

run_shard_client() {
  local shard_id=$1
  local gpu_id=$2
  local port=$3
  local ckpt_result_dir=$4
  local shard_log=$5
  shift 5
  local -a shard_task_list=("$@")

  (
    if [[ "${CLIENT_STAGGER_SEC}" -gt 0 ]]; then
      local client_slot
      client_slot="$(client_slot_for_shard "${shard_id}")"
      sleep $((client_slot * CLIENT_STAGGER_SEC))
    fi
    local -a client_extra_args=()
    [[ "${SAVE_VIDEO}" == "1" ]] && client_extra_args+=(--save-video)
    local task_idx
    for task_idx in "${shard_task_list[@]}"; do
      if [[ "${SKIP_EXISTING}" == "1" ]]; then
        local res_json="${ckpt_result_dir}/${LIBERO_BENCHMARK}_${task_idx}.json"
        if [[ -f "${res_json}" ]] && task_result_complete "${res_json}"; then
          echo "shard${shard_id}/GPU${gpu_id} skip task=${task_idx} (complete json)"
          continue
        fi
      fi
      echo "shard${shard_id}/GPU${gpu_id} task=${task_idx} port=${port}"
      if [[ "${LIBERO_RENDER_GL}" == "egl" ]]; then
        env -u PYOPENGL_PLATFORM MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
        python evaluation/libero/client.py \
          --libero-benchmark "${LIBERO_BENCHMARK}" \
          --port "${port}" \
          --test-num "${TEST_NUM}" \
          --max-env-steps "${MAX_ENV_STEPS}" \
          --task-range "${task_idx}" $((task_idx + 1)) \
          --out-dir "${ckpt_result_dir}" \
          "${client_extra_args[@]}"
      else
        env -u PYOPENGL_PLATFORM -u MUJOCO_EGL_DEVICE_ID MUJOCO_GL=osmesa \
        python evaluation/libero/client.py \
          --libero-benchmark "${LIBERO_BENCHMARK}" \
          --port "${port}" \
          --test-num "${TEST_NUM}" \
          --max-env-steps "${MAX_ENV_STEPS}" \
          --task-range "${task_idx}" $((task_idx + 1)) \
          --out-dir "${ckpt_result_dir}" \
          "${client_extra_args[@]}"
      fi
    done
  ) > "${shard_log}" 2>&1 &
  CLIENT_LAST_PID=$!
}

stop_servers() {
  if [[ -n "${SERVER_PIDS:-}" ]]; then
    # shellcheck disable=SC2086
    kill ${SERVER_PIDS} 2>/dev/null || true
    wait ${SERVER_PIDS} 2>/dev/null || true
  fi
  SERVER_PIDS=""
}

trap stop_servers EXIT INT TERM

preflight

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "CHECKPOINT_PATH not found: ${CHECKPOINT_PATH}" >&2
    exit 1
  fi
  CHECKPOINTS=("$(cd "${CHECKPOINT_PATH}" && pwd)")
else
  mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'checkpoint_step_*' | sort -V)
  if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No checkpoints found under ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
  if [[ "${MAX_CHECKPOINTS}" -gt 0 ]] && [[ ${#CHECKPOINTS[@]} -gt ${MAX_CHECKPOINTS} ]]; then
    CHECKPOINTS=("${CHECKPOINTS[@]:0:MAX_CHECKPOINTS}")
  fi
fi

echo "Checkpoints : ${#CHECKPOINTS[@]} under ${CHECKPOINT_DIR}"
echo "Benchmark   : ${LIBERO_BENCHMARK}"
echo "TEST_NUM    : ${TEST_NUM} per task"
echo "LIBERO_FAST : ${LIBERO_FAST} (config=${EVAL_CONFIG_NAME}, video=${SAVE_VIDEO})"
echo "Client stagger: ${CLIENT_STAGGER_SEC}s per slot (0=off)"
echo "Config      : ${EVAL_CONFIG_NAME}"
echo "GPUs        : 0-$((NUM_GPUS_SHARD - 1)) (${NUM_SHARDS} shards, ${CLIENTS_PER_GPU} server(s)/GPU)"
echo "Symlink cache: ${EVAL_MODEL_CACHE}"

BATCH_TIME=$(date +%Y%m%d_%H%M%S)

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
  CKPT_NAME="$(basename "${CKPT_PATH}")"
  CKPT_RESULT_DIR="${RESULTS_ROOT}/${CKPT_NAME}"
  CKPT_VIS_DIR="${VIS_ROOT}/${CKPT_NAME}"
  mkdir -p "${CKPT_RESULT_DIR}" "${CKPT_VIS_DIR}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && checkpoint_done "${CKPT_RESULT_DIR}"; then
    echo "========== Skip ${CKPT_NAME} (results complete) =========="
    continue
  fi

  echo "========== Evaluating ${CKPT_NAME} =========="

  SERVER_PIDS=""
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu_id="$(gpu_for_shard "${shard_id}")"
    port="$(port_for_shard "${shard_id}")"
    master_port="$(master_port_for_shard "${shard_id}")"
    client_slot="$(client_slot_for_shard "${shard_id}")"
    log_file="${LOG_DIR}/server_${CKPT_NAME}_${LIBERO_BENCHMARK}_${BATCH_TIME}_shard${shard_id}_gpu${gpu_id}_c${client_slot}.log"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    nohup python -m torch.distributed.run \
      --nproc_per_node=1 \
      --master_port="${master_port}" \
      evaluation/libero/run_server_ckpt.py \
      --checkpoint "${CKPT_PATH}" \
      --base-model "${BASE_MODEL}" \
      --config-name "${EVAL_CONFIG_NAME}" \
      --port "${port}" \
      --save-root "${CKPT_VIS_DIR}/${LIBERO_BENCHMARK}" \
      > "${log_file}" 2>&1 &

    pid=$!
    SERVER_PIDS="${SERVER_PIDS} ${pid}"
    echo "Server shard${shard_id}/GPU${gpu_id} port=${port} master=${master_port} pid=${pid} log=${log_file}"
    wait_for_port "${port}" 600 "${pid}"
  done

  echo "All ${NUM_SHARDS} servers ready."
  sleep "${SERVER_WARMUP_SEC}"

  CLIENT_PIDS=""
  CLIENT_FAILED=0
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu_id="$(gpu_for_shard "${shard_id}")"
    port="$(port_for_shard "${shard_id}")"
    mapfile -t shard_task_list < <(shard_tasks "${shard_id}")
    if [[ ${#shard_task_list[@]} -eq 0 ]]; then
      continue
    fi
    shard_log="${LOG_DIR}/client_${CKPT_NAME}_${LIBERO_BENCHMARK}_shard${shard_id}_gpu${gpu_id}_${BATCH_TIME}.log"
    run_shard_client "${shard_id}" "${gpu_id}" "${port}" "${CKPT_RESULT_DIR}" \
      "${shard_log}" "${shard_task_list[@]}"
    pid="${CLIENT_LAST_PID}"
    CLIENT_PIDS="${CLIENT_PIDS} ${pid}"
    echo "Client shard${shard_id}/GPU${gpu_id} tasks=${#shard_task_list[@]} port=${port} log=${shard_log}"
  done

  # shellcheck disable=SC2086
  for pid in ${CLIENT_PIDS}; do
    if ! wait "${pid}"; then
      CLIENT_FAILED=1
    fi
  done
  if [[ "${CLIENT_FAILED}" -ne 0 ]]; then
    echo "WARNING: some clients failed for ${CKPT_NAME}, check ${LOG_DIR}" >&2
  fi
  echo "Finished ${CKPT_NAME}"

  stop_servers
  sleep 3
done

python evaluation/libero/collect_results.py \
  --results-root "${RESULTS_ROOT}" \
  --out-csv "${RESULTS_ROOT}/results.csv" \
  --out-md "${RESULTS_ROOT}/results.md"

echo "All checkpoints evaluated."
echo "Results: ${RESULTS_ROOT}/results.md"
