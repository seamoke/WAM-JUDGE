#!/usr/bin/bash
# Evaluate RoboTwin training checkpoints (50 tasks, multi-GPU server + client sharding).
#
# Prereq: bash script/setup_robotwin_eval.sh
#
# Usage:
#   bash script/run_robotwin_eval.sh
#   TEST_NUM=20 bash script/run_robotwin_eval.sh
#   TASK_CONFIG=demo_randomized bash script/run_robotwin_eval.sh   # hard split
#   RUN_HARD=1 bash script/run_robotwin_eval.sh                    # easy + hard
#   CHECKPOINT_PATH=train_out/robotwin/checkpoints/checkpoint_step_8000 bash script/run_robotwin_eval.sh
#   SKIP_EXISTING=1 MAX_CHECKPOINTS=1 bash script/run_robotwin_eval.sh
#   ROBOTWIN_FAST=1 bash script/run_robotwin_eval.sh   # no video, no expert_check seed hunt
#   ROBOTWIN_EVAL_LOW_RENDER=1 bash script/run_robotwin_eval.sh  # rasterization (no RT 32spp)
#   ROBOTWIN_EVAL_TIMING=1 bash script/run_robotwin_eval.sh  # per-episode infer vs sim timing in client logs
#   CLIENTS_PER_GPU=3 bash script/run_robotwin_eval.sh # 12-way shards, 3 servers+clients/GPU
#   ROBOTWIN_SEED_CACHE=train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json bash script/run_robotwin_eval.sh
#   ROBOTWIN_DYNAMIC_SHARDS=1 bash script/run_robotwin_eval.sh # clients pull tasks from a shared queue
#   ROBOTWIN_TASKS=stack_bowls_three,click_bell bash script/run_robotwin_eval.sh
#
# Before running, stop training and release any site GPU holder through its
# controller/guard command. Never pkill a holder or guard daemon directly.
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NGPU=${NGPU:-${DETECTED_GPUS:-1}}
TEST_NUM=${TEST_NUM:-20}
EVAL_CONFIG_NAME=${EVAL_CONFIG_NAME:-robotwin}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
HARD_TASK_CONFIG=${HARD_TASK_CONFIG:-demo_randomized}
RUN_HARD=${RUN_HARD:-0}
SEED=${SEED:-0}
ST_SEED=$((10000 * (1 + SEED)))
START_PORT=${START_PORT:-29056}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29561}
SERVER_WARMUP_SEC=${SERVER_WARMUP_SEC:-8}
SKIP_EXISTING=${SKIP_EXISTING:-0}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
FULL_MODEL_PATH=${FULL_MODEL_PATH:-}
STRICT_CUROBO_COMPAT=${STRICT_CUROBO_COMPAT:-1}
ROBOTWIN_FAST=${ROBOTWIN_FAST:-0}
ROBOTWIN_EXPERT_CHECK=${ROBOTWIN_EXPERT_CHECK:-1}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-0}
ROBOTWIN_SAVE_COMPARISON_VIDEO=${ROBOTWIN_SAVE_COMPARISON_VIDEO:-0}
ROBOTWIN_SAVE_VISUALIZATION=${ROBOTWIN_SAVE_VISUALIZATION:-0}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-1}
ROBOTWIN_EVAL_LOW_RENDER=${ROBOTWIN_EVAL_LOW_RENDER:-0}
# Keep the official RoboTwin camera/render path by default. Recreating RT
# cameras during an episode changes observations and can materially alter SR.
ROBOTWIN_POLICY_CAMERAS_ONLY=${ROBOTWIN_POLICY_CAMERAS_ONLY:-0}
ROBOTWIN_DEFER_RENDER_UPDATES=${ROBOTWIN_DEFER_RENDER_UPDATES:-0}
ROBOTWIN_RECREATE_CAMERAS_EVERY=${ROBOTWIN_RECREATE_CAMERAS_EVERY:-0}
ROBOTWIN_RESUME_PARTIAL=${ROBOTWIN_RESUME_PARTIAL:-1}
ROBOTWIN_CLIENT_EPISODE_CHUNK=${ROBOTWIN_CLIENT_EPISODE_CHUNK:-1}
ROBOTWIN_CLIENT_TIMEOUT_SEC=${ROBOTWIN_CLIENT_TIMEOUT_SEC:-900}
ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES=${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES:-3}
CLIENT_STAGGER_SEC=${CLIENT_STAGGER_SEC:-0}
ROBOTWIN_DYNAMIC_SHARDS=${ROBOTWIN_DYNAMIC_SHARDS:-1}
ROBOTWIN_SIM_FOLLOWS_SERVER_GPU=${ROBOTWIN_SIM_FOLLOWS_SERVER_GPU:-0}
ROBOTWIN_SEED_CACHE=${ROBOTWIN_SEED_CACHE:-"${ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"}
DEFAULT_ROBOTWIN_TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
ROBOTWIN_TASKS=${ROBOTWIN_TASKS:-${DEFAULT_ROBOTWIN_TASKS}}

if [[ "${ROBOTWIN_FAST}" == "1" ]]; then
  ROBOTWIN_EXPERT_CHECK=0
  ROBOTWIN_EVAL_VIDEO_LOG=0
  ROBOTWIN_SAVE_COMPARISON_VIDEO=0
  ROBOTWIN_SAVE_VISUALIZATION=0
  ROBOTWIN_EVAL_LOW_RENDER=${ROBOTWIN_EVAL_LOW_RENDER:-1}
fi

export ROBOTWIN_EXPERT_CHECK ROBOTWIN_EVAL_VIDEO_LOG ROBOTWIN_SAVE_COMPARISON_VIDEO ROBOTWIN_SAVE_VISUALIZATION ROBOTWIN_EVAL_LOW_RENDER ROBOTWIN_POLICY_CAMERAS_ONLY ROBOTWIN_DEFER_RENDER_UPDATES ROBOTWIN_RECREATE_CAMERAS_EVERY ROBOTWIN_RESUME_PARTIAL ROBOTWIN_CLIENT_EPISODE_CHUNK ROBOTWIN_SEED_CACHE ROBOTWIN_SIM_FOLLOWS_SERVER_GPU

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${ROOT}/train_out/robotwin/checkpoints"}
BASE_MODEL=${BASE_MODEL:-"${ROOT}/checkpoints/lingbot-va-base"}
EVAL_MODEL_CACHE=${EVAL_MODEL_CACHE:-"${TMPDIR:-/tmp}/lingbot_robotwin_eval_symlinks"}
RESULTS_ROOT=${RESULTS_ROOT:-"${ROOT}/train_out/robotwin/eval_results"}
LOG_DIR=${LOG_DIR:-"${ROOT}/logs/robotwin_eval"}
VIS_ROOT=${VIS_ROOT:-"${ROOT}/train_out/robotwin/eval_visualization"}

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${EVAL_MODEL_CACHE}" "${VIS_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export EVAL_MODEL_CACHE
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.robotwin_eval_env" ]] && source "${ROOT}/script/.robotwin_eval_env"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"

NUM_GPUS_SHARD=${NGPU}
REQUESTED_SHARDS=$((NUM_GPUS_SHARD * CLIENTS_PER_GPU))

if [[ "${CLIENTS_PER_GPU}" -lt 1 ]]; then
  echo "CLIENTS_PER_GPU must be >= 1, got ${CLIENTS_PER_GPU}" >&2
  exit 1
fi

ALL_TASKS=(
  stack_bowls_three handover_block hanging_mug scan_object lift_pot
  put_object_cabinet stack_blocks_three place_shoe
  adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad
  pick_dual_bottles shake_bottle place_fan turn_switch
  shake_bottle_horizontally place_container_plate rotate_qrcode
  place_object_stand put_bottles_dustbin move_stapler_pad
  place_burger_fries place_bread_basket
  pick_diverse_bottles open_microwave beat_block_hammer press_stapler
  click_bell move_playingcard_away open_laptop move_can_pot
  stack_bowls_two place_a2b_right stamp_seal place_object_basket
  handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox
  click_alarmclock blocks_ranking_size place_phone_stand place_can_basket
  place_object_scale place_a2b_left grab_roller place_dual_shoes
  place_empty_cup blocks_ranking_rgb
)

if [[ -n "${ROBOTWIN_TASKS}" ]]; then
  SELECTED_TASKS=()
  IFS=',' read -r -a REQUESTED_TASKS <<< "${ROBOTWIN_TASKS}"
  for raw_task in "${REQUESTED_TASKS[@]}"; do
    task_name="${raw_task//[[:space:]]/}"
    [[ -z "${task_name}" ]] && continue
    task_found=0
    for known_task in "${ALL_TASKS[@]}"; do
      if [[ "${task_name}" == "${known_task}" ]]; then
        task_found=1
        break
      fi
    done
    if [[ "${task_found}" -ne 1 ]]; then
      echo "Unknown ROBOTWIN_TASKS entry: ${task_name}" >&2
      exit 1
    fi
    SELECTED_TASKS+=("${task_name}")
  done
  if [[ "${#SELECTED_TASKS[@]}" -eq 0 ]]; then
    echo "ROBOTWIN_TASKS was set but no valid tasks were parsed" >&2
    exit 1
  fi
  ALL_TASKS=("${SELECTED_TASKS[@]}")
fi

NUM_TASKS=${#ALL_TASKS[@]}
if [[ "${NUM_TASKS}" -lt "${REQUESTED_SHARDS}" ]]; then
  NUM_SHARDS=${NUM_TASKS}
else
  NUM_SHARDS=${REQUESTED_SHARDS}
fi

shard_tasks() {
  local shard_id=$1
  local -a out=()
  local i
  for ((i = shard_id; i < NUM_TASKS; i += NUM_SHARDS)); do
    out+=("${ALL_TASKS[$i]}")
  done
  printf '%s\n' "${out[@]}"
}

claim_next_task() {
  local queue_file=$1
  python - <<'PY' "${queue_file}"
import fcntl
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
lock_path.parent.mkdir(parents=True, exist_ok=True)
with open(lock_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    if not queue_path.exists():
        sys.exit(1)
    tasks = [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]
    if not tasks:
        sys.exit(1)
    task = tasks.pop(0)
    queue_path.write_text("".join(f"{item}\n" for item in tasks))
    print(task)
PY
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

# Route SAPIEN Vulkan sim away from GPU1/2 (3 inference servers + Vulkan on same device → DeviceLost).
# Server stays on server_gpu; client CUDA/Vulkan uses sim_gpu. Mapping matches batch 20260701_184555.
sim_gpu_for_server_gpu() {
  local server_gpu=$1
  local client_slot=$2
  if [[ "${ROBOTWIN_SIM_FOLLOWS_SERVER_GPU}" == "1" ]]; then
    echo "${server_gpu}"
    return
  fi
  if [[ "${NUM_GPUS_SHARD}" -lt 4 ]]; then
    echo "${server_gpu}"
    return
  fi
  if [[ "${server_gpu}" -eq 0 ]]; then
    echo 0
    return
  fi
  if [[ "${server_gpu}" -eq 3 ]]; then
    echo 3
    return
  fi
  local offset=0
  if [[ "${server_gpu}" -eq 2 ]]; then
    offset=1
  fi
  if (( (client_slot + offset) % 2 == 0 )); then
    echo 0
  else
    echo 3
  fi
}

sim_slots_for_gpu() {
  local sim_gpu=$1
  if [[ "${ROBOTWIN_SIM_FOLLOWS_SERVER_GPU}" == "1" ]]; then
    echo "${CLIENTS_PER_GPU}"
    return
  fi
  if [[ "${NUM_GPUS_SHARD}" -lt 4 ]]; then
    echo "${CLIENTS_PER_GPU}"
    return
  fi
  # GPU0/3 each host native shards + routed shards from GPU1/2 → up to 2*CLIENTS_PER_GPU sims.
  echo $((CLIENTS_PER_GPU * 2))
}

install_vulkan_icd() {
  if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
    local icd_file
    local old_ifs="${IFS}"
    IFS=':'
    for icd_file in ${VK_ICD_FILENAMES}; do
      if [[ ! -f "${icd_file}" ]]; then
        echo "VK_ICD_FILENAMES contains a missing file: ${icd_file}" >&2
        IFS="${old_ifs}"
        return 1
      fi
    done
    IFS="${old_ifs}"
    echo "Using project Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0
  fi

  local icd_src="${ROOT}/script/nvidia_icd.json"
  local icd_dst="/usr/share/vulkan/icd.d/nvidia_icd.json"
  if [[ ! -f "${icd_src}" ]]; then
    echo "Missing ${icd_src}" >&2
    return 1
  fi
  if [[ ! -f "${icd_dst}" ]] || ! cmp -s "${icd_src}" "${icd_dst}"; then
    mkdir -p /usr/share/vulkan/icd.d
    cp "${icd_src}" "${icd_dst}"
    echo "Installed NVIDIA Vulkan ICD at ${icd_dst}"
  fi
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

task_result_total() {
  local result_file=$1
  python - <<PY "${result_file}"
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(int(json.load(f).get("total_num", 0)))
except (OSError, ValueError, TypeError):
    print(0)
PY
}

preflight() {
  install_vulkan_icd
  if [[ ! -d "${ROBOTWIN_DIR}/envs" ]]; then
    echo "RoboTwin not installed at ${ROBOTWIN_DIR}. Run: bash script/setup_robotwin_eval.sh" >&2
    exit 1
  fi
  if [[ ! -f "${ROBOTWIN_DIR}/policy/ACT/deploy_policy.yml" ]]; then
    echo "Missing RoboTwin policy config under ${ROBOTWIN_DIR}/policy/ACT" >&2
    exit 1
  fi
  if [[ -n "${FULL_MODEL_PATH}" ]]; then
    local component
    for component in vae tokenizer text_encoder transformer; do
      if [[ ! -d "${FULL_MODEL_PATH}/${component}" ]]; then
        echo "Full model incomplete: missing ${FULL_MODEL_PATH}/${component}" >&2
        exit 1
      fi
    done
  elif [[ ! -d "${BASE_MODEL}/vae" || ! -d "${BASE_MODEL}/text_encoder" ]]; then
    echo "Base model incomplete: ${BASE_MODEL}" >&2
    exit 1
  fi
  local gpu_count
  gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "${gpu_count}" -lt "${NUM_GPUS_SHARD}" ]]; then
    echo "Need >= ${NUM_GPUS_SHARD} GPUs, found ${gpu_count}" >&2
    exit 1
  fi
  if [[ "${ROBOTWIN_EVAL_VIDEO_LOG}" == "1" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found (required when ROBOTWIN_EVAL_VIDEO_LOG=1). Run: apt-get install -y ffmpeg" >&2
    exit 1
  fi
  if ! (
    cd "${ROBOTWIN_DIR}"
    python - <<'PY'
from curobo.types.math import Pose  # noqa: F401
from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
print("curobo API compatible")
PY
  ); then
    echo "curobo incompatible; running repair script ..."
    bash "${ROOT}/script/fix_robotwin_curobo.sh"
    (
      cd "${ROBOTWIN_DIR}"
      python - <<'PY'
from curobo.types.math import Pose  # noqa: F401
from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
print("curobo API compatible after repair")
PY
    ) || {
      echo "curobo still incompatible after repair. Check ${ROOT}/script/fix_robotwin_curobo.sh" >&2
      exit 1
    }
  fi
  if [[ "${STRICT_CUROBO_COMPAT}" == "1" ]]; then
    (
      cd "${ROBOTWIN_DIR}"
      python - <<'PY'
import importlib
import sys

planner = importlib.import_module("envs.robot.planner")
if getattr(planner, "CUROBO_FALLBACK_ACTIVE", False):
    print(
        "Strict check failed: RoboTwin planner fell back to compatibility mode.",
        file=sys.stderr,
    )
    sys.exit(1)
print("Strict curobo compatibility check passed.")
PY
    ) || exit 1
  fi
}

checkpoint_done() {
  local result_dir=$1
  local task_name
  local found=0
  for task_name in "${ALL_TASKS[@]}"; do
    local res_json="${result_dir}/stseed-${ST_SEED}/metrics/${task_name}/res.json"
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
  local task_config=$5
  local shard_log=$6
  shift 6
  local -a shard_tasks_arr=("$@")

  (
    if [[ "${CLIENT_STAGGER_SEC}" -gt 0 ]]; then
      local client_slot
      client_slot="$(client_slot_for_shard "${shard_id}")"
      sleep $((client_slot * CLIENT_STAGGER_SEC))
    fi
    local client_slot sim_gpu_id sim_slots
    client_slot="$(client_slot_for_shard "${shard_id}")"
    sim_gpu_id="$(sim_gpu_for_server_gpu "${gpu_id}" "${client_slot}")"
    sim_slots="$(sim_slots_for_gpu "${sim_gpu_id}")"
    # Inference server on server GPU ${gpu_id}; SAPIEN sim on sim GPU ${sim_gpu_id}.
    export CUDA_VISIBLE_DEVICES="${sim_gpu_id}"
    export ROBOTWIN_VULKAN_GPU="${sim_gpu_id}"
    export NVIDIA_VISIBLE_DEVICES="${sim_gpu_id}"
    export ROBOTWIN_VULKAN_SIM_SLOTS="${sim_slots}"
    export ROBOTWIN_DIR
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    cd "${ROBOTWIN_DIR}"
    run_client_task() {
      local task_name=$1
      local res_json="${ckpt_result_dir}/stseed-${ST_SEED}/metrics/${task_name}/res.json"
      local no_progress_retries=0
      if [[ "${SKIP_EXISTING}" == "1" ]]; then
        if [[ -f "${res_json}" ]] && task_result_complete "${res_json}"; then
          echo "shard${shard_id}/serverGPU${gpu_id}/simGPU${sim_gpu_id} skip task=${task_name} (complete res.json)"
          return 0
        fi
      fi
      while true; do
        local before after client_rc
        before="$(task_result_total "${res_json}")"
        echo "shard${shard_id}/serverGPU${gpu_id}/simGPU${sim_gpu_id} task=${task_name} port=${port} resume=${before}/${TEST_NUM}"
        set +e
        PYTHONUNBUFFERED=1 \
        PYTHONWARNINGS=ignore::UserWarning \
        timeout --signal=TERM --kill-after=30s "${ROBOTWIN_CLIENT_TIMEOUT_SEC}" \
          python -m evaluation.robotwin.eval_polict_client_openpi \
          --config policy/ACT/deploy_policy.yml \
          --overrides \
          --task_name "${task_name}" \
          --task_config "${task_config}" \
          --train_config_name 0 \
          --model_name 0 \
          --ckpt_setting 0 \
          --seed "${SEED}" \
          --policy_name ACT \
          --save_root "${ckpt_result_dir}" \
          --video_guidance_scale 5 \
          --action_guidance_scale 1 \
          --test_num "${TEST_NUM}" \
          --port "${port}"
        client_rc=$?
        set -e

        if [[ -f "${res_json}" ]] && task_result_complete "${res_json}"; then
          return 0
        fi
        after="$(task_result_total "${res_json}")"
        if [[ "${after}" -gt "${before}" ]]; then
          no_progress_retries=0
          echo "task=${task_name} chunk complete: ${before} -> ${after}; restarting client"
          continue
        fi

        no_progress_retries=$((no_progress_retries + 1))
        echo "task=${task_name} client rc=${client_rc}, no progress ${before} -> ${after}; retry ${no_progress_retries}/${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES}" >&2
        if [[ "${ROBOTWIN_CLIENT_EPISODE_CHUNK}" -le 0 || "${no_progress_retries}" -ge "${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES}" ]]; then
          return 1
        fi
        sleep 5
      done
    }

    local task_name
    if [[ "${ROBOTWIN_DYNAMIC_SHARDS}" == "1" ]]; then
      while task_name="$(claim_next_task "${TASK_QUEUE_FILE}")"; do
        run_client_task "${task_name}"
      done
    else
      for task_name in "${shard_tasks_arr[@]}"; do
        run_client_task "${task_name}"
      done
    fi
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

eval_one_task_config() {
  local active_task_config=$1

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
  echo "Task config : ${active_task_config}"
  echo "TEST_NUM    : ${TEST_NUM} per task"
  echo "ROBOTWIN_FAST: ${ROBOTWIN_FAST} (expert_check=${ROBOTWIN_EXPERT_CHECK}, video=${ROBOTWIN_EVAL_VIDEO_LOG}, low_render=${ROBOTWIN_EVAL_LOW_RENDER}, policy_cameras_only=${ROBOTWIN_POLICY_CAMERAS_ONLY}, defer_render_updates=${ROBOTWIN_DEFER_RENDER_UPDATES}, recreate_cameras_every=${ROBOTWIN_RECREATE_CAMERAS_EVERY}, resume_partial=${ROBOTWIN_RESUME_PARTIAL}, client_episode_chunk=${ROBOTWIN_CLIENT_EPISODE_CHUNK}, client_timeout_sec=${ROBOTWIN_CLIENT_TIMEOUT_SEC}, no_progress_retries=${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES})"
  echo "Seed cache  : ${ROBOTWIN_SEED_CACHE:-<none>}"
  echo "Dynamic shards: ${ROBOTWIN_DYNAMIC_SHARDS}"
  echo "Tasks       : ${NUM_TASKS}${ROBOTWIN_TASKS:+ (${ROBOTWIN_TASKS})}"
  echo "Client stagger: ${CLIENT_STAGGER_SEC}s per slot (0=off)"
  echo "Config      : ${EVAL_CONFIG_NAME}"
  if [[ "${NUM_GPUS_SHARD}" -lt 4 ]]; then
    echo "GPUs        : 0-$((NUM_GPUS_SHARD - 1)) (${NUM_SHARDS} shards, ${CLIENTS_PER_GPU} server(s)/GPU; sim follows server GPU, sim_slots=${CLIENTS_PER_GPU})"
  else
    echo "GPUs        : 0-$((NUM_GPUS_SHARD - 1)) (${NUM_SHARDS} shards, ${CLIENTS_PER_GPU} server(s)/GPU; sim GPU1/2 -> 0/3, sim_slots=$((CLIENTS_PER_GPU * 2)))"
  fi
  echo "RoboTwin    : ${ROBOTWIN_DIR}"

  local BATCH_TIME
  BATCH_TIME=$(date +%Y%m%d_%H%M%S)
  local TASK_RESULTS_ROOT="${RESULTS_ROOT}/${active_task_config}"

  for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="$(basename "${CKPT_PATH}")"
    CKPT_RESULT_DIR="${TASK_RESULTS_ROOT}/${CKPT_NAME}"
    CKPT_VIS_DIR="${VIS_ROOT}/${CKPT_NAME}/${active_task_config}"
    mkdir -p "${CKPT_RESULT_DIR}" "${CKPT_VIS_DIR}"

    if [[ "${SKIP_EXISTING}" == "1" ]] && checkpoint_done "${CKPT_RESULT_DIR}"; then
      echo "========== Skip ${CKPT_NAME} (${active_task_config}, results complete) =========="
      continue
    fi

    echo "========== Evaluating ${CKPT_NAME} (${active_task_config}) =========="

    SERVER_PIDS=""
    SERVER_PORTS=()
    SERVER_WAIT_PIDS=()
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
      gpu_id="$(gpu_for_shard "${shard_id}")"
      port="$(port_for_shard "${shard_id}")"
      master_port="$(master_port_for_shard "${shard_id}")"
      client_slot="$(client_slot_for_shard "${shard_id}")"
      log_file="${LOG_DIR}/server_${CKPT_NAME}_${active_task_config}_${BATCH_TIME}_shard${shard_id}_gpu${gpu_id}_c${client_slot}.log"

      local -a server_model_args
      if [[ -n "${FULL_MODEL_PATH}" ]]; then
        server_model_args=(--model-path "${FULL_MODEL_PATH}")
      else
        server_model_args=(--checkpoint "${CKPT_PATH}" --base-model "${BASE_MODEL}")
      fi

      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      nohup python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port="${master_port}" \
        evaluation/libero/run_server_ckpt.py \
        "${server_model_args[@]}" \
        --config-name "${EVAL_CONFIG_NAME}" \
        --port "${port}" \
        --save-root "${CKPT_VIS_DIR}" \
        > "${log_file}" 2>&1 &

      pid=$!
      SERVER_PIDS="${SERVER_PIDS} ${pid}"
      SERVER_PORTS+=("${port}")
      SERVER_WAIT_PIDS+=("${pid}")
      echo "Server shard${shard_id}/GPU${gpu_id} port=${port} master=${master_port} pid=${pid} log=${log_file}"
    done

    for idx in "${!SERVER_PORTS[@]}"; do
      wait_for_port "${SERVER_PORTS[$idx]}" 600 "${SERVER_WAIT_PIDS[$idx]}"
    done

    echo "All ${NUM_SHARDS} servers ready."
    sleep "${SERVER_WARMUP_SEC}"

    TASK_QUEUE_FILE=""
    if [[ "${ROBOTWIN_DYNAMIC_SHARDS}" == "1" ]]; then
      TASK_QUEUE_FILE="${LOG_DIR}/task_queue_${CKPT_NAME}_${active_task_config}_${BATCH_TIME}.txt"
      printf '%s\n' "${ALL_TASKS[@]}" > "${TASK_QUEUE_FILE}"
      echo "Dynamic task queue: ${TASK_QUEUE_FILE}"
    fi

    CLIENT_PIDS=""
    CLIENT_FAILED=0
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
      gpu_id="$(gpu_for_shard "${shard_id}")"
      port="$(port_for_shard "${shard_id}")"
      client_slot="$(client_slot_for_shard "${shard_id}")"
      mapfile -t shard_task_list < <(shard_tasks "${shard_id}")
      shard_log="${LOG_DIR}/client_${CKPT_NAME}_${active_task_config}_shard${shard_id}_gpu${gpu_id}_${BATCH_TIME}.log"
      run_shard_client "${shard_id}" "${gpu_id}" "${port}" "${CKPT_RESULT_DIR}" "${active_task_config}" \
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
      echo "ERROR: some clients failed for ${CKPT_NAME}, check ${LOG_DIR}" >&2
      stop_servers
      return 1
    fi
    if ! checkpoint_done "${CKPT_RESULT_DIR}"; then
      echo "ERROR: incomplete task results for ${CKPT_NAME} (${active_task_config})" >&2
      stop_servers
      return 1
    fi

    echo "Finished ${CKPT_NAME} (${active_task_config})"
    stop_servers
    sleep 3
  done

  python evaluation/robotwin/collect_results.py \
    --results-root "${RESULTS_ROOT}" \
    --task-config "${active_task_config}" \
    --st-seed "${ST_SEED}" \
    --out-csv "${TASK_RESULTS_ROOT}/results.csv" \
    --out-md "${TASK_RESULTS_ROOT}/results.md"
}

trap stop_servers EXIT INT TERM

preflight

eval_one_task_config "${TASK_CONFIG}"
if [[ "${RUN_HARD}" == "1" && "${TASK_CONFIG}" != "${HARD_TASK_CONFIG}" ]]; then
  eval_one_task_config "${HARD_TASK_CONFIG}"
fi

echo "All checkpoints evaluated."
echo "Easy results : ${RESULTS_ROOT}/${TASK_CONFIG}/results.md"
if [[ "${RUN_HARD}" == "1" ]]; then
  echo "Hard results : ${RESULTS_ROOT}/${HARD_TASK_CONFIG}/results.md"
fi
