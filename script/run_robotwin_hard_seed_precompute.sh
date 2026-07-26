#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lingbot-va
cd "${ROOT}"

TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
TASKS_0="adjust_bottle,click_alarmclock,dump_bin_bigbin,handover_mic,move_can_pot,move_stapler_pad,pick_dual_bottles,place_a2b_right,place_container_plate,place_empty_cup,place_burger_fries,place_object_scale,place_phone_stand,place_shoe,rotate_qrcode,stamp_seal"
TASKS_1="beat_block_hammer,click_bell,grab_roller,lift_pot,move_playingcard_away,pick_diverse_bottles,place_a2b_left,place_bread_skillet,place_dual_shoes,place_fan,place_mouse_pad,place_object_stand,move_pillbottle_pad,press_stapler,scan_object,turn_switch"
TASK_CONFIG=demo_randomized
PER_TASK=${PER_TASK:-100}
SEED=${SEED:-0}
RUN_ID=${RUN_ID:-robotwin_hard_seed_precompute_2x5090_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="${ROOT}/logs/robotwin_seed_precompute/${RUN_ID}"
CACHE_DIR="${ROOT}/train_out/robotwin/eval_seed_cache"
SHARD_0="${CACHE_DIR}/${TASK_CONFIG}_seed${SEED}_n${PER_TASK}.shard0.json"
SHARD_1="${CACHE_DIR}/${TASK_CONFIG}_seed${SEED}_n${PER_TASK}.shard1.json"
OUTPUT="${CACHE_DIR}/${TASK_CONFIG}_seed${SEED}_n${PER_TASK}.json"
SOURCE="${ROOT}/script/precompute_robotwin_eval_seeds.py"
SOURCE_SHA256="$(sha256sum "${SOURCE}" | awk '{print $1}')"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

if pgrep -af "[g]pu_vram_hold.py" >/dev/null; then
  echo "Refusing to start while gpu_vram_hold.py is running" >&2
  exit 1
fi
if pgrep -af "precompute_robotwin_eval_seeds.py|run_robotwin_eval.sh|run_server_ckpt.py|eval_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin workload is running" >&2
  exit 1
fi

{
  echo "RUN_ID=${RUN_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "task_config=${TASK_CONFIG}"
  echo "per_task=${PER_TASK}"
  echo "seed=${SEED}"
  echo "tasks=${TASKS}"
  echo "source=${SOURCE}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "output=${OUTPUT}"
} > "${LOG_DIR}/meta.txt"
echo "running" > "${LOG_DIR}/status.txt"

run_shard() {
  local gpu=$1
  local tasks=$2
  local output=$3
  local log=$4
  (
    cd "${ROOT}/third_party/RoboTwin"
    env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${ROOT}" \
      ROBOTWIN_VULKAN_GPU="${gpu}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      NVIDIA_VISIBLE_DEVICES="${gpu}" \
      "${ROOT}/.venv/bin/python" "${SOURCE}" \
        --task-config "${TASK_CONFIG}" \
        --seed "${SEED}" \
        --per-task "${PER_TASK}" \
        --tasks "${tasks}" \
        --output "${output}" \
        --no-expert-check
  ) > "${log}" 2>&1
}

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM

run_shard 0 "${TASKS_0}" "${SHARD_0}" "${LOG_DIR}/shard0.log" &
PID_0=$!
run_shard 1 "${TASKS_1}" "${SHARD_1}" "${LOG_DIR}/shard1.log" &
PID_1=$!
echo "${PID_0}" > "${LOG_DIR}/shard0.pid"
echo "${PID_1}" > "${LOG_DIR}/shard1.pid"

rc=0
wait "${PID_0}" || rc=1
wait "${PID_1}" || rc=1
if [[ "${rc}" -ne 0 ]]; then
  echo "failed: one or more seed shards exited nonzero" > "${LOG_DIR}/status.txt"
  exit 1
fi

"${ROOT}/.venv/bin/python" "${ROOT}/script/merge_robotwin_seed_caches.py" \
  --inputs "${SHARD_0}" "${SHARD_1}" \
  --output "${OUTPUT}" \
  --task-config "${TASK_CONFIG}" \
  --per-task "${PER_TASK}" \
  --tasks "${TASKS}" \
  --source-sha256 "${SOURCE_SHA256}" \
  | tee "${LOG_DIR}/merge.log"

sha256sum "${OUTPUT}" | tee "${OUTPUT}.sha256" "${LOG_DIR}/SHA256SUMS"
echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${LOG_DIR}/meta.txt"
echo "done" > "${LOG_DIR}/status.txt"
echo "Seed cache ready: ${OUTPUT}"
