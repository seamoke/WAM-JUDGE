#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lingbot-va
cd "${ROOT}"

RUN_ID=${RUN_ID:-robotwin_rt_cpg1_official_step600_n20_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="${ROOT}/logs/robotwin_eval_rt_cpg1_official_step600_n20/${RUN_ID}"
RESULTS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_results_rt_cpg1_official_step600_n20/${RUN_ID}"
VIS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_visualization_rt_cpg1_official_step600_n20/${RUN_ID}"
SEED_CACHE="${ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"
CHECKPOINT_PATH="${ROOT}/train_out/robotwin-short-3gpu/checkpoints/checkpoint_step_18000"
DEFAULT_TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
TASKS=${TASKS_OVERRIDE:-${DEFAULT_TASKS}}
TEST_NUM=${TEST_NUM:-20}
NGPU=${NGPU:-2}
WAN_VA_ENABLE_OFFLOAD=${WAN_VA_ENABLE_OFFLOAD:-0}
WAN_VA_OFFLOAD_VAE=${WAN_VA_OFFLOAD_VAE:-0}
WAN_VA_OFFLOAD_TEXT_ENCODER=${WAN_VA_OFFLOAD_TEXT_ENCODER:-1}
RESTORE_GPU_HOLD_ON_EXIT=${RESTORE_GPU_HOLD_ON_EXIT:-0}

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${VIS_ROOT}"

{
  echo "RUN_ID=${RUN_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "seed_cache=${SEED_CACHE}"
  echo "tasks=${TASKS}"
  echo "results_root=${RESULTS_ROOT}"
  echo "vis_root=${VIS_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "protocol=rt_cpg1_official"
  echo "concurrency=NGPU=${NGPU} CLIENTS_PER_GPU=1"
  echo "timing=ROBOTWIN_EVAL_TIMING=1"
  echo "test_num=${TEST_NUM}"
  echo "camera_path=official policy_cameras_only=0 defer_render_updates=0 recreate_cameras_every=0"
  echo "model_offload=enable=${WAN_VA_ENABLE_OFFLOAD} vae=${WAN_VA_OFFLOAD_VAE} text_encoder=${WAN_VA_OFFLOAD_TEXT_ENCODER}"
  echo "restore_gpu_hold_on_exit=${RESTORE_GPU_HOLD_ON_EXIT}"
  echo "resilience=client_episode_chunk=1 client_timeout_sec=${ROBOTWIN_CLIENT_TIMEOUT_SEC:-900} no_progress_retries=${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES:-3}"
  echo "max_checkpoints=1"
} > "${LOG_DIR}/meta.txt"

echo "running" > "${LOG_DIR}/status.txt"

set +e
WAN_VA_ENABLE_OFFLOAD="${WAN_VA_ENABLE_OFFLOAD}" \
WAN_VA_OFFLOAD_VAE="${WAN_VA_OFFLOAD_VAE}" \
WAN_VA_OFFLOAD_TEXT_ENCODER="${WAN_VA_OFFLOAD_TEXT_ENCODER}" \
NGPU="${NGPU}" \
CLIENTS_PER_GPU=1 \
ROBOTWIN_DYNAMIC_SHARDS=1 \
ROBOTWIN_TASKS="${TASKS}" \
TEST_NUM="${TEST_NUM}" \
MAX_CHECKPOINTS=1 \
ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
ROBOTWIN_FAST=0 \
ROBOTWIN_EVAL_LOW_RENDER=0 \
ROBOTWIN_POLICY_CAMERAS_ONLY=0 \
ROBOTWIN_DEFER_RENDER_UPDATES=0 \
ROBOTWIN_RECREATE_CAMERAS_EVERY=0 \
ROBOTWIN_RESUME_PARTIAL=1 \
ROBOTWIN_CLIENT_EPISODE_CHUNK=1 \
ROBOTWIN_CLIENT_TIMEOUT_SEC="${ROBOTWIN_CLIENT_TIMEOUT_SEC:-900}" \
ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES="${ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES:-3}" \
ROBOTWIN_EVAL_TIMING=1 \
SKIP_EXISTING=1 \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
LOG_DIR="${LOG_DIR}" \
VIS_ROOT="${VIS_ROOT}" \
bash script/run_robotwin_eval.sh
rc=$?
set -e

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${LOG_DIR}/meta.txt"
echo "exit_code=${rc}" >> "${LOG_DIR}/meta.txt"
if [[ "${rc}" -eq 0 ]]; then
  echo "done" > "${LOG_DIR}/status.txt"
  if [[ "${RESTORE_GPU_HOLD_ON_EXIT}" == "1" ]] && \
      ! pgrep -af "script/gpu_vram_hold.py" | grep -v pgrep >/dev/null 2>&1; then
    nohup .venv/bin/python -u script/gpu_vram_hold.py \
      --gib-per-gpu "${HOLD_GIB_PER_GPU:-16}" \
      --interval "${HOLD_INTERVAL_SEC:-60}" \
      > "${LOG_DIR}/gpu_vram_hold.log" 2>&1 < /dev/null &
    echo $! > "${LOG_DIR}/gpu_vram_hold.pid"
  fi
else
  echo "failed rc=${rc}" > "${LOG_DIR}/status.txt"
fi
exit "${rc}"
