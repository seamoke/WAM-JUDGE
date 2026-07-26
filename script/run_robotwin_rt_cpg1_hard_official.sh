#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lingbot-va
cd "${ROOT}"

RUN_KIND=${RUN_KIND:-formal}
TEST_NUM=${TEST_NUM:-20}
ALL_TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
TASKS=${TASKS:-"${ALL_TASKS}"}
NGPU=${NGPU:-2}
RUN_ID=${RUN_ID:-robotwin_rt_cpg1_hard_official_step600_n${TEST_NUM}_2x5090_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="${ROOT}/logs/robotwin_eval_rt_cpg1_hard_official_step600_n${TEST_NUM}/${RUN_ID}"
RESULTS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_results_rt_cpg1_hard_official_step600_n${TEST_NUM}/${RUN_ID}"
VIS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_visualization_rt_cpg1_hard_official_step600_n${TEST_NUM}/${RUN_ID}"
SEED_CACHE="${ROOT}/train_out/robotwin/eval_seed_cache/demo_randomized_seed0_n100.json"
CHECKPOINT_PATH="${ROOT}/train_out/robotwin-short-3gpu/checkpoints/checkpoint_step_18000"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${VIS_ROOT}"

if pgrep -af "[g]pu_vram_hold.py" >/dev/null; then
  echo "Refusing to start while gpu_vram_hold.py is running" >&2
  exit 1
fi
if pgrep -af "run_robotwin_eval.sh|run_server_ckpt.py|eval_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin evaluation is running" >&2
  exit 1
fi
test -d "${CHECKPOINT_PATH}"
test -s "${SEED_CACHE}"

"${ROOT}/.venv/bin/python" "${ROOT}/script/merge_robotwin_seed_caches.py" \
  --inputs "${SEED_CACHE}" \
  --output /scratch/robotwin_seed_cache_validation.json \
  --task-config demo_randomized \
  --per-task 100 \
  --tasks "${ALL_TASKS}" \
  --source-sha256 "$(sha256sum "${ROOT}/script/precompute_robotwin_eval_seeds.py" | awk '{print $1}')" \
  > "${LOG_DIR}/seed_cache_validation.log"

{
  echo "RUN_ID=${RUN_ID}"
  echo "run_kind=${RUN_KIND}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "seed_cache=${SEED_CACHE}"
  echo "seed_cache_sha256=$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
  echo "tasks=${TASKS}"
  echo "results_root=${RESULTS_ROOT}"
  echo "vis_root=${VIS_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "task_config=demo_randomized"
  echo "protocol=rt_cpg1_official"
  echo "concurrency=NGPU=${NGPU} CLIENTS_PER_GPU=1"
  echo "timing=ROBOTWIN_EVAL_TIMING=1"
  echo "test_num=${TEST_NUM}"
  echo "camera_path=official policy_cameras_only=0 defer_render_updates=0 recreate_cameras_every=0"
  echo "model_offload=enable=0 vae=0 text_encoder=1"
  echo "restore_gpu_hold_on_exit=0"
  echo "resilience=client_episode_chunk=1 client_timeout_sec=900 no_progress_retries=3"
  echo "max_checkpoints=1"
} > "${LOG_DIR}/meta.txt"
echo "running" > "${LOG_DIR}/status.txt"

set +e
NGPU="${NGPU}" \
CLIENTS_PER_GPU=1 \
ROBOTWIN_DYNAMIC_SHARDS=1 \
ROBOTWIN_TASKS="${TASKS}" \
TASK_CONFIG=demo_randomized \
RUN_HARD=0 \
TEST_NUM="${TEST_NUM}" \
MAX_CHECKPOINTS=1 \
ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
ROBOTWIN_EXPERT_CHECK=0 \
ROBOTWIN_FAST=0 \
ROBOTWIN_EVAL_LOW_RENDER=0 \
ROBOTWIN_POLICY_CAMERAS_ONLY=0 \
ROBOTWIN_DEFER_RENDER_UPDATES=0 \
ROBOTWIN_RECREATE_CAMERAS_EVERY=0 \
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
else
  echo "failed rc=${rc}" > "${LOG_DIR}/status.txt"
fi
exit "${rc}"
