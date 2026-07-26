#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lingbot-va
cd "${ROOT}"

RUN_ID=${RUN_ID:-robotwin_protocol_compare_step600_n20_$(date +%Y%m%d_%H%M%S)}
BASE_LOG_DIR="${ROOT}/logs/robotwin_protocol_compare/${RUN_ID}"
BASE_RESULTS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_protocol_compare_step600_n20/${RUN_ID}"
BASE_VIS_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_protocol_compare_visualization_step600_n20/${RUN_ID}"
SEED_CACHE="${ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"
CHECKPOINT_PATH="${ROOT}/train_out/robotwin-short-3gpu/checkpoints/checkpoint_step_18000"
TASKS="place_dual_shoes,lift_pot,place_shoe,click_bell,grab_roller,dump_bin_bigbin"

mkdir -p "${BASE_LOG_DIR}" "${BASE_RESULTS_ROOT}" "${BASE_VIS_ROOT}"

{
  echo "RUN_ID=${RUN_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "seed_cache=${SEED_CACHE}"
  echo "tasks=${TASKS}"
  echo "test_num=20"
  echo "protocols=fast_lowrender_cpg1 rt_cpg1 fast_lowrender_cpg3"
} > "${BASE_LOG_DIR}/meta.txt"

run_protocol() {
  local name=$1
  local clients_per_gpu=$2
  local low_render=$3
  local fast=$4

  local log_dir="${BASE_LOG_DIR}/${name}"
  local results_root="${BASE_RESULTS_ROOT}/${name}"
  local vis_root="${BASE_VIS_ROOT}/${name}"
  mkdir -p "${log_dir}" "${results_root}" "${vis_root}"

  {
    echo "protocol=${name}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "NGPU=${NGPU:-1}"
    echo "CLIENTS_PER_GPU=${clients_per_gpu}"
    echo "ROBOTWIN_EVAL_LOW_RENDER=${low_render}"
    echo "ROBOTWIN_FAST=${fast}"
  } > "${log_dir}/meta.txt"

  echo "running" > "${log_dir}/status.txt"
  set +e
  NGPU="${NGPU:-1}" \
  CLIENTS_PER_GPU="${clients_per_gpu}" \
  CLIENT_STAGGER_SEC=1 \
  ROBOTWIN_DYNAMIC_SHARDS=1 \
  ROBOTWIN_TASKS="${TASKS}" \
  TEST_NUM=20 \
  MAX_CHECKPOINTS=1 \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
  ROBOTWIN_FAST="${fast}" \
  ROBOTWIN_EVAL_LOW_RENDER="${low_render}" \
  ROBOTWIN_EVAL_TIMING=1 \
  SKIP_EXISTING=0 \
  RESULTS_ROOT="${results_root}" \
  LOG_DIR="${log_dir}" \
  VIS_ROOT="${vis_root}" \
  bash script/run_robotwin_eval.sh > "${log_dir}/launcher.log" 2>&1
  local rc=$?
  set -e

  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${log_dir}/meta.txt"
  echo "exit_code=${rc}" >> "${log_dir}/meta.txt"
  if [[ "${rc}" -eq 0 ]]; then
    echo "done" > "${log_dir}/status.txt"
  else
    echo "failed rc=${rc}" > "${log_dir}/status.txt"
  fi
  return "${rc}"
}

overall_rc=0
run_protocol fast_lowrender_cpg1 1 1 1 || overall_rc=1
run_protocol rt_cpg1 1 0 0 || overall_rc=1
run_protocol fast_lowrender_cpg3 3 1 1 || overall_rc=1

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${BASE_LOG_DIR}/meta.txt"
echo "exit_code=${overall_rc}" >> "${BASE_LOG_DIR}/meta.txt"
if [[ "${overall_rc}" -eq 0 ]]; then
  echo "done" > "${BASE_LOG_DIR}/status.txt"
else
  echo "failed rc=${overall_rc}" > "${BASE_LOG_DIR}/status.txt"
fi
exit "${overall_rc}"
