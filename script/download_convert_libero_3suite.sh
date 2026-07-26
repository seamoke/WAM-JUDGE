#!/usr/bin/env bash
# Download official LIBERO HDF5 datasets and convert to LingBot-VA LeRobot v2.1 format.
#
# Source: https://github.com/Lifelong-Robot-Learning/LIBERO
# Dataset mirror: yifengzhu-hf/LIBERO-datasets (used by LIBERO download_utils)
#
# Usage:
#   bash script/download_convert_libero_3suite.sh
#   bash script/download_convert_libero_3suite.sh download   # download only
#   bash script/download_convert_libero_3suite.sh convert    # convert only
#   bash script/download_convert_libero_3suite.sh mix        # assemble libero-mix
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}/third_party/LIBERO:${PYTHONPATH:-}"

RAW_DIR="${RAW_DIR:-${ROOT}/data/libero-raw}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/libero-3suite-lerobot}"
LONG_DIR="${LONG_DIR:-${ROOT}/data/libero-long-lerobot}"
MIX_DIR="${MIX_DIR:-${ROOT}/data/libero-mix}"
TARGET="${1:-all}"

LIBERO_SUITES=(libero_spatial libero_object libero_goal)

# shellcheck disable=SC1091
source .venv/bin/activate

download_suites() {
  echo "==> Download official LIBERO HDF5 suites: ${LIBERO_SUITES[*]}"
  echo "    Source: yifengzhu-hf/LIBERO-datasets"
  echo "    Output: ${RAW_DIR}"
  mkdir -p "${RAW_DIR}"
  export RAW_DIR

  RAW_DIR="${RAW_DIR}" python - <<'PY'
import glob
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

raw = os.environ["RAW_DIR"]
suites = ["libero_spatial", "libero_object", "libero_goal"]
patterns = [f"{suite}/*.hdf5" for suite in suites]
snapshot_download(
    "yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir=raw,
    allow_patterns=patterns,
    max_workers=4,
)
for suite in suites:
    n = len(glob.glob(os.path.join(raw, suite, "*.hdf5")))
    print(f"  {suite}: {n} hdf5 files")
PY
}

convert_suites() {
  echo "==> Convert HDF5 -> LingBot-VA LeRobot v2.1"
  echo "    Input:  ${RAW_DIR}"
  echo "    Output: ${OUT_DIR}"
  pip install -q h5py jsonlines
  python script/convert_libero_hdf5_to_lerobot.py \
    --hdf5-root "${RAW_DIR}" \
    --output-dir "${OUT_DIR}" \
    --suites "${LIBERO_SUITES[@]}"
}

assemble_mix() {
  echo "==> Assemble ${MIX_DIR}"
  mkdir -p "${MIX_DIR}"

  local src_10
  src_10="$(find "${LONG_DIR}" -path '*/libero_10/*/meta/info.json' 2>/dev/null | head -1 || true)"
  if [[ -n "${src_10}" ]]; then
    local suite_root
    suite_root="$(dirname "$(dirname "${src_10}")")"
    ln -sfn "${suite_root}" "${MIX_DIR}/libero_10"
    echo "  linked libero_10 -> ${suite_root}"
  else
    echo "  [warn] libero_10 not found under ${LONG_DIR}"
  fi

  for suite in "${LIBERO_SUITES[@]}"; do
    local info
    info="$(find "${OUT_DIR}/${suite}" -name info.json 2>/dev/null | head -1 || true)"
    if [[ -n "${info}" ]]; then
      local suite_root
      suite_root="$(dirname "$(dirname "${info}")")"
      ln -sfn "${suite_root}" "${MIX_DIR}/${suite}"
      echo "  linked ${suite} -> ${suite_root}"
    else
      echo "  [warn] ${suite} not converted yet"
    fi
  done

  if [[ -f "${LONG_DIR}/empty_emb.pt" && ! -e "${MIX_DIR}/empty_emb.pt" ]]; then
    ln -sfn "${LONG_DIR}/empty_emb.pt" "${MIX_DIR}/empty_emb.pt"
    echo "  linked empty_emb.pt"
  fi

  echo ""
  echo "libero-mix layout:"
  find "${MIX_DIR}" -name info.json 2>/dev/null | while read -r f; do
    echo "  ${f}"
  done || true
  echo ""
  echo "Next: extract WAN latents for spatial/object/goal, then train with:"
  echo "  DATASET_PATH=${MIX_DIR} bash script/run_libero_train.sh"
}

case "${TARGET}" in
  download) download_suites ;;
  convert)  convert_suites ;;
  mix)      assemble_mix ;;
  all)      download_suites; convert_suites; assemble_mix ;;
  *)
    echo "Unknown target: ${TARGET}"
    echo "Usage: $0 [all|download|convert|mix]"
    exit 1
    ;;
esac

echo "Done."
