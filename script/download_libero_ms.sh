#!/usr/bin/env bash
# Download LIBERO datasets for LingBot-VA post-training.
#
# ModelScope:
#   Robbyant/libero-long-lerobot  -> libero_10 with WAN latents (train-ready)
#
# hf-mirror (nvidia/LIBERO_LeRobot_v3, per-suite):
#   libero_spatial / libero_object / libero_goal  -> raw LeRobot v3.0 (no latents)
#   (libero_90 and libero_10 are NOT downloaded by default)
#
# Usage:
#   bash script/download_libero_ms.sh              # long + 3 suites + mix
#   bash script/download_libero_ms.sh long         # libero_10 with latents only
#   bash script/download_libero_ms.sh suites       # spatial/object/goal only
#   bash script/download_libero_ms.sh mix          # assemble libero-mix dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LONG_DIR="${LONG_DIR:-${ROOT}/data/libero-long-lerobot}"
SUITES_DIR="${SUITES_DIR:-${ROOT}/data/libero-3suite-lerobot}"
MIX_DIR="${MIX_DIR:-${ROOT}/data/libero-mix}"
TARGET="${1:-all}"

# shellcheck disable=SC1091
source .venv/bin/activate

LIBERO_SUITES=(libero_spatial libero_object libero_goal)

download_long() {
  echo "==> [long] Download Robbyant/libero-long-lerobot -> ${LONG_DIR}"
  bash script/download_cn.sh libero-long-lerobot "${LONG_DIR}"

  local tgz="${LONG_DIR}/libero_10.tgz"
  if [[ -f "${tgz}" ]]; then
    local marker="${LONG_DIR}/.libero_10_extracted"
    if [[ ! -f "${marker}" ]]; then
      echo "==> [long] Extract ${tgz}"
      tar -xzf "${tgz}" -C "${LONG_DIR}"
      touch "${marker}"
    else
      echo "==> [long] libero_10.tgz already extracted, skip"
    fi
  fi
}

download_suites() {
  echo "==> [suites] Download spatial/object/goal (skip libero_90, libero_10)"
  echo "    Source: nvidia/LIBERO_LeRobot_v3 via hf-mirror"
  echo "    Output: ${SUITES_DIR}"
  mkdir -p "${SUITES_DIR}"

  python - <<PY
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

out = "${SUITES_DIR}"
patterns = ["libero_spatial/**", "libero_object/**", "libero_goal/**"]
ignore_patterns = ["**/*.swp", "**/.DS_Store"]
snapshot_download(
    "nvidia/LIBERO_LeRobot_v3",
    repo_type="dataset",
    local_dir=out,
    allow_patterns=patterns,
    ignore_patterns=ignore_patterns,
    max_workers=4,
    etag_timeout=60,
)
print("OK:", out)
for s in ["libero_spatial", "libero_object", "libero_goal"]:
    p = os.path.join(out, s, "meta", "info.json")
    print(f"  {s}: {'OK' if os.path.isfile(p) else 'MISSING'}")
PY
}

assemble_mix() {
  echo "==> [mix] Assemble ${MIX_DIR}"
  mkdir -p "${MIX_DIR}"

  # libero_10 (train-ready, with latents)
  local src_10
  src_10="$(find "${LONG_DIR}" -path '*/libero_10/*/meta/info.json' 2>/dev/null | head -1 || true)"
  if [[ -n "${src_10}" ]]; then
    local suite_root
    suite_root="$(dirname "$(dirname "${src_10}")")"
    if [[ ! -e "${MIX_DIR}/libero_10" ]]; then
      ln -sfn "${suite_root}" "${MIX_DIR}/libero_10"
      echo "  linked libero_10 -> ${suite_root}"
    fi
  else
    echo "  [warn] libero_10 not found under ${LONG_DIR}; run: bash $0 long"
  fi

  # spatial / object / goal (raw LeRobot v3.0)
  for suite in "${LIBERO_SUITES[@]}"; do
    if [[ -f "${SUITES_DIR}/${suite}/meta/info.json" ]]; then
      if [[ ! -e "${MIX_DIR}/${suite}" ]]; then
        ln -sfn "${SUITES_DIR}/${suite}" "${MIX_DIR}/${suite}"
        echo "  linked ${suite} -> ${SUITES_DIR}/${suite}"
      fi
    else
      echo "  [warn] ${suite} not found; run: bash $0 suites"
    fi
  done

  if [[ -f "${LONG_DIR}/empty_emb.pt" && ! -f "${MIX_DIR}/empty_emb.pt" ]]; then
    ln -sfn "${LONG_DIR}/empty_emb.pt" "${MIX_DIR}/empty_emb.pt"
    echo "  linked empty_emb.pt"
  fi

  echo ""
  echo "libero-mix layout:"
  find "${MIX_DIR}" -name 'info.json' 2>/dev/null | while read -r f; do
    echo "  ${f}"
  done || true
  echo ""
  echo "Train (libero_10 only, latents ready):"
  echo "  DATASET_PATH=${LONG_DIR} bash script/run_libero_train.sh"
  echo ""
  echo "Train (4 suites): spatial/object/goal need latent extraction + v2.1 conversion."
  echo "  DATASET_PATH=${MIX_DIR} bash script/run_libero_train.sh"
}

case "${TARGET}" in
  long)   download_long ;;
  suites|3suite|spatial-object-goal) download_suites ;;
  mix)    download_long; download_suites; assemble_mix ;;
  all)    download_long; download_suites; assemble_mix ;;
  # legacy alias: full HuggingFaceVLA/libero (40 tasks, no libero_90) — not recommended
  vla)
    echo "[warn] 'vla' downloads all 40 tasks merged; use 'suites' for spatial/object/goal only."
    exit 1
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    echo "Usage: $0 [all|long|suites|mix]"
    exit 1
    ;;
esac

echo "Done."
