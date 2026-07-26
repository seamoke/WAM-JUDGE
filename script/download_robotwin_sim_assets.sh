#!/usr/bin/env bash
# Download RoboTwin sim assets (background_texture / embodiments / objects) via HF mirror.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"
ASSETS_DIR="${ROBOTWIN_DIR}/assets"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_ENDPOINT

mkdir -p "${ASSETS_DIR}"
cd "${ASSETS_DIR}"

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = "TianxingChen/RoboTwin2.0"
files = ["background_texture.zip", "embodiments.zip", "objects.zip"]
local_dir = os.getcwd()
for name in files:
    if os.path.isfile(name):
        print(f"skip existing {name}")
        continue
    print(f"downloading {name} ...")
    path = hf_hub_download(
        repo_id=repo,
        filename=name,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True,
    )
    print(f"done {path}")
PY

for z in background_texture.zip embodiments.zip objects.zip; do
  if [[ -f "${z}" && ! -d "${z%.zip}" ]]; then
    echo "unzip ${z} ..."
    unzip -q -o "${z}"
    rm -f "${z}"
  fi
done

cd "${ROBOTWIN_DIR}"
if [[ ! -d "${ASSETS_DIR}/embodiments" ]]; then
  echo "ERROR: assets incomplete after download" >&2
  exit 1
fi
ASSETS_PATH="${ROBOTWIN_DIR}" python ./script/update_embodiment_config_path.py

echo "RoboTwin sim assets ready under ${ASSETS_DIR}"
