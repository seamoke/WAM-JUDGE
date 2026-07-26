#!/usr/bin/env bash
# Setup LIBERO post-training: deps, downloads (CN mirrors), config patch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
MODEL_DIR="${ROOT}/checkpoints/lingbot-va-base"
DATASET_DIR="${ROOT}/data/libero-long-lerobot"
TRAIN_OUT="${ROOT}/train_out/libero"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [1/5] Post-training deps (lerobot --no-deps, keep torch cu126)"
uv pip install "lerobot==0.3.3" scipy wandb --no-deps --index-url "${PYPI_MIRROR}"
uv pip install datasets "huggingface_hub>=0.36" av pyarrow pandas jsonlines \
  --index-url "${PYPI_MIRROR}"

echo "==> [2/5] Download base model (ModelScope -> hf-mirror)"
bash script/download_cn.sh lingbot-va-base "${MODEL_DIR}"

echo "==> [3/5] Download LIBERO dataset (ModelScope -> hf-mirror)"
bash script/download_cn.sh libero-long-lerobot "${DATASET_DIR}"

echo "==> [4/5] Set attn_mode=flex for training"
CONFIG_JSON="${MODEL_DIR}/transformer/config.json"
if [[ -f "${CONFIG_JSON}" ]]; then
  python - <<PY
import json
p = "${CONFIG_JSON}"
with open(p) as f:
    cfg = json.load(f)
cfg["attn_mode"] = "flex"
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print("Updated", p, "attn_mode -> flex")
PY
else
  echo "[warn] ${CONFIG_JSON} not found; skip attn_mode patch"
fi

echo "==> [5/5] Verify layout"
python - <<PY
import os, json, glob
model = "${MODEL_DIR}"
data = "${DATASET_DIR}"
assert os.path.isdir(model), f"missing model: {model}"
assert os.path.isdir(data), f"missing dataset: {data}"
info = glob.glob(os.path.join(data, "**/meta/info.json"), recursive=True)
assert info, f"no LeRobot info.json under {data}"
empty_emb = os.path.join(data, "empty_emb.pt")
if not os.path.isfile(empty_emb):
    for root, _, files in os.walk(data):
        if "empty_emb.pt" in files:
            empty_emb = os.path.join(root, "empty_emb.pt")
            break
print("model:", model)
print("dataset:", data)
print("info.json:", info[0])
print("empty_emb:", empty_emb if os.path.isfile(empty_emb) else "NOT FOUND (required for training)")
PY

mkdir -p "${TRAIN_OUT}"

cat <<EOF

LIBERO training setup complete.

Paths (already written to configs):
  model:   ${MODEL_DIR}
  dataset: ${DATASET_DIR}
  output:  ${TRAIN_OUT}

Start training (4 GPUs):
  bash script/run_libero_train.sh

Or customize:
  NGPU=8 bash script/run_libero_train.sh
EOF
