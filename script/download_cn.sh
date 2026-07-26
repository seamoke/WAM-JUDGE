#!/usr/bin/env bash
# Download models/datasets: ModelScope first, hf-mirror fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KEY="${1:?Usage: $0 <model-or-dataset-key> [output_dir]}"
OUT_DIR="${2:-}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

python - "$KEY" "$OUT_DIR" "$ROOT" <<'PY'
import os
import sys

key, out_dir, root = sys.argv[1:4]

MODEL_MAP = {
    "lingbot-va-base": ("Robbyant/lingbot-va-base", "robbyant/lingbot-va-base", "model"),
    "lingbot-va-posttrain-robotwin": ("Robbyant/lingbot-va-posttrain-robotwin", "robbyant/lingbot-va-posttrain-robotwin", "model"),
    "lingbot-va-posttrain-libero-long": ("Robbyant/lingbot-va-posttrain-libero-long", "robbyant/lingbot-va-posttrain-libero-long", "model"),
}
DATASET_MAP = {
    "robotwin-clean-and-aug-lerobot": ("Robbyant/robotwin-clean-and-aug-lerobot", "robbyant/robotwin-clean-and-aug-lerobot", "dataset"),
    "libero-long-lerobot": ("Robbyant/libero-long-lerobot", "robbyant/libero-long-lerobot", "dataset"),
    "libero-vla-lerobot": ("HuggingFaceVLA/libero", "HuggingFaceVLA/libero", "dataset"),
}

if key in DATASET_MAP:
    ms_id, hf_id, repo_type = DATASET_MAP[key]
    default_out = os.path.join(root, "data", key)
elif key in MODEL_MAP:
    ms_id, hf_id, repo_type = MODEL_MAP[key]
    default_out = os.path.join(root, "checkpoints", key)
else:
    print(f"Unknown key: {key}")
    print("Models:", ", ".join(MODEL_MAP))
    print("Datasets:", ", ".join(DATASET_MAP))
    sys.exit(1)

out_dir = out_dir or default_out
os.makedirs(out_dir, exist_ok=True)

def via_modelscope():
    from modelscope import snapshot_download
    print(f"[ModelScope] {ms_id} -> {out_dir}")
    snapshot_download(ms_id, local_dir=out_dir, cache_dir=out_dir, repo_type=repo_type)

def via_hf_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download
    print(f"[hf-mirror] {hf_id} -> {out_dir}")
    snapshot_download(hf_id, local_dir=out_dir, repo_type=repo_type)

for name, fn in [("ModelScope", via_modelscope), ("hf-mirror", via_hf_mirror)]:
    try:
        fn()
        print(f"OK via {name}: {out_dir}")
        break
    except Exception as e:
        print(f"[warn] {name} failed: {e}")
else:
    sys.exit(1)
PY

echo "Saved to: ${OUT_DIR:-$(python -c "import sys; print({'lingbot-va-base':'checkpoints','libero-long-lerobot':'data'}.get('$KEY','$KEY'))")}"
