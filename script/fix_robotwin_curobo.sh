#!/usr/bin/env bash
# Pin curobo to v0.7.x API used by RoboTwin planner.py (not v0.8 rewrite).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.7}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://ghfast.top/https://github.com}"
CUROBO_DIR="${ROBOTWIN_DIR}/envs/curobo"

if python - <<'PY' 2>/dev/null; then
from curobo.types.math import Pose  # noqa: F401
from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
print("curobo API already compatible")
PY
  exit 0
fi

echo "Repairing curobo (install ${CUROBO_TAG}) ..."
rm -rf "${CUROBO_DIR}"
git clone --depth 1 --branch "${CUROBO_TAG}" "${GITHUB_MIRROR}/NVlabs/curobo.git" "${CUROBO_DIR}"
pip install -e "${CUROBO_DIR}" --no-build-isolation

# warp-lang>=1.14 removed wp.torch.*; patch curobo 0.7.x for compatibility.
sed -i 's/wp\.torch\.device_from_torch/wp.device_from_torch/g' \
  "${CUROBO_DIR}/src/curobo/geom/sdf/world_mesh.py"

python - <<'PY'
from curobo.types.math import Pose  # noqa: F401
from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
print("curobo API repair OK")
PY
