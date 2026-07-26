#!/usr/bin/env bash
# Resume RoboTwin eval setup after partial install (pytorch3d / curobo / assets).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://ghfast.top/https://github.com}"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export ROBOTWIN_DIR

echo "==> setuptools (for sapien pkg_resources)"
pip install -q 'setuptools<81'  # sapien needs pkg_resources

if ! python -c "import pytorch3d" 2>/dev/null; then
  echo "==> pytorch3d"
  pip install "git+${GITHUB_MIRROR}/facebookresearch/pytorch3d.git@stable" --no-build-isolation
else
  echo "==> pytorch3d already installed"
fi

echo "==> patch sapien / mplib (from RoboTwin _install.sh)"
SAPIEN_LOCATION="$(pip show sapien | awk '/^Location:/ {print $2}')/sapien"
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
if [[ -f "${URDF_LOADER}" ]]; then
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
fi
MPLIB_LOCATION="$(pip show mplib | awk '/^Location:/ {print $2}')/mplib"
PLANNER="${MPLIB_LOCATION}/planner.py"
if [[ -f "${PLANNER}" ]]; then
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
fi

echo "==> curobo"
pushd "${ROBOTWIN_DIR}/envs" >/dev/null
if [[ ! -d curobo/.git ]]; then
  rm -rf curobo
  git clone --depth 1 "${GITHUB_MIRROR}/NVlabs/curobo.git" curobo
fi
pip install -e curobo --no-build-isolation
popd >/dev/null

echo "==> sim assets"
bash "${ROOT}/script/download_robotwin_sim_assets.sh"

echo "==> verify"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python - <<'PY'
import os
os.environ.setdefault("ROBOTWIN_DIR", os.environ["ROBOTWIN_DIR"])
import sapien.core  # noqa: F401
from evaluation.robotwin.test_render import Sapien_TEST
Sapien_TEST()
print("RoboTwin render OK")
PY

cat > "${ROOT}/script/.robotwin_eval_env" <<EOF
export ROBOTWIN_DIR="${ROBOTWIN_DIR}"
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH="\${PYTHONPATH:-}:${ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EOF

echo "RoboTwin eval setup complete (no eval launched)."
