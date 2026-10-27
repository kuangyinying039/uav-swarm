#!/usr/bin/env bash
# Extreme-only A/B eval: physx_closed dynamics with target_diameter=0.5 (radius 1.0).
# NOT the formal ladder (formal extreme uses diameter=0.75).
#
# Temporarily points the level loader at the d05 JSON by swapping the live
# extreme_physx_closed file; always restores diameter=0.75 production on exit.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TRAIN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/train"
LOG_DIR="${PROJECT_RUNS}/isaac_dual_${TAG}/logs"
mkdir -p "$LOG_DIR"

CFG_DIR="${ROOT}/configs/pursuit_v2"
LIVE_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed.json"
D05_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed_d05.json"
PROD_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed.json.bak_d075"
LOG="${LOG_DIR}/eval_extreme_d05_contrast.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${PROJECT_RUNS}/isaac_dual_${TAG}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${PROJECT_RUNS}/isaac_dual_${TAG}/num_envs.override")"
fi
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set +u
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES

restore_cfg() {
  if [[ -f "${PROD_CFG}" ]]; then
    cp -a "${PROD_CFG}" "${LIVE_CFG}"
  else
    # Fallback: rewrite diameter=0.75 in place if bak missing.
    python3 - <<'PY'
import json
from pathlib import Path
p = Path("/srv/data/projects/kyy/xieton/repro/configs/pursuit_v2/radar5_dual_extreme_physx_closed.json")
d = json.loads(p.read_text())
d["target_diameter"] = 0.75
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  fi
  echo "[d05-contrast] $(date -Is) restored formal extreme config to diameter=0.75" | tee -a "$LOG"
}
trap restore_cfg EXIT

if [[ ! -f "${D05_CFG}" ]]; then
  echo "missing ${D05_CFG}" >&2
  exit 1
fi
# Keep a known-good production snapshot before swap.
cp -a "${LIVE_CFG}" "${PROD_CFG}"
python3 - <<PY
import json
from pathlib import Path
live = json.loads(Path("${LIVE_CFG}").read_text())
assert float(live["target_diameter"]) == 0.75, live["target_diameter"]
PY
cp -a "${D05_CFG}" "${LIVE_CFG}"
echo "[d05-contrast] $(date -Is) swapped extreme config → diameter=0.5 (num_envs=${NUM_ENVS})" | tee -a "$LOG"
python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
d = json.loads(Path("${LIVE_CFG}").read_text())
assert float(d["target_diameter"]) == 0.5, d["target_diameter"]
print("live extreme target_diameter=", d["target_diameter"])
PY

cd "${ROOT}"
for method in apf frpn mpc; do
  out="${TRAIN_ROOT}/isaac_dual_eval_baseline_${method}_${TAG}_d05"
  mkdir -p "$out"
  echo "[d05-contrast] $(date -Is) baseline ${method}" | tee -a "$LOG"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --levels extreme \
    --no-skip-existing-levels \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    --out "$out" \
    | tee -a "$LOG"
done

echo "[d05-contrast] $(date -Is) done" | tee -a "$LOG"
