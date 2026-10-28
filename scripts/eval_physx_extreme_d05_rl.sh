#!/usr/bin/env bash
# Resume diameter=0.5 extreme contrast for MATD3 / MAPPO only.
# Baselines (APF/FRPN/MPC) already finished under *_physx_closed_d05.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER="${TEACHER:-mpc}"
TRAIN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/train"
LOG_DIR="${PROJECT_RUNS}/isaac_dual_${TAG}/logs"
mkdir -p "$LOG_DIR"

CFG_DIR="${ROOT}/configs/pursuit_v2"
LIVE_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed.json"
D05_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed_d05.json"
BACKUP_CFG="${CFG_DIR}/radar5_dual_extreme_physx_closed.json.bak_d075"
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
  if [[ -f "${BACKUP_CFG}" ]]; then
    mv -f "${BACKUP_CFG}" "${LIVE_CFG}"
  else
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

cp -a "${LIVE_CFG}" "${BACKUP_CFG}"
cp -a "${D05_CFG}" "${LIVE_CFG}"
python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
d = json.loads(Path("${LIVE_CFG}").read_text())
assert float(d["target_diameter"]) == 0.5, d["target_diameter"]
print("[d05-contrast] RL resume live diameter=", d["target_diameter"], "num_envs=${NUM_ENVS}")
PY

MATD3_CKPT="${TRAIN_ROOT}/isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11/best_capture.pt"
MAPPO_CKPT="${TRAIN_ROOT}/isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11/best_capture.pt"
test -f "${MATD3_CKPT}"
test -f "${MAPPO_CKPT}"

MATD3_OUT="${TRAIN_ROOT}/isaac_dual_eval_matd3_${TEACHER}_${TAG}_d05_seed11"
MAPPO_OUT="${TRAIN_ROOT}/isaac_dual_eval_mappo_${TEACHER}_${TAG}_d05_seed11"
mkdir -p "${MATD3_OUT}" "${MAPPO_OUT}"

echo "[d05-contrast] $(date -Is) start matd3" | tee -a "$LOG"
python -u "${ROOT}/evaluate_isaac_pursuit.py" \
  --method matd3 \
  --checkpoint "${MATD3_CKPT}" \
  --levels extreme \
  --no-skip-existing-levels \
  --physx-closed-loop \
  --no-native-lidar \
  --num-envs "${NUM_ENVS}" \
  --seed-start "${SEED_START}" \
  --seed-count "${SEED_COUNT}" \
  --log-every 10 \
  --out "${MATD3_OUT}" \
  2>&1 | tee -a "$LOG"
echo "[d05-contrast] $(date -Is) done matd3" | tee -a "$LOG"

echo "[d05-contrast] $(date -Is) start mappo" | tee -a "$LOG"
python -u "${ROOT}/evaluate_isaac_pursuit.py" \
  --method mappo \
  --checkpoint "${MAPPO_CKPT}" \
  --levels extreme \
  --no-skip-existing-levels \
  --physx-closed-loop \
  --no-native-lidar \
  --num-envs "${NUM_ENVS}" \
  --seed-start "${SEED_START}" \
  --seed-count "${SEED_COUNT}" \
  --log-every 10 \
  --out "${MAPPO_OUT}" \
  2>&1 | tee -a "$LOG"
echo "[d05-contrast] $(date -Is) done mappo" | tee -a "$LOG"
echo "[d05-contrast] $(date -Is) all extreme d05 evals complete" | tee -a "$LOG"
