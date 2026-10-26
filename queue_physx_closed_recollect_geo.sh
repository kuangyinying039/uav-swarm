#!/usr/bin/env bash
# Recollect FRPN prior + DAgger with geometric LiDAR, then retrain physx_closed.
#
# Prior/DAgger use hybrid Isaac (physx_closed=False): FRPN reaches ~75% there.
# Closed-loop Lee currently lags pursuers (~8% FRPN even with kinematic
# evader), so teacher demos are collected in hybrid, then RL trains/evals on
# physx_closed + kinematic_evader + geometric sensing.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"
LOG="${LOG_DIR}/recollect_geo_pipeline.log"
exec >>"$LOG" 2>&1

NUM_ENVS="${NUM_ENVS:-8}"
# Same hard numbers as closed ladder; hybrid backend for teacher success.
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${DATA_ROOT}/archive_zero_success_${STAMP}"

echo "[recollect-geo] $(date -Is) start NUM_ENVS=${NUM_ENVS}"

if pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py' >/dev/null 2>&1; then
  echo "[recollect-geo] another Isaac job is running; abort" >&2
  exit 1
fi

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
cd "${ROOT}"

mkdir -p "$ARCHIVE"
for item in hard_frpn_prior.pt hard_frpn_prior.json dagger_frpn_hard_seed11; do
  if [[ -e "${DATA_ROOT}/${item}" ]]; then
    mv "${DATA_ROOT}/${item}" "${ARCHIVE}/"
    echo "[recollect-geo] archived ${item} -> ${ARCHIVE}/"
  fi
done

PRIOR_PT="${DATA_ROOT}/hard_frpn_prior.pt"
echo "[recollect-geo] $(date -Is) collect FRPN prior on HYBRID Isaac + geometric lidar (min_successes=40)"
python -u "${ROOT}/collect_isaac_pursuit_data.py" \
  --env-config "$ENV_CFG" \
  --teacher frpn \
  --demo-seed 2000000 \
  --rollouts 200 \
  --no-native-lidar \
  --min-successes 40 \
  --dataset "$PRIOR_PT"

DAGGER_DIR="${DATA_ROOT}/dagger_frpn_hard_seed11"
echo "[recollect-geo] $(date -Is) FRPN DAgger on HYBRID Isaac + geometric lidar"
python -u "${ROOT}/pretrain_isaac_pursuit_actor.py" \
  --env-config "$ENV_CFG" \
  --teacher frpn \
  --seed 11 --demo-seed 2000000 \
  --bc-episodes 50 --bc-updates 1500 \
  --dagger-rounds 2 --dagger-episodes 40 --dagger-updates 800 \
  --no-native-lidar \
  --out "$DAGGER_DIR"

TRAIN_A6="${RUN_ROOT}/train/isaac_dual_A6_frpn_hard_${TAG}_seed11"
TRAIN_MAPPO="${RUN_ROOT}/train/isaac_dual_mappo_plus_frpn_hard_${TAG}_seed11"
for item in "$TRAIN_A6" "$TRAIN_MAPPO"; do
  if [[ -e "$item" ]]; then
    mv "$item" "${ARCHIVE}/$(basename "$item")"
    echo "[recollect-geo] archived $(basename "$item")"
  fi
done

echo "[recollect-geo] $(date -Is) train physx_closed with --no-native-lidar (kinematic_evader default ON)"
NUM_ENVS="${NUM_ENVS}" bash "${ROOT}/run_physx_closed_train.sh"

echo "[recollect-geo] $(date -Is) eval physx_closed with --no-native-lidar"
NUM_ENVS="${NUM_ENVS}" bash "${ROOT}/run_physx_closed_eval.sh"

echo "[recollect-geo] $(date -Is) pipeline complete"
