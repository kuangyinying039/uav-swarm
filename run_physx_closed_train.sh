#!/usr/bin/env bash
# Retrain MAPPO+ / MATD3-A6 on pure PhysX-closed hard with parallel envs.
# Default expert is MPC (geometric LiDAR). Override with TEACHER=frpn if needed.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER="${TEACHER:-mpc}"
OUT_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/train"
DATA_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/data"
LOG_DIR="${PROJECT_RUNS}/isaac_dual_${TAG}/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
export OMNIDRONES_SOURCE=/opt/lab/src/OmniDrones-isaac41
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

ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
ACTOR_INIT="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/best_imitation.pt"
PRIOR="${DATA_ROOT}/hard_${TEACHER}_prior.pt"
DEMOS="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/expert_episodes.pt"
AUX_DEMOS="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/correction_episodes.pt"
NUM_ENVS="${NUM_ENVS:-4}"
# Held-out validation during RL. Prefer training throughput; roll back only on
# clear drops. Override with VAL_EPISODES / VAL_INTERVAL.
VAL_EPISODES="${VAL_EPISODES:-20}"
VAL_INTERVAL="${VAL_INTERVAL:-250}"
# Optional MAPPO residual-on-MPC. Default off so primary tables compare pure RL
# against classical baselines; set RESIDUAL_BASE=mpc to enable residual RL.
RESIDUAL_BASE="${RESIDUAL_BASE:-none}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.45}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.70}"
# If set, resume MATD3-A6 from this checkpoint (skips critic pretrain / ep0 val).
RESUME_CKPT="${RESUME_CKPT:-}"

require() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
}

require "$ACTOR_INIT"
require "$PRIOR"
require "$DEMOS"
require "$AUX_DEMOS"
if [[ ! -f "${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/dagger_validation.json" ]]; then
  echo "missing dagger_validation.json (imitation was not validated; rebuild DAgger)" >&2
  exit 1
fi
if ! python - <<PY
from pursuit.data.transition_dataset import prior_is_complete
import sys
sys.exit(0 if prior_is_complete("${PRIOR}", min_episodes=200, min_successes=200, allow_failures=False) else 1)
PY
then
  echo "prior is incomplete or failure-polluted: ${PRIOR}" >&2
  exit 1
fi

run_matd3() {
  local name="$1"
  local out="${OUT_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  mkdir -p "$out"
  local resume_args=()
  if [[ -n "${RESUME_CKPT}" ]]; then
    resume_args+=(--checkpoint "${RESUME_CKPT}")
    echo "[physx_closed] $(date -Is) resume ${name} from ${RESUME_CKPT} num_envs=${NUM_ENVS} val=${VAL_EPISODES}@${VAL_INTERVAL} teacher=${TEACHER}" | tee -a "$log"
  else
    : >"$log"
    echo "[physx_closed] $(date -Is) start ${name} num_envs=${NUM_ENVS} val=${VAL_EPISODES}@${VAL_INTERVAL} teacher=${TEACHER}" | tee -a "$log"
  fi
  if ! python -u "${ROOT}/train_isaac_pursuit.py" matd3 --variant A6 \
    --num-envs "${NUM_ENVS}" \
    --env-config "$ENV_CFG" \
    --physx-closed-loop \
    --no-native-lidar \
    --episodes 1000 \
    --seed 11 \
    --validation-episodes "${VAL_EPISODES}" \
    --validation-interval "${VAL_INTERVAL}" \
    --budget-checkpoints 100,300,1000 \
    --batch-size 256 \
    --hidden-dim 128 \
    --actor-init "$ACTOR_INIT" \
    --prior "$PRIOR" \
    "${resume_args[@]}" \
    --out "$out" \
    >>"$log" 2>&1
  then
    echo "[physx_closed] $(date -Is) FAILED ${name}" | tee -a "$log"
    exit 1
  fi
  if [[ ! -f "${out}/final.pt" ]]; then
    if [[ ! -f "${out}/best_capture.pt" ]]; then
      echo "[physx_closed] $(date -Is) FAILED ${name}: no checkpoint written" | tee -a "$log"
      exit 1
    fi
  fi
  echo "[physx_closed] $(date -Is) done ${name}" | tee -a "$log"
}

run_mappo() {
  local name="$1"
  local out="${OUT_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  mkdir -p "$out"
  : >"$log"
  echo "[physx_closed] $(date -Is) start ${name} num_envs=${NUM_ENVS} teacher=${TEACHER}" | tee -a "$log"
  mappo_extra=()
  if [[ "${RESIDUAL_BASE}" != "none" ]]; then
    mappo_extra+=(--residual-base "${RESIDUAL_BASE}" --residual-weight "${RESIDUAL_WEIGHT}")
    if [[ -n "${RESIDUAL_WEIGHT_FINAL}" && "${RESIDUAL_WEIGHT_FINAL}" != "none" ]]; then
      mappo_extra+=(--residual-weight-final "${RESIDUAL_WEIGHT_FINAL}")
    fi
  fi
  if ! python -u "${ROOT}/train_isaac_pursuit.py" mappo --variant mappo_plus \
    --num-envs "${NUM_ENVS}" \
    --env-config "$ENV_CFG" \
    --physx-closed-loop \
    --no-native-lidar \
    --episodes 1000 \
    --seed 11 \
    --validation-episodes "${VAL_EPISODES}" \
    --validation-interval "${VAL_INTERVAL}" \
    --budget-checkpoints 100,300,1000 \
    --batch-size 256 \
    --hidden-dim 128 \
    --actor-init "$ACTOR_INIT" \
    --demos "$DEMOS" \
    --aux-demos "$AUX_DEMOS" \
    "${mappo_extra[@]}" \
    --out "$out" \
    >>"$log" 2>&1
  then
    echo "[physx_closed] $(date -Is) FAILED ${name}" | tee -a "$log"
    exit 1
  fi
  if [[ ! -f "${out}/final.pt" ]]; then
    if [[ ! -f "${out}/best_capture.pt" ]]; then
      echo "[physx_closed] $(date -Is) FAILED ${name}: no checkpoint written" | tee -a "$log"
      exit 1
    fi
  fi
  echo "[physx_closed] $(date -Is) done ${name}" | tee -a "$log"
}

SKIP_MATD3="${SKIP_MATD3:-0}"
if [[ "$SKIP_MATD3" != "1" ]]; then
  run_matd3 "isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11"
else
  echo "[physx_closed] $(date -Is) skip MATD3-A6 (SKIP_MATD3=1)"
fi
run_mappo "isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11"

echo "[physx_closed] all train jobs finished $(date -Is) teacher=${TEACHER}"
