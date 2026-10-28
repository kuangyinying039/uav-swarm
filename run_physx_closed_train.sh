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
# P0 validation: larger held-out + denser checks; early-stop kicks in after
# repeated capture-rate rollbacks (see Matd3Config.max_safeguard_rollbacks).
# Formal tables MUST use best_capture.pt — never training-roll summary.json.
VAL_EPISODES="${VAL_EPISODES:-40}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
# Optional residual/blend on a classical teacher. Default off so primary tables
# compare pure RL against classical baselines; set RESIDUAL_BASE=mpc|frpn to enable.
RESIDUAL_BASE="${RESIDUAL_BASE:-none}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.45}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.70}"
# If set, resume MATD3-A6 from this checkpoint (skips critic pretrain / ep0 val).
RESUME_CKPT="${RESUME_CKPT:-}"
# P0 budget: still allow 1000 ep, but keep mid checkpoints; collapse usually
# appears by ~300–500 so early-stop / best_capture are the real selectors.
EPISODES="${EPISODES:-1000}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-100,300,500,1000}"
# Optional name suffix so residual runs never overwrite pure-RL dirs.
# Example: NAME_SUFFIX=_residual_mpc_ep200
NAME_SUFFIX="${NAME_SUFFIX:-}"
SKIP_MAPPO="${SKIP_MAPPO:-0}"

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
# Near-miss priors (keeps_near_misses) are allowed; all-failure pollution is not.
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
  residual_args=()
  if [[ "${RESIDUAL_BASE}" != "none" ]]; then
    residual_args+=(--residual-base "${RESIDUAL_BASE}" --residual-weight "${RESIDUAL_WEIGHT}")
    if [[ -n "${RESIDUAL_WEIGHT_FINAL}" && "${RESIDUAL_WEIGHT_FINAL}" != "none" ]]; then
      residual_args+=(--residual-weight-final "${RESIDUAL_WEIGHT_FINAL}")
    fi
  fi
  if ! python -u "${ROOT}/train_isaac_pursuit.py" matd3 --variant A6 \
    --num-envs "${NUM_ENVS}" \
    --env-config "$ENV_CFG" \
    --physx-closed-loop \
    --no-native-lidar \
    --episodes "${EPISODES}" \
    --seed 11 \
    --validation-episodes "${VAL_EPISODES}" \
    --validation-interval "${VAL_INTERVAL}" \
    --budget-checkpoints "${BUDGET_CHECKPOINTS}" \
    --batch-size 256 \
    --hidden-dim 128 \
    --actor-init "$ACTOR_INIT" \
    --prior "$PRIOR" \
    "${residual_args[@]}" \
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
  echo "[physx_closed] $(date -Is) start ${name} num_envs=${NUM_ENVS} episodes=${EPISODES} residual=${RESIDUAL_BASE} teacher=${TEACHER}" | tee -a "$log"
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
    --episodes "${EPISODES}" \
    --seed 11 \
    --validation-episodes "${VAL_EPISODES}" \
    --validation-interval "${VAL_INTERVAL}" \
    --budget-checkpoints "${BUDGET_CHECKPOINTS}" \
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
# PARALLEL_TRAIN=1: launch A6 and MAPPO together (each uses PARALLEL_NUM_ENVS).
# Default PARALLEL_NUM_ENVS = max(4, NUM_ENVS/2) to avoid dual-Isaac OOM on one GPU.
PARALLEL_TRAIN="${PARALLEL_TRAIN:-0}"
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-}"
if [[ -z "${PARALLEL_NUM_ENVS}" ]]; then
  PARALLEL_NUM_ENVS=$(( NUM_ENVS / 2 ))
  if [[ "${PARALLEL_NUM_ENVS}" -lt 4 ]]; then
    PARALLEL_NUM_ENVS=4
  fi
fi

if [[ "${PARALLEL_TRAIN}" == "1" && "${SKIP_MATD3}" != "1" && "${SKIP_MAPPO}" != "1" ]]; then
  echo "[physx_closed] $(date -Is) parallel train A6+MAPPO each num_envs=${PARALLEL_NUM_ENVS} (from NUM_ENVS=${NUM_ENVS})"
  (
    NUM_ENVS="${PARALLEL_NUM_ENVS}" run_matd3 "isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
  ) &
  pid_a6=$!
  (
    NUM_ENVS="${PARALLEL_NUM_ENVS}" run_mappo "isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
  ) &
  pid_mappo=$!
  status=0
  if ! wait "${pid_a6}"; then
    echo "[physx_closed] $(date -Is) FAILED parallel A6 (pid=${pid_a6})"
    status=1
  fi
  if ! wait "${pid_mappo}"; then
    echo "[physx_closed] $(date -Is) FAILED parallel MAPPO (pid=${pid_mappo})"
    status=1
  fi
  if [[ "${status}" -ne 0 ]]; then
    exit 1
  fi
else
  if [[ "$SKIP_MATD3" != "1" ]]; then
    run_matd3 "isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
  else
    echo "[physx_closed] $(date -Is) skip MATD3-A6 (SKIP_MATD3=1)"
  fi
  if [[ "$SKIP_MAPPO" != "1" ]]; then
    run_mappo "isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
  else
    echo "[physx_closed] $(date -Is) skip MAPPO+ (SKIP_MAPPO=1)"
  fi
fi

echo "[physx_closed] all train jobs finished $(date -Is) teacher=${TEACHER} suffix=${NAME_SUFFIX:-none}"
