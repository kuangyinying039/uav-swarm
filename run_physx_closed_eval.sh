#!/usr/bin/env bash
# Held-out eval for physx_closed — baselines + trained ckpts, parallel envs.
# Geometric LiDAR (PhysX RangeSensor does not register under GPU pipeline).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER="${TEACHER:-mpc}"
TRAIN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/train"
LOG_DIR="${PROJECT_RUNS}/isaac_dual_${TAG}/logs"
mkdir -p "$LOG_DIR"

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

NUM_ENVS="${NUM_ENVS:-4}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
# Match train-time residual (MAPPO). Default none = pure RL vs classical tables.
RESIDUAL_BASE="${RESIDUAL_BASE:-none}"
# Eval uses the annealed end weight when training annealed RL toward the teacher.
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT_FINAL:-${RESIDUAL_WEIGHT:-0.70}}"
# Pipeline re-runs must not reuse stale per-level JSON from older ckpts/priors.
SKIP_EXISTING_LEVELS="${SKIP_EXISTING_LEVELS:-0}"
LEVELS=(nominal medium hard severe extreme)

best_ckpt() {
  local dir="$1"
  if [[ -f "${dir}/best_capture.pt" ]]; then echo "${dir}/best_capture.pt"
  else echo "${dir}/final.pt"; fi
}

eval_skip_flag=(--no-skip-existing-levels)
if [[ "${SKIP_EXISTING_LEVELS}" == "1" ]]; then
  eval_skip_flag=(--skip-existing-levels)
fi

eval_baseline() {
  local method="$1"
  local out="${TRAIN_ROOT}/isaac_dual_eval_baseline_${method}_${TAG}"
  mkdir -p "$out"
  echo "[eval-closed] $(date -Is) baseline ${method} num_envs=${NUM_ENVS} skip_existing=${SKIP_EXISTING_LEVELS}"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --levels "${LEVELS[@]}" \
    "${eval_skip_flag[@]}" \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    --out "$out" \
    | tee -a "${LOG_DIR}/eval_baseline_${method}_${TAG}.log"
}

eval_ckpt() {
  local method="$1" train_name="$2" out_name="$3"
  local train_dir="${TRAIN_ROOT}/${train_name}"
  local ckpt
  ckpt="$(best_ckpt "$train_dir")"
  local out="${TRAIN_ROOT}/${out_name}"
  local extra=()
  if [[ "$method" == "mappo" && "${RESIDUAL_BASE}" != "none" ]]; then
    # Train uses residual mode for MAPPO; eval must match.
    extra+=(--residual-base "${RESIDUAL_BASE}" --residual-mode residual --residual-weight "${RESIDUAL_WEIGHT}")
  fi
  mkdir -p "$out"
  echo "[eval-closed] $(date -Is) ${method} ckpt=${ckpt} num_envs=${NUM_ENVS} residual=${RESIDUAL_BASE}:${RESIDUAL_WEIGHT}"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels "${LEVELS[@]}" \
    "${eval_skip_flag[@]}" \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    "${extra[@]}" \
    --out "$out" \
    | tee -a "${LOG_DIR}/${out_name}.log"
}

eval_baseline apf
eval_baseline frpn
eval_baseline mpc
eval_ckpt matd3 "isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11" "isaac_dual_eval_matd3_${TEACHER}_${TAG}_seed11"
eval_ckpt mappo "isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11" "isaac_dual_eval_mappo_${TEACHER}_${TAG}_seed11"
echo "[eval-closed] done $(date -Is) teacher=${TEACHER}"
