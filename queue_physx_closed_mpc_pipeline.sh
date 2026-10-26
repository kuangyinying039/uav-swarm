#!/usr/bin/env bash
# Pure PhysX-closed full pipeline with MPC expert (geometric LiDAR).
# Collect → DAgger → MATD3-A6 + MAPPO+ train → ladder eval.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"
LOG="${LOG_DIR}/mpc_closed_pipeline.log"
exec >>"$LOG" 2>&1

NUM_ENVS="${NUM_ENVS:-4}"
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${DATA_ROOT}/archive_pre_mpc_${STAMP}"

echo "[mpc-closed] $(date -Is) start NUM_ENVS=${NUM_ENVS}"

if pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; then
  echo "[mpc-closed] another Isaac job is running; abort" >&2
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
for item in hard_frpn_prior.pt hard_frpn_prior.json hard_mpc_prior.pt hard_mpc_prior.json \
            hard_mpc_prior.progress.pt hard_mpc_prior.progress.json \
            hard_frpn_prior.progress.pt hard_frpn_prior.progress.json \
            hard_mpc_prior.polluted_incomplete.pt hard_mpc_prior.polluted_incomplete.json \
            hard_mpc_prior.pt.tmp dagger_frpn_hard_seed11 dagger_mpc_hard_seed11; do
  if [[ -e "${DATA_ROOT}/${item}" ]]; then
    mv "${DATA_ROOT}/${item}" "${ARCHIVE}/"
    echo "[mpc-closed] archived ${item} -> ${ARCHIVE}/"
  fi
done
# Stale eval JSON would be reused by --skip-existing-levels; archive them too.
shopt -s nullglob
for item in "${RUN_ROOT}/train"/isaac_dual_eval_*; do
  mv "$item" "${ARCHIVE}/$(basename "$item")"
  echo "[mpc-closed] archived $(basename "$item") -> ${ARCHIVE}/"
done
shopt -u nullglob

PRIOR_PT="${DATA_ROOT}/hard_mpc_prior.pt"
echo "[mpc-closed] $(date -Is) collect MPC prior on physx_closed + geometric lidar (success-only, no failure pollution)"
# Progress writes to hard_mpc_prior.progress.pt; final .pt only after 200 successes.
python -u "${ROOT}/collect_isaac_pursuit_data.py" \
  --env-config "$ENV_CFG" \
  --teacher mpc \
  --demo-seed 2000000 \
  --rollouts 200 \
  --max-attempts 800 \
  --no-keep-failures \
  --min-successes 200 \
  --physx-closed-loop \
  --no-native-lidar \
  --force \
  --dataset "$PRIOR_PT"

DAGGER_DIR="${DATA_ROOT}/dagger_mpc_hard_seed11"
echo "[mpc-closed] $(date -Is) MPC DAgger on physx_closed + geometric lidar (success-only BC + val select)"
# demo-seed 2100000 keeps BC/correction banks disjoint from prior seeds 2000000+.
python -u "${ROOT}/pretrain_isaac_pursuit_actor.py" \
  --env-config "$ENV_CFG" \
  --teacher mpc \
  --seed 11 --demo-seed 2100000 \
  --bc-episodes 80 --bc-attempts 320 --bc-updates 2000 \
  --success-only-bc \
  --dagger-rounds 3 --dagger-episodes 40 --dagger-updates 800 \
  --dagger-validation-episodes 20 --dagger-validation-seed 5000000 \
  --physx-closed-loop \
  --no-native-lidar \
  --out "$DAGGER_DIR"

for item in \
  "${RUN_ROOT}/train/isaac_dual_A6_frpn_hard_${TAG}_seed11" \
  "${RUN_ROOT}/train/isaac_dual_mappo_plus_frpn_hard_${TAG}_seed11" \
  "${RUN_ROOT}/train/isaac_dual_A6_mpc_hard_${TAG}_seed11" \
  "${RUN_ROOT}/train/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11"
do
  if [[ -e "$item" ]]; then
    mv "$item" "${ARCHIVE}/$(basename "$item")"
    echo "[mpc-closed] archived $(basename "$item")"
  fi
done

# Prefer on-disk override so num_envs can be raised mid-pipeline without killing collect.
# Bench: 8 stable; 12+ previously OOM with native LiDAR. Geometric closed: try 10.
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
NUM_ENVS="${NUM_ENVS:-10}"
echo "[mpc-closed] $(date -Is) train NUM_ENVS=${NUM_ENVS}"
NUM_ENVS="${NUM_ENVS}" TEACHER=mpc bash "${ROOT}/run_physx_closed_train.sh"

echo "[mpc-closed] $(date -Is) eval NUM_ENVS=${NUM_ENVS}"
NUM_ENVS="${NUM_ENVS}" TEACHER=mpc SKIP_EXISTING_LEVELS=0 bash "${ROOT}/run_physx_closed_eval.sh"

echo "[mpc-closed] $(date -Is) pipeline complete"
