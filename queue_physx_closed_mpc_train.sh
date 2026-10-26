#!/usr/bin/env bash
# Resume: DAgger → train → eval (prior already collected).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"
LOG="${LOG_DIR}/mpc_closed_train_pipeline.log"
exec >>"$LOG" 2>&1

NUM_ENVS="${NUM_ENVS:-4}"
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
PRIOR_PT="${DATA_ROOT}/hard_mpc_prior.pt"
DAGGER_DIR="${DATA_ROOT}/dagger_mpc_hard_seed11"

echo "[mpc-train] $(date -Is) start NUM_ENVS=${NUM_ENVS}"

if pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; then
  echo "[mpc-train] another Isaac job is running; abort" >&2
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

# Refuse incomplete / failure-polluted priors (the old 75-ep interrupted collect).
if ! python - <<PY
from pursuit.data.transition_dataset import prior_is_complete
import sys
sys.exit(0 if prior_is_complete("${PRIOR_PT}", min_episodes=200, min_successes=200, allow_failures=False) else 1)
PY
then
  echo "[mpc-train] $(date -Is) prior missing/incomplete/polluted; collecting clean success-only prior" >&2
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
else
  echo "[mpc-train] $(date -Is) clean prior ready (${PRIOR_PT})"
fi

# Rebuild DAgger unless a success-only + validated checkpoint already exists.
NEED_DAGGER=1
if [[ -f "${DAGGER_DIR}/best_imitation.pt" && -f "${DAGGER_DIR}/imitation_manifest.json" && -f "${DAGGER_DIR}/dagger_validation.json" ]]; then
  if DAGGER_MANIFEST="${DAGGER_DIR}/imitation_manifest.json" python - <<'PY'
import json, os, sys
m = json.load(open(os.environ["DAGGER_MANIFEST"]))
ok = (
    bool(m.get("success_only_bc"))
    and int(m.get("dagger_validation_episodes", 0)) >= 20
    and int(m.get("bc_episodes", 0)) >= 80
)
sys.exit(0 if ok else 1)
PY
  then
    NEED_DAGGER=0
  fi
fi
if [[ "$NEED_DAGGER" == "1" ]]; then
  rm -rf "$DAGGER_DIR"
  echo "[mpc-train] $(date -Is) MPC DAgger (success-only BC + val select)"
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
else
  echo "[mpc-train] $(date -Is) skip DAgger (validated success-only checkpoint exists)"
fi

echo "[mpc-train] $(date -Is) train"
NUM_ENVS="${NUM_ENVS}" TEACHER=mpc bash "${ROOT}/run_physx_closed_train.sh"

echo "[mpc-train] $(date -Is) eval"
NUM_ENVS="${NUM_ENVS}" TEACHER=mpc SKIP_EXISTING_LEVELS=0 bash "${ROOT}/run_physx_closed_eval.sh"

echo "[mpc-train] $(date -Is) pipeline complete"
