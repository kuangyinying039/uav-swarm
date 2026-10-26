#!/usr/bin/env bash
# After postlc eval finishes: full physx_closed redo —
# smoke → FRPN prior collect → FRPN DAgger → parallel train → parallel eval.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"
LOG="${LOG_DIR}/queue_pipeline.log"
exec >>"$LOG" 2>&1

NUM_ENVS="${NUM_ENVS:-8}"
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
POSTLC_EVAL_PID_FILE="${PROJECT_RUNS}/isaac_dual_postlc/logs/queue_parallel_eval_postlc.pid"

echo "[queue] $(date -Is) waiting for postlc eval / GPU isaac jobs to finish"

wait_gpu_idle() {
  while true; do
    busy=0
    if [[ -f "$POSTLC_EVAL_PID_FILE" ]]; then
      local ppid
      ppid="$(cat "$POSTLC_EVAL_PID_FILE" 2>/dev/null || true)"
      if [[ -n "${ppid}" ]] && kill -0 "$ppid" 2>/dev/null; then
        busy=1
      fi
    fi
    if pgrep -f 'evaluate_isaac_pursuit.py|train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|queue_parallel_eval_postlc.sh' >/dev/null 2>&1; then
      busy=1
    fi
    if (( busy == 0 )); then
      break
    fi
    sleep 60
  done
}

wait_gpu_idle
# Brief settle after Kit teardown.
sleep 30
echo "[queue] $(date -Is) GPU idle — starting physx_closed full pipeline NUM_ENVS=${NUM_ENVS}"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set -u
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

echo "[queue] $(date -Is) smoke"
SMOKE_NUM_ENVS=2 python -u "${ROOT}/isaac_smoke_physx_closed.py"

PRIOR_PT="${DATA_ROOT}/hard_frpn_prior.pt"
PRIOR_JSON="${DATA_ROOT}/hard_frpn_prior.json"
if [[ -f "$PRIOR_PT" && -f "$PRIOR_JSON" ]]; then
  echo "[queue] $(date -Is) skip FRPN prior (exists)"
else
  echo "[queue] $(date -Is) collect FRPN prior 200 rollouts"
  # Geometric LiDAR for teacher data (native range sensors are flaky during
  # long collect); dynamics still use PhysX closed-loop articulations.
  python -u "${ROOT}/collect_isaac_pursuit_data.py" \
    --env-config "$ENV_CFG" \
    --teacher frpn \
    --demo-seed 2000000 \
    --rollouts 200 \
    --physx-closed-loop \
    --dataset "$PRIOR_PT"
fi

DAGGER_DIR="${DATA_ROOT}/dagger_frpn_hard_seed11"
if [[ -f "${DAGGER_DIR}/best_imitation.pt" && -f "${DAGGER_DIR}/imitation_manifest.json" ]]; then
  echo "[queue] $(date -Is) skip DAgger (exists)"
else
  echo "[queue] $(date -Is) FRPN DAgger seed=11"
  python -u "${ROOT}/pretrain_isaac_pursuit_actor.py" \
    --env-config "$ENV_CFG" \
    --teacher frpn \
    --seed 11 --demo-seed 2000000 \
    --bc-episodes 50 --bc-updates 1500 \
    --dagger-rounds 2 --dagger-episodes 40 --dagger-updates 800 \
    --physx-closed-loop \
    --out "$DAGGER_DIR"
fi

echo "[queue] $(date -Is) train (parallel num_envs=${NUM_ENVS})"
NUM_ENVS="${NUM_ENVS}" bash "${ROOT}/run_physx_closed_train.sh"

echo "[queue] $(date -Is) eval (parallel num_envs=${NUM_ENVS})"
NUM_ENVS="${NUM_ENVS}" bash "${ROOT}/run_physx_closed_eval.sh"

echo "[queue] $(date -Is) pipeline complete"
