#!/usr/bin/env bash
# Stage3+4 only: residual A6∥MAPPO train then locked hard eval.
# Assumes prior + π0/DAgger artifacts already exist.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/teacher_enhance_pipeline.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
PRIOR_PT="${DATA_ROOT}/hard_mpc_prior.pt"
DAGGER_DIR="${DATA_ROOT}/dagger_mpc_hard_seed11"
PI0="${DAGGER_DIR}/pi0_frozen.pt"
RESIDUAL_EPISODES="${RESIDUAL_EPISODES:-200}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.0}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.25}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-5}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
NAME_SUFFIX="_residual_mpc_zeroinit_ep${RESIDUAL_EPISODES}"
MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES:-}"
if [[ -z "${MIN_PRIOR_SUCCESSES}" && -f "${DATA_ROOT}/hard_mpc_prior.min_successes" ]]; then
  MIN_PRIOR_SUCCESSES="$(tr -d '[:space:]' < "${DATA_ROOT}/hard_mpc_prior.min_successes")"
fi
MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES:-164}"

{
  echo "[teacher-enhance] $(date -Is) train+eval only NUM_ENVS=${NUM_ENVS} PARALLEL=${PARALLEL_TRAIN}×${PARALLEL_NUM_ENVS}"
  echo "[teacher-enhance] residual α ${RESIDUAL_WEIGHT}→${RESIDUAL_WEIGHT_FINAL} ep=${RESIDUAL_EPISODES} min_prior_succ=${MIN_PRIOR_SUCCESSES}"
} | tee -a "$LOG"

[[ -f "$PRIOR_PT" ]] || { echo "missing prior $PRIOR_PT" >&2; exit 1; }
[[ -f "$PI0" ]] || { echo "missing pi0 $PI0" >&2; exit 1; }
[[ -f "${DAGGER_DIR}/dagger_validation.json" ]] || { echo "missing dagger_validation.json" >&2; exit 1; }

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set -euo pipefail
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

echo "[teacher-enhance] $(date -Is) stage3 residual-MPC A6∥MAPPO α:0→0.25 parallel=${PARALLEL_TRAIN}" | tee -a "$LOG"
NUM_ENVS="${NUM_ENVS}" \
TEACHER=mpc \
RESIDUAL_BASE=mpc \
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT}" \
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL}" \
EPISODES="${RESIDUAL_EPISODES}" \
BUDGET_CHECKPOINTS="50,100,200" \
VAL_EPISODES=40 \
VAL_INTERVAL=50 \
NAME_SUFFIX="${NAME_SUFFIX}" \
PARALLEL_TRAIN="${PARALLEL_TRAIN}" \
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS}" \
MIN_PRIOR_EPISODES=200 \
MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES}" \
SKIP_MATD3=0 \
SKIP_MAPPO=0 \
  bash "${ROOT}/run_physx_closed_train.sh" 2>&1 | tee -a "$LOG"

eval_one() {
  local method="$1" ckpt="$2" name="$3"
  shift 3
  local extra=("$@")
  local out="${TRAIN_ROOT}/isaac_dual_eval_teacher_enhance/${name}"
  mkdir -p "$out"
  echo "[teacher-enhance] $(date -Is) eval ${name}" | tee -a "$LOG"
  local args=(
    -u "${ROOT}/evaluate_isaac_pursuit.py"
    --method "$method"
    --levels nominal medium hard severe extreme
    --no-skip-existing-levels
    --physx-closed-loop
    --no-native-lidar
    --num-envs "${NUM_ENVS}"
    --seed-start "${SEED_START}"
    --seed-count "${SEED_COUNT}"
    --log-every 10
    --out "$out"
  )
  if [[ -n "$ckpt" ]]; then
    args+=(--checkpoint "$ckpt")
  fi
  args+=("${extra[@]}")
  python "${args[@]}" 2>&1 | tee -a "$LOG"
}

mkdir -p "${TRAIN_ROOT}/isaac_dual_eval_teacher_enhance"
echo "[teacher-enhance] $(date -Is) stage4 locked hard eval (baselines + π0 + residual)" | tee -a "$LOG"

eval_one apf "" baseline_apf
eval_one frpn "" baseline_frpn
eval_one mpc "" baseline_mpc
eval_one mappo "$PI0" pi0_dagger_frozen

RES_A6="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_${TAG}_seed11${NAME_SUFFIX}"
RES_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11${NAME_SUFFIX}"
if [[ -f "${RES_A6}/best_capture.pt" ]]; then
  eval_one matd3 "${RES_A6}/best_capture.pt" residual_a6_best \
    --residual-base mpc --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi
if [[ -f "${RES_MAPPO}/best_capture.pt" ]]; then
  eval_one mappo "${RES_MAPPO}/best_capture.pt" residual_mappo_best \
    --residual-base mpc --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi

ARCH_A6="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_${TAG}_seed11/best_capture.pt"
ARCH_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11/best_capture.pt"
if [[ -f "$ARCH_A6" ]]; then
  eval_one matd3 "$ARCH_A6" pure_rl_a6_best_archive
fi
if [[ -f "$ARCH_MAPPO" ]]; then
  eval_one mappo "$ARCH_MAPPO" pure_rl_mappo_best_archive
fi

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${TRAIN_ROOT}/isaac_dual_eval_teacher_enhance")
print("=== teacher-enhance locked hard snapshot ===")
for p in sorted(root.glob("*/hard.json")):
    s = json.loads(p.read_text())["summary"]
    key = next(iter(s))
    row = s[key]
    print(f"{p.parent.name:28s} capture={row.get('capture_rate')} steps={row.get('mean_censored_steps')}")
print("wrote under", root)
PY
echo "[teacher-enhance] $(date -Is) train+eval complete" | tee -a "$LOG"
