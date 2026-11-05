#!/usr/bin/env bash
# FRPN-teacher enhance: collect → DAgger π0 → residual blend A6∥MAPPO → locked ladder.
# Use when locked hard shows FRPN ≥ MPC; residual base = FRPN so RL starts from the
# stronger classical floor. MAPPO/A6 both use blend (full-action DAgger compatible).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER=frpn
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/frpn_teacher_enhance_pipeline.log"

NUM_ENVS="${NUM_ENVS:-10}"
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
PRIOR_PT="${DATA_ROOT}/hard_${TEACHER}_prior.pt"
DAGGER_DIR="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11"
NEAR_MISS_GAP="${NEAR_MISS_GAP:-1.5}"
COLLECT_ROLLOUTS="${COLLECT_ROLLOUTS:-240}"
COLLECT_MIN_SUCCESS="${COLLECT_MIN_SUCCESS:-200}"
COLLECT_MAX_ATTEMPTS="${COLLECT_MAX_ATTEMPTS:-900}"
RESIDUAL_EPISODES="${RESIDUAL_EPISODES:-400}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.0}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.35}"
RESIDUAL_MODE="${RESIDUAL_MODE:-blend}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-5}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
SKIP_DAGGER="${SKIP_DAGGER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
NAME_SUFFIX="_residual_${TEACHER}_blend_ep${RESIDUAL_EPISODES}"

{
  echo "[frpn-enhance] $(date -Is) start teacher=${TEACHER} residual=${RESIDUAL_MODE} α:${RESIDUAL_WEIGHT}→${RESIDUAL_WEIGHT_FINAL}"
  echo "[frpn-enhance] ep=${RESIDUAL_EPISODES} parallel=${PARALLEL_TRAIN}×${PARALLEL_NUM_ENVS}"
} | tee -a "$LOG"

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

# ---- Stage 1: collect FRPN prior ----
if [[ "${FORCE_COLLECT}" == "1" || ! -f "$PRIOR_PT" ]]; then
  echo "[frpn-enhance] $(date -Is) stage1 collect FRPN prior" | tee -a "$LOG"
  collect_args=(
    --env-config "$ENV_CFG"
    --teacher "${TEACHER}"
    --demo-seed 2000000
    --rollouts "${COLLECT_ROLLOUTS}"
    --max-attempts "${COLLECT_MAX_ATTEMPTS}"
    --min-successes "${COLLECT_MIN_SUCCESS}"
    --keep-near-misses
    --near-miss-gap "${NEAR_MISS_GAP}"
    --no-keep-failures
    --physx-closed-loop
    --no-native-lidar
    --dataset "$PRIOR_PT"
  )
  if [[ "${FORCE_COLLECT}" == "1" ]]; then
    collect_args+=(--force)
  fi
  python -u "${ROOT}/collect_isaac_pursuit_data.py" "${collect_args[@]}" 2>&1 | tee -a "$LOG"
else
  echo "[frpn-enhance] $(date -Is) stage1 reuse ${PRIOR_PT}" | tee -a "$LOG"
fi
[[ -f "$PRIOR_PT" ]] || { echo "missing prior $PRIOR_PT" >&2; exit 1; }

# ---- Stage 2: DAgger ----
PI0="${DAGGER_DIR}/pi0_frozen.pt"
if [[ "${SKIP_DAGGER}" == "1" && -f "$PI0" && -f "${DAGGER_DIR}/dagger_validation.json" ]]; then
  echo "[frpn-enhance] $(date -Is) stage2 skip (reuse ${PI0})" | tee -a "$LOG"
else
  echo "[frpn-enhance] $(date -Is) stage2 DAgger on FRPN" | tee -a "$LOG"
  rm -rf "$DAGGER_DIR"
  python -u "${ROOT}/pretrain_isaac_pursuit_actor.py" \
    --env-config "$ENV_CFG" \
    --teacher "${TEACHER}" \
    --seed 11 --demo-seed 2100000 \
    --bc-episodes 80 --bc-attempts 320 --bc-updates 2000 \
    --success-only-bc \
    --dagger-rounds 3 --dagger-episodes 40 --dagger-updates 800 \
    --dagger-validation-episodes 40 --dagger-validation-seed 5000000 \
    --physx-closed-loop \
    --no-native-lidar \
    --out "$DAGGER_DIR" \
    2>&1 | tee -a "$LOG"
  cp -a "${DAGGER_DIR}/best_imitation.pt" "$PI0"
fi
[[ -f "$PI0" ]] || { echo "missing pi0 $PI0" >&2; exit 1; }

# ---- Stage 3: residual blend train ----
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[frpn-enhance] $(date -Is) stage3 residual-${TEACHER} blend A6∥MAPPO" | tee -a "$LOG"
  NUM_ENVS="${NUM_ENVS}" \
  TEACHER="${TEACHER}" \
  RESIDUAL_BASE="${TEACHER}" \
  RESIDUAL_MODE="${RESIDUAL_MODE}" \
  RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT}" \
  RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL}" \
  EPISODES="${RESIDUAL_EPISODES}" \
  BUDGET_CHECKPOINTS="50,100,200,400" \
  VAL_EPISODES=40 \
  VAL_INTERVAL=50 \
  NAME_SUFFIX="${NAME_SUFFIX}" \
  PARALLEL_TRAIN="${PARALLEL_TRAIN}" \
  PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS}" \
  MIN_PRIOR_EPISODES=200 \
  MIN_PRIOR_SUCCESSES="${COLLECT_MIN_SUCCESS}" \
  SKIP_MATD3=0 \
  SKIP_MAPPO=0 \
    bash "${ROOT}/run_physx_closed_train.sh" 2>&1 | tee -a "$LOG"
fi

# ---- Stage 4: locked ladder eval ----
if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[frpn-enhance] $(date -Is) stage4 skipped" | tee -a "$LOG"
  exit 0
fi

EVAL_OUT="${TRAIN_ROOT}/isaac_dual_eval_frpn_teacher_enhance"
LEVELS=(nominal medium hard severe extreme)
mkdir -p "$EVAL_OUT"

eval_one() {
  local method="$1" ckpt="$2" name="$3"
  shift 3
  local extra=("$@")
  local out="${EVAL_OUT}/${name}"
  mkdir -p "$out"
  echo "[frpn-enhance] $(date -Is) eval ${name}" | tee -a "$LOG"
  local args=(
    -u "${ROOT}/evaluate_isaac_pursuit.py"
    --method "$method"
    --levels "${LEVELS[@]}"
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

echo "[frpn-enhance] $(date -Is) stage4 locked ladder" | tee -a "$LOG"
eval_one apf "" baseline_apf
eval_one frpn "" baseline_frpn
eval_one mpc "" baseline_mpc
eval_one mappo "$PI0" pi0_dagger_frozen

RES_A6="${TRAIN_ROOT}/isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}/best_capture.pt"
RES_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}/best_capture.pt"
if [[ -f "$RES_A6" ]]; then
  eval_one matd3 "$RES_A6" residual_a6_frpn_blend \
    --residual-base frpn --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi
if [[ -f "$RES_MAPPO" ]]; then
  eval_one mappo "$RES_MAPPO" residual_mappo_frpn_blend \
    --residual-base frpn --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_frpn_teacher_enhance")
levels = ["nominal", "medium", "hard", "severe", "extreme"]
print("=== frpn-teacher locked snapshot ===")
print(f"{'method':36s}", " ".join(f"{lv:>8s}" for lv in levels))
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    rates = []
    for lv in levels:
        p = d / f"{lv}.json"
        if not p.exists():
            rates.append(f"{'—':>8s}")
            continue
        s = json.loads(p.read_text())["summary"]
        rates.append(f"{s['capture_rate']:8.3f}")
    print(f"{d.name:36s}", " ".join(rates))
PY

echo "[frpn-enhance] $(date -Is) complete" | tee -a "$LOG"
