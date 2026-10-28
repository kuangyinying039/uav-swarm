#!/usr/bin/env bash
# Teacher-enhance mainline (advisor protocol).
#
# Research claim hierarchy (not "pure RL beats MPC"):
#   1) capability retention vs π0
#   2) verifiable gain of π_best over π0 on locked test
#   3) MPC + residual improves MPC (or non-inferior capture + lower cost)
#
# Stages:
#   0) Offline O2O diagnosis
#   1) Enriched collect: ≥200 successes + near-misses
#   2) DAgger + freeze π0
#   3) Parallel residual-on-MPC: A6 + MAPPO (α: 0→0.25, num_envs each)
#   4) Locked hard eval: APF/FRPN/MPC + π0 + residual A6/MAPPO (+ archive pure RL)
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
DATA_ROOT="${RUN_ROOT}/data"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/teacher_enhance_pipeline.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
PRIOR_PT="${DATA_ROOT}/hard_mpc_prior.pt"
DAGGER_DIR="${DATA_ROOT}/dagger_mpc_hard_seed11"
NEAR_MISS_GAP="${NEAR_MISS_GAP:-1.5}"
COLLECT_ROLLOUTS="${COLLECT_ROLLOUTS:-240}"
COLLECT_MIN_SUCCESS="${COLLECT_MIN_SUCCESS:-200}"
COLLECT_MAX_ATTEMPTS="${COLLECT_MAX_ATTEMPTS:-900}"
RESIDUAL_EPISODES="${RESIDUAL_EPISODES:-200}"
# Zero-init soft residual: a=(1-α)MPC + α·π  with α: 0 → 0.25
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.0}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.25}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
# Per-job envs when A6∥MAPPO; default floor 4 (override with PARALLEL_NUM_ENVS).
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-5}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
SKIP_STAGE0="${SKIP_STAGE0:-0}"
NAME_SUFFIX="_residual_mpc_zeroinit_ep${RESIDUAL_EPISODES}"

{
  echo "[teacher-enhance] $(date -Is) start NUM_ENVS=${NUM_ENVS} PARALLEL_TRAIN=${PARALLEL_TRAIN} PARALLEL_NUM_ENVS=${PARALLEL_NUM_ENVS}"
  echo "[teacher-enhance] diameter=0.5 capture_radius=1.0 near_miss_gap=${NEAR_MISS_GAP}"
  echo "[teacher-enhance] residual α ${RESIDUAL_WEIGHT}→${RESIDUAL_WEIGHT_FINAL} ep=${RESIDUAL_EPISODES}"
} | tee -a "$LOG"

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
p=Path("/srv/data/projects/kyy/xieton/repro/configs/pursuit_v2/radar5_dual_hard_physx_closed.json")
d=json.loads(p.read_text())
assert float(d["target_diameter"])==0.5, d["target_diameter"]
print("[teacher-enhance] hard target_diameter=0.5 OK")
PY

if [[ "${SKIP_STAGE0}" != "1" ]]; then
  echo "[teacher-enhance] $(date -Is) stage0 diagnose_o2o_collapse" | tee -a "$LOG"
  python3 "${ROOT}/scripts/diagnose_o2o_collapse.py" \
    --out "${TRAIN_ROOT}/o2o_diagnosis.json" | tee -a "$LOG"
fi

while pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; do
  # Allow this script's own collect child after we launch it; only wait on *other* jobs at start.
  if pgrep -f 'train_isaac_pursuit.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; then
    echo "[teacher-enhance] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
    sleep 30
    continue
  fi
  break
done

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

# ---- Stage 1: enriched collect (resume progress unless FORCE_COLLECT=1) ----
echo "[teacher-enhance] $(date -Is) stage1 collect success+near-miss" | tee -a "$LOG"
collect_args=(
  --env-config "$ENV_CFG"
  --teacher mpc
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

# ---- Stage 2: DAgger ----
echo "[teacher-enhance] $(date -Is) stage2 DAgger (BC success-only, keep correction states)" | tee -a "$LOG"
rm -rf "$DAGGER_DIR"
python -u "${ROOT}/pretrain_isaac_pursuit_actor.py" \
  --env-config "$ENV_CFG" \
  --teacher mpc \
  --seed 11 --demo-seed 2100000 \
  --bc-episodes 80 --bc-attempts 320 --bc-updates 2000 \
  --success-only-bc \
  --dagger-rounds 3 --dagger-episodes 40 --dagger-updates 800 \
  --dagger-validation-episodes 40 --dagger-validation-seed 5000000 \
  --physx-closed-loop \
  --no-native-lidar \
  --out "$DAGGER_DIR" \
  2>&1 | tee -a "$LOG"

PI0="${DAGGER_DIR}/pi0_frozen.pt"
cp -a "${DAGGER_DIR}/best_imitation.pt" "$PI0"
echo "[teacher-enhance] froze π0 → ${PI0}" | tee -a "$LOG"

# ---- Stage 3: parallel residual A6 + MAPPO ----
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
SKIP_MATD3=0 \
SKIP_MAPPO=0 \
  bash "${ROOT}/run_physx_closed_train.sh" 2>&1 | tee -a "$LOG"

# ---- Stage 4: locked hard eval with classical baselines ----
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
    --levels hard
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

# Classical baselines
eval_one apf "" baseline_apf
eval_one frpn "" baseline_frpn
eval_one mpc "" baseline_mpc

# Frozen imitation zero-step baseline
eval_one mappo "$PI0" pi0_dagger_frozen

RES_A6="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_${TAG}_seed11${NAME_SUFFIX}"
RES_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11${NAME_SUFFIX}"
if [[ -f "${RES_A6}/best_capture.pt" ]]; then
  eval_one matd3 "${RES_A6}/best_capture.pt" residual_a6_best \
    --residual-base mpc --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi
if [[ -f "${RES_MAPPO}/best_capture.pt" ]]; then
  eval_one mappo "${RES_MAPPO}/best_capture.pt" residual_mappo_best \
    --residual-base mpc --residual-mode residual --residual-weight "${RESIDUAL_WEIGHT_FINAL}"
fi

# Archive pure-RL best as controls if present
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
    print(f"{p.parent.name:32s} rate={s['capture_rate']:.3f} steps={s['mean_censored_steps']:.1f}")
PY

echo "[teacher-enhance] $(date -Is) pipeline complete" | tee -a "$LOG"
