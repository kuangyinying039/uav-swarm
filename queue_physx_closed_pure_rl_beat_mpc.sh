#!/usr/bin/env bash
# Pure-RL beat MPC: α curriculum 0→1, then one locked ladder eval (no mid-train val).
#
# Train: a=(1-α)a_MPC + α a_RL with α:0→1 (blend). Default VAL_EPISODES=0 so
# training does not pause for held-out capture tests; budget ckpts are kept.
# Eval: locked five-level ladder n=200, NO residual (deployable pure policy),
# preferring final/latest (then best_capture if present).
# Gate: MAPPO hard ≥ MPC 0.505. MATD3-A6 is parallel对照; MAPPO is primary.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER=mpc
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/pure_rl_beat_mpc_pipeline.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
EPISODES="${EPISODES:-600}"
# 0 = no mid-train held-out eval (recommended). Set >0 only if you need online best_capture.
VAL_EPISODES="${VAL_EPISODES:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-50,100,200,400,600}"
RESIDUAL_BASE=mpc
RESIDUAL_MODE=blend
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.0}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-1.0}"
VALIDATION_PURE_RL="${VALIDATION_PURE_RL:-1}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_MATD3="${SKIP_MATD3:-0}"
SKIP_MAPPO="${SKIP_MAPPO:-0}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-200}"
NAME_SUFFIX="${NAME_SUFFIX:-_pure_curriculum_a0to1}"
EVAL_OUT="${TRAIN_ROOT}/isaac_dual_eval_pure_rl_beat_mpc${NAME_SUFFIX}"
LEVELS=(nominal medium hard severe extreme)
MPC_HARD_GATE="${MPC_HARD_GATE:-0.505}"
# Existing near-miss MPC prior has 164 successes (see hard_mpc_prior.min_successes).
MIN_PRIOR_EPISODES="${MIN_PRIOR_EPISODES:-200}"
MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES:-}"
if [[ -z "${MIN_PRIOR_SUCCESSES}" && -f "${RUN_ROOT}/data/hard_mpc_prior.min_successes" ]]; then
  MIN_PRIOR_SUCCESSES="$(tr -d '[:space:]' < "${RUN_ROOT}/data/hard_mpc_prior.min_successes")"
fi
MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES:-164}"

{
  echo "[pure-rl-beat-mpc] $(date -Is) start"
  echo "[pure-rl-beat-mpc] α:${RESIDUAL_WEIGHT}→${RESIDUAL_WEIGHT_FINAL} mode=${RESIDUAL_MODE}"
  echo "[pure-rl-beat-mpc] ep=${EPISODES} mid_train_val=${VAL_EPISODES}@${VAL_INTERVAL} (0=off) pure_val_flag=${VALIDATION_PURE_RL}"
  echo "[pure-rl-beat-mpc] suffix=${NAME_SUFFIX} parallel=${PARALLEL_TRAIN}×${PARALLEL_NUM_ENVS}"
  echo "[pure-rl-beat-mpc] locked n=${SEED_COUNT} gate hard>=${MPC_HARD_GATE}"
  echo "[pure-rl-beat-mpc] min_prior ep=${MIN_PRIOR_EPISODES} succ=${MIN_PRIOR_SUCCESSES}"
} | tee -a "$LOG"

# Exclusive Isaac use.
while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|pretrain_isaac_pursuit_actor.py|collect_isaac_pursuit_data.py' >/dev/null 2>&1; do
  echo "[pure-rl-beat-mpc] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
  sleep 60
done

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

# Reuse existing MPC prior / DAgger; never overwrite formal pure-RL dirs (suffix).
require_artifact() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
}
require_artifact "${RUN_ROOT}/data/hard_mpc_prior.pt"
require_artifact "${RUN_ROOT}/data/dagger_mpc_hard_seed11/best_imitation.pt"
require_artifact "${RUN_ROOT}/data/dagger_mpc_hard_seed11/dagger_validation.json"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[pure-rl-beat-mpc] $(date -Is) stage1 train MAPPO(+A6) α curriculum" | tee -a "$LOG"
  NUM_ENVS="${NUM_ENVS}" \
  TEACHER="${TEACHER}" \
  RESIDUAL_BASE="${RESIDUAL_BASE}" \
  RESIDUAL_MODE="${RESIDUAL_MODE}" \
  RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT}" \
  RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL}" \
  VALIDATION_PURE_RL="${VALIDATION_PURE_RL}" \
  EPISODES="${EPISODES}" \
  BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS}" \
  VAL_EPISODES="${VAL_EPISODES}" \
  VAL_INTERVAL="${VAL_INTERVAL}" \
  NAME_SUFFIX="${NAME_SUFFIX}" \
  PARALLEL_TRAIN="${PARALLEL_TRAIN}" \
  PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS}" \
  SKIP_MATD3="${SKIP_MATD3}" \
  SKIP_MAPPO="${SKIP_MAPPO}" \
  MIN_PRIOR_EPISODES="${MIN_PRIOR_EPISODES}" \
  MIN_PRIOR_SUCCESSES="${MIN_PRIOR_SUCCESSES}" \
    bash "${ROOT}/run_physx_closed_train.sh" 2>&1 | tee -a "$LOG"
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[pure-rl-beat-mpc] $(date -Is) stage2 eval skipped" | tee -a "$LOG"
  exit 0
fi

mkdir -p "$EVAL_OUT"

eval_pure() {
  local method="$1" ckpt="$2" name="$3"
  local out="${EVAL_OUT}/${name}"
  mkdir -p "$out"
  echo "[pure-rl-beat-mpc] $(date -Is) EVAL ${name} (pure RL, no residual)" | tee -a "$LOG"
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
  python "${args[@]}" 2>&1 | tee -a "$LOG"
  echo "[pure-rl-beat-mpc] $(date -Is) DONE ${name}" | tee -a "$LOG"
}

pick_ckpt() {
  # Prefer end-of-train weights when mid-train val was disabled.
  local dir="$1"
  local c
  for c in final.pt latest.pt best_capture.pt; do
    if [[ -f "${dir}/${c}" ]]; then
      echo "${dir}/${c}"
      return 0
    fi
  done
  # Fall back to the highest budget episode if present.
  local best=""
  local f
  for f in "${dir}"/budgets/ep*.pt; do
    [[ -f "$f" ]] || continue
    best="$f"
  done
  if [[ -n "$best" ]]; then
    echo "$best"
    return 0
  fi
  return 1
}

echo "[pure-rl-beat-mpc] $(date -Is) stage2 locked pure-RL ladder n=${SEED_COUNT}" | tee -a "$LOG"
# Classical reference (same seeds) for the gate table.
eval_pure mpc "" baseline_mpc

DIR_A6="${TRAIN_ROOT}/isaac_dual_A6_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
DIR_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11${NAME_SUFFIX}"
if CKPT_A6="$(pick_ckpt "$DIR_A6")"; then
  echo "[pure-rl-beat-mpc] A6 ckpt=${CKPT_A6}" | tee -a "$LOG"
  eval_pure matd3 "$CKPT_A6" pure_rl_a6_final
else
  echo "[pure-rl-beat-mpc] skip A6 eval (no ckpt under ${DIR_A6})" | tee -a "$LOG"
fi
if CKPT_MAPPO="$(pick_ckpt "$DIR_MAPPO")"; then
  echo "[pure-rl-beat-mpc] MAPPO ckpt=${CKPT_MAPPO}" | tee -a "$LOG"
  eval_pure mappo "$CKPT_MAPPO" pure_rl_mappo_final
else
  echo "[pure-rl-beat-mpc] skip MAPPO eval (no ckpt under ${DIR_MAPPO})" | tee -a "$LOG"
fi

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${EVAL_OUT}")
levels = ["nominal", "medium", "hard", "severe", "extreme"]
gate = float("${MPC_HARD_GATE}")
print("=== pure RL beat MPC locked @${SEED_COUNT} ===")
print(f"{'method':36s}", " ".join(f"{lv:>8s}" for lv in levels))
rows = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    rates = []
    hard = None
    for lv in levels:
        p = d / f"{lv}.json"
        if not p.exists():
            rates.append(f"{'—':>8s}")
            continue
        s = json.loads(p.read_text())["summary"]
        rates.append(f"{s['capture_rate']:8.3f}")
        if lv == "hard":
            hard = float(s["capture_rate"])
    rows[d.name] = hard
    print(f"{d.name:36s}", " ".join(rates))
mpc_hard = rows.get("baseline_mpc")
mappo_hard = rows.get("pure_rl_mappo_final", rows.get("pure_rl_mappo_best_capture"))
a6_hard = rows.get("pure_rl_a6_final", rows.get("pure_rl_a6_best_capture"))
print("--- gate ---")
if mpc_hard is not None:
    print(f"MPC hard={mpc_hard:.3f} (ref gate={gate:.3f})")
if mappo_hard is not None:
    ok = mappo_hard >= gate
    print(f"MAPPO pure hard={mappo_hard:.3f} {'PASS' if ok else 'FAIL'} vs gate {gate:.3f}")
if a6_hard is not None:
    ok = a6_hard >= gate
    print(f"A6 pure hard={a6_hard:.3f} {'PASS' if ok else 'FAIL'} vs gate {gate:.3f}")
PY

echo "[pure-rl-beat-mpc] $(date -Is) complete" | tee -a "$LOG"
