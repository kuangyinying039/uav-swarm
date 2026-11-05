#!/usr/bin/env bash
# Pure RL from scratch (NO BC / DAgger / prior / residual): 600 episodes + locked eval.
# Answers whether reward shaping alone can learn capture without imitation warm-start.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/pure_rl_scratch_ep600.nohup"

EPISODES="${EPISODES:-600}"
VAL_EPISODES="${VAL_EPISODES:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-50,100,200,400,600}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_NUM_ENVS="${PARALLEL_NUM_ENVS:-5}"
NUM_ENVS="${NUM_ENVS:-10}"
NAME_SUFFIX="${NAME_SUFFIX:-_pure_scratch_ep600}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-200}"
MPC_HARD_GATE="${MPC_HARD_GATE:-0.505}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_MATD3="${SKIP_MATD3:-0}"
SKIP_MAPPO="${SKIP_MAPPO:-0}"
EVAL_OUT="${TRAIN_ROOT}/isaac_dual_eval_pure_rl_scratch${NAME_SUFFIX}"
LEVELS=(nominal medium hard severe extreme)
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"

{
  echo "[scratch-rl] $(date -Is) start"
  echo "[scratch-rl] from_scratch=1 residual=none episodes=${EPISODES} val=${VAL_EPISODES}"
  echo "[scratch-rl] parallel=${PARALLEL_TRAIN}×${PARALLEL_NUM_ENVS} suffix=${NAME_SUFFIX}"
  echo "[scratch-rl] locked n=${SEED_COUNT} gate hard>=${MPC_HARD_GATE}"
} | tee -a "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|pretrain_isaac_pursuit_actor.py|collect_isaac_pursuit_data.py' >/dev/null 2>&1; do
  echo "[scratch-rl] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
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

run_one() {
  local algo="$1" variant="$2" name="$3"
  local out="${TRAIN_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  mkdir -p "$out"
  : >"$log"
  echo "[scratch-rl] $(date -Is) start ${name} (${algo}/${variant}) num_envs=${PARALLEL_NUM_ENVS}" | tee -a "$LOG" "$log"
  local args=(
    -u "${ROOT}/train_isaac_pursuit.py" "$algo" --variant "$variant"
    --from-scratch
    --num-envs "${PARALLEL_NUM_ENVS}"
    --env-config "$ENV_CFG"
    --physx-closed-loop
    --no-native-lidar
    --episodes "${EPISODES}"
    --seed 11
    --validation-episodes "${VAL_EPISODES}"
    --validation-interval "${VAL_INTERVAL}"
    --budget-checkpoints "${BUDGET_CHECKPOINTS}"
    --batch-size 256
    --hidden-dim 128
    --out "$out"
  )
  if ! python "${args[@]}" >>"$log" 2>&1; then
    echo "[scratch-rl] $(date -Is) FAILED ${name}" | tee -a "$LOG" "$log"
    return 1
  fi
  if [[ ! -f "${out}/final.pt" && ! -f "${out}/best_capture.pt" ]]; then
    echo "[scratch-rl] $(date -Is) FAILED ${name}: no checkpoint" | tee -a "$LOG" "$log"
    return 1
  fi
  echo "[scratch-rl] $(date -Is) done ${name}" | tee -a "$LOG" "$log"
}

DIR_A6="${TRAIN_ROOT}/isaac_dual_A6_scratch_hard_${TAG}_seed11${NAME_SUFFIX}"
DIR_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_scratch_hard_${TAG}_seed11${NAME_SUFFIX}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ "${PARALLEL_TRAIN}" == "1" && "${SKIP_MATD3}" != "1" && "${SKIP_MAPPO}" != "1" ]]; then
    run_one matd3 A6 "$(basename "$DIR_A6")" &
    pid_a6=$!
    run_one mappo mappo_plus "$(basename "$DIR_MAPPO")" &
    pid_mappo=$!
    status=0
    wait "$pid_a6" || status=1
    wait "$pid_mappo" || status=1
    if [[ "$status" -ne 0 ]]; then
      echo "[scratch-rl] $(date -Is) train failed" | tee -a "$LOG"
      exit 1
    fi
  else
    if [[ "${SKIP_MATD3}" != "1" ]]; then
      run_one matd3 A6 "$(basename "$DIR_A6")"
    fi
    if [[ "${SKIP_MAPPO}" != "1" ]]; then
      run_one mappo mappo_plus "$(basename "$DIR_MAPPO")"
    fi
  fi
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[scratch-rl] $(date -Is) eval skipped" | tee -a "$LOG"
  exit 0
fi

mkdir -p "$EVAL_OUT"

eval_pure() {
  local method="$1" ckpt="$2" name="$3"
  local out="${EVAL_OUT}/${name}"
  mkdir -p "$out"
  echo "[scratch-rl] $(date -Is) EVAL ${name}" | tee -a "$LOG"
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
  echo "[scratch-rl] $(date -Is) DONE ${name}" | tee -a "$LOG"
}

pick_ckpt() {
  local dir="$1" c
  for c in final.pt latest.pt best_capture.pt; do
    if [[ -f "${dir}/${c}" ]]; then
      echo "${dir}/${c}"
      return 0
    fi
  done
  return 1
}

echo "[scratch-rl] $(date -Is) stage2 locked ladder n=${SEED_COUNT}" | tee -a "$LOG"
eval_pure mpc "" baseline_mpc
if CKPT_A6="$(pick_ckpt "$DIR_A6")"; then
  echo "[scratch-rl] A6 ckpt=${CKPT_A6}" | tee -a "$LOG"
  eval_pure matd3 "$CKPT_A6" pure_rl_a6_scratch
fi
if CKPT_MAPPO="$(pick_ckpt "$DIR_MAPPO")"; then
  echo "[scratch-rl] MAPPO ckpt=${CKPT_MAPPO}" | tee -a "$LOG"
  eval_pure mappo "$CKPT_MAPPO" pure_rl_mappo_scratch
fi

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${EVAL_OUT}")
levels = ["nominal", "medium", "hard", "severe", "extreme"]
gate = float("${MPC_HARD_GATE}")
print("=== pure RL from scratch locked @${SEED_COUNT} ===")
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
print("--- gate ---")
mpc_hard = rows.get("baseline_mpc")
if mpc_hard is not None:
    print(f"MPC hard={mpc_hard:.3f} (ref gate={gate:.3f})")
for key, label in (
    ("pure_rl_mappo_scratch", "MAPPO scratch"),
    ("pure_rl_a6_scratch", "A6 scratch"),
):
    hard = rows.get(key)
    if hard is None:
        continue
    print(f"{label} hard={hard:.3f} {'PASS' if hard >= gate else 'FAIL'} vs gate {gate:.3f}")
PY

echo "[scratch-rl] $(date -Is) complete" | tee -a "$LOG"
