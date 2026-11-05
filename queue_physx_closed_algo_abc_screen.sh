#!/usr/bin/env bash
# Algorithm screen on the same MPC DAgger π0 (no main reward retune):
#   A = prior curriculum α:0→1 result (reference only; not retrained here)
#   B = strong O2O anchor (higher reference_kl + demo floor), α:0→1, eval pure RL
#   C = residual deploy α:0→0.8, eval WITH MPC blend @0.8
# Short train 300 ep + locked hard n=100 for fast screening.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TEACHER=mpc
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR" "$TRAIN_ROOT"
LOG="${LOG_DIR}/algo_abc_screen.nohup"

EPISODES="${EPISODES:-300}"
NUM_ENVS="${NUM_ENVS:-8}"
VAL_EPISODES="${VAL_EPISODES:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-50,100,200,300}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
LEVELS=(hard)
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
ACTOR_INIT="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/best_imitation.pt"
DEMOS="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/expert_episodes.pt"
AUX_DEMOS="${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/correction_episodes.pt"
EVAL_OUT="${TRAIN_ROOT}/isaac_dual_eval_algo_abc_screen"

{
  echo "[algo-abc] $(date -Is) start episodes=${EPISODES} num_envs=${NUM_ENVS} hard_n=${SEED_COUNT}"
  echo "[algo-abc] A=reference prior pure_curriculum_a0to1 hard@200"
  echo "[algo-abc] B=strong_anchor α:0→1 pure eval"
  echo "[algo-abc] C=residual α:0→0.8 blend eval"
} | tee -a "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; do
  echo "[algo-abc] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
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

for f in "$ACTOR_INIT" "$DEMOS" "$AUX_DEMOS" "${DATA_ROOT}/dagger_${TEACHER}_hard_seed11/dagger_validation.json"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

train_mappo() {
  local name="$1"; shift
  local out="${TRAIN_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  mkdir -p "$out"
  : >"$log"
  echo "[algo-abc] $(date -Is) TRAIN ${name}" | tee -a "$LOG" "$log"
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
    --out "$out" \
    "$@" >>"$log" 2>&1
  then
    echo "[algo-abc] $(date -Is) FAILED ${name}" | tee -a "$LOG" "$log"
    exit 1
  fi
  echo "[algo-abc] $(date -Is) done ${name}" | tee -a "$LOG" "$log"
}

pick_ckpt() {
  local dir="$1" c
  for c in final.pt latest.pt best_capture.pt; do
    [[ -f "${dir}/${c}" ]] && { echo "${dir}/${c}"; return 0; }
  done
  return 1
}

eval_hard() {
  local method="$1" ckpt="$2" name="$3"; shift 3
  local out="${EVAL_OUT}/${name}"
  mkdir -p "$out"
  echo "[algo-abc] $(date -Is) EVAL ${name} $*" | tee -a "$LOG"
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
  python "${args[@]}" "$@" 2>&1 | tee -a "$LOG"
  echo "[algo-abc] $(date -Is) DONE ${name}" | tee -a "$LOG"
}

DIR_B="${TRAIN_ROOT}/isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11_algo_strong_anchor_ep${EPISODES}"
DIR_C="${TRAIN_ROOT}/isaac_dual_mappo_plus_${TEACHER}_hard_${TAG}_seed11_algo_residual_a08_ep${EPISODES}"

# B: strong anchor, still anneal to pure RL at deploy
train_mappo "$(basename "$DIR_B")" \
  --residual-base mpc --residual-mode blend \
  --residual-weight 0.0 --residual-weight-final 1.0 \
  --validation-pure-rl \
  --reference-kl-weight 0.35 \
  --demo-weight 0.55 \
  --demo-weight-floor 0.40

# C: stop short of pure RL; deploy with blend
train_mappo "$(basename "$DIR_C")" \
  --residual-base mpc --residual-mode blend \
  --residual-weight 0.0 --residual-weight-final 0.8 \
  --no-validation-pure-rl \
  --reference-kl-weight 0.25 \
  --demo-weight 0.50 \
  --demo-weight-floor 0.30

mkdir -p "$EVAL_OUT"
eval_hard mpc "" baseline_mpc_hard_n${SEED_COUNT}
CKPT_B="$(pick_ckpt "$DIR_B")"
eval_hard mappo "$CKPT_B" arm_B_strong_anchor_pure
CKPT_C="$(pick_ckpt "$DIR_C")"
eval_hard mappo "$CKPT_C" arm_C_residual_a08_blend \
  --residual-base mpc --residual-mode blend --residual-weight 0.8

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${EVAL_OUT}")
gate = 0.505
print("=== algo ABC hard screen n=${SEED_COUNT} ===")
# Prior A reference (curriculum @200) if present
prior = Path("${TRAIN_ROOT}/isaac_dual_eval_pure_rl_beat_mpc_pure_curriculum_a0to1/pure_rl_mappo_final/hard.json")
rows = {}
if prior.exists():
    a = float(json.loads(prior.read_text())["summary"]["capture_rate"])
    rows["A_prior_curriculum_pure@200"] = a
    print(f"{'A_prior_curriculum_pure@200':40s} {a:8.3f}")
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    p = d / "hard.json"
    if not p.exists():
        continue
    r = float(json.loads(p.read_text())["summary"]["capture_rate"])
    rows[d.name] = r
    print(f"{d.name:40s} {r:8.3f}")
mpc = rows.get("baseline_mpc_hard_n${SEED_COUNT}")
print("---")
if mpc is not None:
    print(f"MPC hard={mpc:.3f} (ref gate={gate:.3f})")
for key, label in (
    ("A_prior_curriculum_pure@200", "A prior"),
    ("arm_B_strong_anchor_pure", "B strong-anchor pure"),
    ("arm_C_residual_a08_blend", "C residual@0.8 blend"),
):
    if key not in rows:
        continue
    r = rows[key]
    print(f"{label}: {r:.3f} {'PASS' if r >= gate else 'FAIL'} vs gate {gate:.3f}")
PY

echo "[algo-abc] $(date -Is) complete" | tee -a "$LOG"
