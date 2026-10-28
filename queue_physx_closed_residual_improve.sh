#!/usr/bin/env bash
# Residual-MPC improve pipeline (does NOT overwrite pure-RL formal dirs).
#
# Order:
#   0) wait until no other Isaac train/eval is running
#   1) Eval-only: A6 best vs final, DAgger freeze, MAPPO best vs final (hard, n=100)
#   2) Main: A6 + MAPPO residual-on-MPC, 200 episodes, NAME_SUFFIX=_residual_mpc_ep200
#   3) Eval residual runs on hard (+ optional ladder if EVAL_LADDER=1)
#   4) Optional: FRPN residual A6-only 200 ep (SKIP_FRPN_RESIDUAL=0 to enable)
#
# O2O tight package is auto-applied in train_isaac_pursuit.py when residual-base is set.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/residual_improve_pipeline.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
# P0 denser val on short residual runs; formal capture uses target_diameter=0.5.
VAL_EPISODES="${VAL_EPISODES:-40}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
EPISODES="${EPISODES:-200}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-50,100,200}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.35}"
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL:-0.55}"
SKIP_FRPN_RESIDUAL="${SKIP_FRPN_RESIDUAL:-1}"
EVAL_LADDER="${EVAL_LADDER:-0}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"

{
  echo "[residual-improve] $(date -Is) start NUM_ENVS=${NUM_ENVS} EPISODES=${EPISODES}"
  echo "[residual-improve] residual mpc weight ${RESIDUAL_WEIGHT}→${RESIDUAL_WEIGHT_FINAL}"
} | tee -a "$LOG"

# Wait for exclusive Isaac use (d05 contrast / other evals).
while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|pretrain_isaac_pursuit_actor.py|collect_isaac_pursuit_data.py' >/dev/null 2>&1; do
  echo "[residual-improve] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
  sleep 60
done

echo "[residual-improve] $(date -Is) step1 eval-only best vs final vs dagger" | tee -a "$LOG"
NUM_ENVS="${NUM_ENVS}" bash "${ROOT}/scripts/eval_ckpt_compare_hard.sh" >>"$LOG" 2>&1

echo "[residual-improve] $(date -Is) step2 residual-MPC train A6+MAPPO ${EPISODES}ep" | tee -a "$LOG"
NUM_ENVS="${NUM_ENVS}" \
TEACHER=mpc \
RESIDUAL_BASE=mpc \
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT}" \
RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL}" \
EPISODES="${EPISODES}" \
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS}" \
VAL_EPISODES="${VAL_EPISODES}" \
VAL_INTERVAL="${VAL_INTERVAL}" \
NAME_SUFFIX="_residual_mpc_ep${EPISODES}" \
  bash "${ROOT}/run_physx_closed_train.sh" >>"$LOG" 2>&1

eval_residual_hard() {
  local method="$1" train_name="$2" out_name="$3" rbase="$4"
  local ckpt="${TRAIN_ROOT}/${train_name}/best_capture.pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="${TRAIN_ROOT}/${train_name}/final.pt"
  fi
  local out="${TRAIN_ROOT}/${out_name}"
  mkdir -p "$out"
  local levels=(hard)
  if [[ "${EVAL_LADDER}" == "1" ]]; then
    levels=(nominal medium hard severe extreme)
  fi
  echo "[residual-improve] $(date -Is) eval ${out_name} residual=${rbase}" | tee -a "$LOG"
  source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
  conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
  export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
  set +u
  # shellcheck disable=SC1091
  source "${ISAACSIM_PATH}/setup_conda_env.sh"
  set +u
  export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  unset DISPLAY || true
  export OMNI_KIT_ACCEPT_EULA=YES
  cd "${ROOT}"
  local mode=blend
  if [[ "$method" == "mappo" ]]; then
    mode=residual
  fi
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels "${levels[@]}" \
    --no-skip-existing-levels \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --residual-base "$rbase" \
    --residual-mode "$mode" \
    --residual-weight "${RESIDUAL_WEIGHT_FINAL}" \
    --log-every 10 \
    --out "$out" \
    >>"$LOG" 2>&1
}

echo "[residual-improve] $(date -Is) step3 eval residual-MPC hard" | tee -a "$LOG"
eval_residual_hard matd3 \
  "isaac_dual_A6_mpc_hard_${TAG}_seed11_residual_mpc_ep${EPISODES}" \
  "isaac_dual_eval_matd3_mpc_${TAG}_residual_ep${EPISODES}_seed11" \
  mpc
eval_residual_hard mappo \
  "isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11_residual_mpc_ep${EPISODES}" \
  "isaac_dual_eval_mappo_mpc_${TAG}_residual_ep${EPISODES}_seed11" \
  mpc

if [[ "${SKIP_FRPN_RESIDUAL}" != "1" ]]; then
  echo "[residual-improve] $(date -Is) step4 optional FRPN residual A6 ${EPISODES}ep" | tee -a "$LOG"
  if [[ ! -f "${RUN_ROOT}/data/dagger_frpn_hard_seed11/best_imitation.pt" ]]; then
    echo "[residual-improve] missing FRPN DAgger; skip FRPN residual" | tee -a "$LOG"
  else
    NUM_ENVS="${NUM_ENVS}" \
    TEACHER=frpn \
    RESIDUAL_BASE=frpn \
    RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT}" \
    RESIDUAL_WEIGHT_FINAL="${RESIDUAL_WEIGHT_FINAL}" \
    EPISODES="${EPISODES}" \
    BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS}" \
    VAL_EPISODES="${VAL_EPISODES}" \
    VAL_INTERVAL="${VAL_INTERVAL}" \
    NAME_SUFFIX="_residual_frpn_ep${EPISODES}" \
    SKIP_MAPPO=1 \
      bash "${ROOT}/run_physx_closed_train.sh" >>"$LOG" 2>&1
    eval_residual_hard matd3 \
      "isaac_dual_A6_frpn_hard_${TAG}_seed11_residual_frpn_ep${EPISODES}" \
      "isaac_dual_eval_matd3_frpn_${TAG}_residual_ep${EPISODES}_seed11" \
      frpn
  fi
fi

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${TRAIN_ROOT}")
print("=== residual improve snapshot ===")
pairs = [
    ("compare/a6_best", root/"isaac_dual_eval_ckpt_compare_hard/a6_mpc_best_capture/hard.json"),
    ("compare/a6_final", root/"isaac_dual_eval_ckpt_compare_hard/a6_mpc_final/hard.json"),
    ("compare/dagger", root/"isaac_dual_eval_ckpt_compare_hard/dagger_mpc_best_imitation/hard.json"),
    ("compare/mappo_best", root/"isaac_dual_eval_ckpt_compare_hard/mappo_mpc_best_capture/hard.json"),
    ("compare/mappo_final", root/"isaac_dual_eval_ckpt_compare_hard/mappo_mpc_final/hard.json"),
    ("residual/a6", root/f"isaac_dual_eval_matd3_mpc_${TAG}_residual_ep${EPISODES}_seed11/hard.json"),
    ("residual/mappo", root/f"isaac_dual_eval_mappo_mpc_${TAG}_residual_ep${EPISODES}_seed11/hard.json"),
]
for label, path in pairs:
    if not path.exists():
        print(f"{label:22s} MISSING {path}")
        continue
    s = json.loads(path.read_text())["summary"]
    print(f"{label:22s} rate={s['capture_rate']:.3f} steps={s['mean_censored_steps']:.1f} ret={s['mean_return']:.1f}")
PY

echo "[residual-improve] $(date -Is) pipeline complete" | tee -a "$LOG"
