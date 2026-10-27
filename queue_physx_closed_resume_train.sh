#!/usr/bin/env bash
# Resume physx_closed train after a mid-run validation policy change.
# Continues MATD3-A6 from latest.pt, then MAPPO+, then ladder eval.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_physx_closed"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/mpc_closed_resume.log"

if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
NUM_ENVS="${NUM_ENVS:-10}"
VAL_EPISODES="${VAL_EPISODES:-20}"
VAL_INTERVAL="${VAL_INTERVAL:-250}"
TEACHER="${TEACHER:-mpc}"
RESUME_CKPT="${RESUME_CKPT:-${RUN_ROOT}/train/isaac_dual_A6_${TEACHER}_hard_physx_closed_seed11/latest.pt}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

{
  echo "[mpc-closed-resume] $(date -Is) resume A6 from ${RESUME_CKPT}"
  echo "[mpc-closed-resume] NUM_ENVS=${NUM_ENVS} VAL_EPISODES=${VAL_EPISODES} VAL_INTERVAL=${VAL_INTERVAL}"
} | tee -a "$LOG"

NUM_ENVS="${NUM_ENVS}" \
TEACHER="${TEACHER}" \
VAL_EPISODES="${VAL_EPISODES}" \
VAL_INTERVAL="${VAL_INTERVAL}" \
RESUME_CKPT="${RESUME_CKPT}" \
  bash "${ROOT}/run_physx_closed_train.sh" >>"$LOG" 2>&1

echo "[mpc-closed-resume] $(date -Is) eval NUM_ENVS=${NUM_ENVS}" | tee -a "$LOG"
NUM_ENVS="${NUM_ENVS}" TEACHER="${TEACHER}" SKIP_EXISTING_LEVELS=0 \
  bash "${ROOT}/run_physx_closed_eval.sh" >>"$LOG" 2>&1

echo "[mpc-closed-resume] $(date -Is) pipeline complete" | tee -a "$LOG"
