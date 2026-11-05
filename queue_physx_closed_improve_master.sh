#!/usr/bin/env bash
# Master queue after residual parallel-env fix:
#   1) finish (or wait for) residual α diagnostic
#   2) full locked ladder eval of existing ckpts (fixed compose + MAPPO blend)
#   3) FRPN-teacher collect → DAgger → residual blend retrain → ladder eval
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
RUN=/srv/data/projects/kyy/runs/isaac_dual_physx_closed
LOG="${RUN}/logs/improve_master.nohup"
mkdir -p "${RUN}/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[master] $(date -Is) improve queue start"

wait_isaac_idle() {
  local reason="$1"
  while pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py|residual_diag_fixed' >/dev/null 2>&1; do
    echo "[master] $(date -Is) waiting (${reason})..."
    sleep 60
  done
}

# 1) Let current residual diag finish (w=0 already confirmed 0.525)
if pgrep -f 'residual_diag_fixed.sh' >/dev/null 2>&1; then
  echo "[master] $(date -Is) waiting for residual_diag_fixed to finish"
  wait_isaac_idle "residual diag"
  echo "[master] diag log RESULT lines:"
  rg -n '^RESULT ' "${RUN}/logs/residual_diag.nohup" || true
else
  echo "[master] residual diag not running; continue"
fi

# 2) Full locked eval (5 levels) with fixed residual wrap
echo "[master] $(date -Is) === stage: full locked eval ==="
chmod +x "${ROOT}/queue_physx_closed_full_locked_eval.sh"
NUM_ENVS=10 MAPPO_RESIDUAL_MODE=blend RESIDUAL_WEIGHT=0.25 \
  bash "${ROOT}/queue_physx_closed_full_locked_eval.sh"

# 3) FRPN teacher enhance (stronger hard floor)
echo "[master] $(date -Is) === stage: FRPN teacher enhance ==="
chmod +x "${ROOT}/queue_physx_closed_frpn_teacher_enhance.sh"
NUM_ENVS=10 PARALLEL_TRAIN=1 PARALLEL_NUM_ENVS=5 \
RESIDUAL_EPISODES=400 RESIDUAL_WEIGHT=0.0 RESIDUAL_WEIGHT_FINAL=0.35 \
RESIDUAL_MODE=blend FORCE_COLLECT=1 \
  bash "${ROOT}/queue_physx_closed_frpn_teacher_enhance.sh"

echo "[master] $(date -Is) ALL COMPLETE"
