#!/usr/bin/env bash
# Wait for GPU, live-record MAPPO physx_closed seed=7200000, then GUI xwd side(+top).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
OUT_DIR=/home/wb/runs/isaac_dual_logs
SEED=7200000
CKPT="${CKPT:-/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_mappo_plus_mpc_hard_physx_closed_seed11_pure_curriculum_a0to1/final.pt}"
ENV_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json"
STEM="physx_closed_mappo_seed${SEED}"
TRAJ="${OUT_DIR}/${STEM}.npz"
SIDE_MP4="${OUT_DIR}/${STEM}_isaac_side.mp4"
TOP_MP4="${OUT_DIR}/${STEM}_isaac_top.mp4"
LOG="${OUT_DIR}/record_grab_${STEM}.log"
GRAB="${ROOT}/queue_grab_physx_closed_side.sh"

mkdir -p "$OUT_DIR"
{
  echo "[mappo720] $(date -Is) ckpt=${CKPT}"
  echo "[mappo720] will write traj=${TRAJ} side=${SIDE_MP4} top=${TOP_MP4}"
} | tee "$LOG"

wait_gpu() {
  while true; do
    # Free if no foreign heavy python eval / our Isaac trainers.
    if pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|replay_isaac_pursuit_traj.py|export_pursuit_traj' >/dev/null 2>&1; then
      echo "[mappo720] $(date -Is) our Isaac busy; sleep 30s" | tee -a "$LOG"
      sleep 30
      continue
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    # Need ~6–8 GiB headroom for Isaac GUI replay; wait until < 4 GiB used.
    if [[ "${used:-99999}" -lt 4000 ]]; then
      echo "[mappo720] $(date -Is) GPU free enough used_mib=${used}" | tee -a "$LOG"
      break
    fi
    echo "[mappo720] $(date -Is) GPU used_mib=${used}; waiting" | tee -a "$LOG"
    sleep 45
  done
}

wait_gpu

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

echo "[mappo720] $(date -Is) live record MAPPO seed=${SEED}" | tee -a "$LOG"
set +e
python -u "${ROOT}/record_isaac_pursuit_video.py" \
  --method mappo \
  --checkpoint "$CKPT" \
  --env-config "$ENV_CFG" \
  --physx-closed-loop \
  --seed "$SEED" \
  --max-steps 300 \
  --fps 12 \
  --width 1440 --height 900 \
  --skip-live-rgb \
  --out "${OUT_DIR}/${STEM}_live_placeholder.mp4" \
  --traj-out "$TRAJ" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ ! -f "$TRAJ" ]]; then
  echo "[mappo720] FAILED no traj rc=${rc}" | tee -a "$LOG"
  exit 1
fi
python - <<PY
import json,numpy as np
m=json.loads(str(np.load("$TRAJ",allow_pickle=False)["meta"]))
print(f"[mappo720] traj meta seed={m.get('seed')} cap={m.get('captured')} steps={m.get('steps')} method={m.get('method')}", flush=True)
PY

wait_gpu

echo "[mappo720] $(date -Is) GUI side grab" | tee -a "$LOG"
TRAJ="$TRAJ" STEM="${STEM}_isaac_side" CAMERA_VIEW=side SPEED=2.5 \
  bash "$GRAB" 2>&1 | tee -a "$LOG"

echo "[mappo720] $(date -Is) GUI top grab" | tee -a "$LOG"
TRAJ="$TRAJ" STEM="${STEM}_isaac_top" CAMERA_VIEW=top SPEED=2.5 \
  bash "$GRAB" 2>&1 | tee -a "$LOG"

# Re-grab prior MPC seed7200000 with restored light translucent buildings.
MPC_TRAJ="${OUT_DIR}/physx_closed_mpc_seed7200000_isaac_side.npz"
if [[ -f "$MPC_TRAJ" ]]; then
  echo "[mappo720] $(date -Is) re-grab MPC side/top with light buildings" | tee -a "$LOG"
  TRAJ="$MPC_TRAJ" STEM="physx_closed_mpc_seed7200000_isaac_side" CAMERA_VIEW=side SPEED=2.5 \
    bash "$GRAB" 2>&1 | tee -a "$LOG"
  TRAJ="$MPC_TRAJ" STEM="physx_closed_mpc_seed7200000_isaac_top" CAMERA_VIEW=top SPEED=2.5 \
    bash "$GRAB" 2>&1 | tee -a "$LOG"
fi

echo "[mappo720] $(date -Is) done" | tee -a "$LOG"
ls -lah "$TRAJ" "$SIDE_MP4" "$TOP_MP4" \
  "${OUT_DIR}/physx_closed_mpc_seed7200000_isaac_side.mp4" \
  "${OUT_DIR}/physx_closed_mpc_seed7200000_isaac_top.mp4" \
  2>&1 | tee -a "$LOG"
