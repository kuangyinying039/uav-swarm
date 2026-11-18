#!/usr/bin/env bash
# 1) MPC seed7200000 top-down GUI xwd MP4
# 2) MAPPO seed7000110 capture side GUI xwd MP4 (same visual recipe)
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
OUT_DIR=/home/wb/runs/isaac_dual_logs
GRAB="${ROOT}/queue_grab_physx_closed_side.sh"
LOG="${OUT_DIR}/grab_mpc_top_then_mappo.log"
{
  echo "[queue] $(date -Is) start mpc-top then mappo-side"
} | tee "$LOG"

echo "[queue] === MPC top ===" | tee -a "$LOG"
TRAJ="${OUT_DIR}/physx_closed_mpc_seed7200000_isaac_side.npz" \
STEM="physx_closed_mpc_seed7200000_isaac_top" \
CAMERA_VIEW=top \
SPEED=2.5 \
  bash "$GRAB" 2>&1 | tee -a "$LOG"

echo "[queue] === MAPPO side capture seed=7000110 ===" | tee -a "$LOG"
TRAJ="${OUT_DIR}/mappo_physx_closed_capture_7000110.npz" \
STEM="mappo_physx_closed_capture_7000110_isaac_side" \
CAMERA_VIEW=side \
SPEED=2.0 \
  bash "$GRAB" 2>&1 | tee -a "$LOG"

echo "[queue] $(date -Is) done" | tee -a "$LOG"
ls -lah \
  "${OUT_DIR}/physx_closed_mpc_seed7200000_isaac_top.mp4" \
  "${OUT_DIR}/mappo_physx_closed_capture_7000110_isaac_side.mp4" \
  | tee -a "$LOG"
