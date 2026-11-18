#!/usr/bin/env bash
# After GPU is free: record MPC on pure PhysX-closed, then side-view export
# matching coop_mpc_seed7200000_isaac_side.mp4 style (1440x900, side, ~12fps).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
OUT_DIR="${OUT_DIR:-/home/wb/runs/isaac_dual_logs}"
SEED="${SEED:-7200000}"
METHOD="${METHOD:-mpc}"
ENV_CFG="${ENV_CFG:-${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json}"
STEM="${STEM:-physx_closed_${METHOD}_seed${SEED}_isaac_side}"
OUT_MP4="${OUT_DIR}/${STEM}.mp4"
TRAJ="${OUT_DIR}/${STEM}.npz"
LOG="${OUT_DIR}/record_${STEM}.log"
WIDTH="${WIDTH:-1440}"
HEIGHT="${HEIGHT:-900}"
FPS="${FPS:-12}"

mkdir -p "$OUT_DIR"
{
  echo "[physx-side] $(date -Is) wait for free Isaac GPU"
  echo "[physx-side] method=${METHOD} seed=${SEED} out=${OUT_MP4}"
} | tee "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|export_pursuit_traj' >/dev/null 2>&1; do
  echo "[physx-side] $(date -Is) busy; sleeping 60s" | tee -a "$LOG"
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

echo "[physx-side] $(date -Is) live record PhysX-closed" | tee -a "$LOG"
set +e
python -u "${ROOT}/record_isaac_pursuit_video.py" \
  --method "$METHOD" \
  --env-config "$ENV_CFG" \
  --physx-closed-loop \
  --seed "$SEED" \
  --max-steps 400 \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --out "$OUT_MP4" \
  --traj-out "$TRAJ" \
  --skip-live-rgb \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ ! -f "$TRAJ" ]]; then
  echo "[physx-side] FAILED no traj rc=${rc}" | tee -a "$LOG"
  exit 1
fi

echo "[physx-side] $(date -Is) export side replay ${WIDTH}x${HEIGHT}@${FPS}" | tee -a "$LOG"
set +e
python -u "${ROOT}/export_pursuit_traj_video.py" \
  --traj "$TRAJ" \
  --env-config "$ENV_CFG" \
  --physx-closed-loop \
  --camera-view side \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --no-gif \
  --out "$OUT_MP4" \
  2>&1 | tee -a "$LOG"
erc=${PIPESTATUS[0]}
set -e

if [[ $erc -ne 0 || ! -f "$OUT_MP4" || $(stat -c%s "$OUT_MP4") -lt 10000 ]]; then
  echo "[physx-side] FAILED export erc=${erc}" | tee -a "$LOG"
  exit 1
fi

echo "[physx-side] $(date -Is) complete ${OUT_MP4}" | tee -a "$LOG"
ls -lah "$OUT_MP4" "$TRAJ" | tee -a "$LOG"
ffprobe -hide_banner "$OUT_MP4" 2>&1 | tee -a "$LOG" || true
