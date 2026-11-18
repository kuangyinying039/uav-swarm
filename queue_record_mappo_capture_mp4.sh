#!/usr/bin/env bash
# Live MAPPO PhysX-closed capture → traj npz → (if needed) pose-replay MP4.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
CKPT="${CKPT:-/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_mappo_plus_mpc_hard_physx_closed_seed11_pure_curriculum_a0to1/final.pt}"
# Prefer seeds that captured in locked pure_rl_mappo_final hard eval, plus 7000110
# which already captured once under the live video warmup path.
SEEDS="${SEEDS:-7000110 7000105 7000012 7000113 7000007 7000174 7000087 7000024}"
OUT_DIR="${OUT_DIR:-/home/wb/runs/isaac_dual_logs}"
LOG="${LOG:-${OUT_DIR}/record_mappo_capture_mp4.log}"
ENV_CFG="${ENV_CFG:-${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json}"
CAMERA_VIEW="${CAMERA_VIEW:-side}"

mkdir -p "$OUT_DIR"
{
  echo "[record-queue] $(date -Is) wait for free Isaac GPU"
  echo "[record-queue] ckpt=${CKPT}"
  echo "[record-queue] seeds=${SEEDS}"
} | tee "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|export_pursuit_traj_video.py' >/dev/null 2>&1; do
  echo "[record-queue] $(date -Is) busy; sleeping 60s" | tee -a "$LOG"
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

FINAL_MP4=""
for SEED in $SEEDS; do
  OUT="${OUT_DIR}/mappo_physx_closed_capture_${SEED}.mp4"
  TRAJ="${OUT_DIR}/mappo_physx_closed_capture_${SEED}.npz"
  echo "[record-queue] $(date -Is) live record seed=${SEED}" | tee -a "$LOG"
  set +e
  python -u "${ROOT}/record_isaac_pursuit_video.py" \
    --method mappo \
    --checkpoint "$CKPT" \
    --env-config "$ENV_CFG" \
    --physx-closed-loop \
    --seed "$SEED" \
    --max-steps 300 \
    --fps 10 \
    --out "$OUT" \
    --traj-out "$TRAJ" \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ -f "$OUT" && $(stat -c%s "$OUT") -gt 10000 ]]; then
    FINAL_MP4="$OUT"
    echo "[record-queue] live RGB mp4 ok seed=${SEED}" | tee -a "$LOG"
    break
  fi

  if [[ -f "$TRAJ" ]]; then
    cap=$(python - <<PY
import json,numpy as np
m=json.loads(str(np.load("$TRAJ",allow_pickle=False)["meta"]))
print("1" if m.get("captured") else "0")
PY
)
    if [[ "$cap" == "1" ]]; then
      echo "[record-queue] $(date -Is) export mp4 from traj seed=${SEED}" | tee -a "$LOG"
      set +e
      python -u "${ROOT}/export_pursuit_traj_video.py" \
        --traj "$TRAJ" \
        --env-config "$ENV_CFG" \
        --physx-closed-loop \
        --camera-view "$CAMERA_VIEW" \
        --fps 10 \
        --no-gif \
        --out "$OUT" \
        2>&1 | tee -a "$LOG"
      erc=${PIPESTATUS[0]}
      set -e
      if [[ $erc -eq 0 && -f "$OUT" && $(stat -c%s "$OUT") -gt 10000 ]]; then
        FINAL_MP4="$OUT"
        break
      fi
    else
      echo "[record-queue] seed=${SEED} no capture (rc=${rc}); try next" | tee -a "$LOG"
    fi
  else
    echo "[record-queue] seed=${SEED} no traj (rc=${rc}); try next" | tee -a "$LOG"
  fi
done

if [[ -z "$FINAL_MP4" ]]; then
  echo "[record-queue] FAILED: no capture MP4 from seeds=${SEEDS}" | tee -a "$LOG"
  exit 1
fi

echo "[record-queue] $(date -Is) complete out=${FINAL_MP4}" | tee -a "$LOG"
ls -lah "$FINAL_MP4" | tee -a "$LOG"
