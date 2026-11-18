#!/usr/bin/env bash
# GUI pose-replay on DISPLAY=:2 + ffmpeg x11grab → MP4 (Oct20 coop path).
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
TRAJ="${TRAJ:-/home/wb/runs/isaac_dual_logs/mappo_physx_closed_capture_7000110.npz}"
ENV_CFG="${ENV_CFG:-${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json}"
OUT_DIR="${OUT_DIR:-/home/wb/runs/isaac_dual_logs}"
SEED_TAG=$(python3 - <<PY
import json,numpy as np
print(json.loads(str(np.load("$TRAJ",allow_pickle=False)["meta"]))["seed"])
PY
)
OUT_RAW="${OUT_DIR}/mappo_physx_closed_capture_${SEED_TAG}_raw.mkv"
OUT_MP4="${OUT_DIR}/mappo_physx_closed_capture_${SEED_TAG}.mp4"
LOG="${OUT_DIR}/grab_mappo_capture_${SEED_TAG}.log"
FFLOG="${OUT_DIR}/ffmpeg_mappo_grab_${SEED_TAG}.log"
CHECK_DIR=/tmp/mappo_capture_check
mkdir -p "$OUT_DIR" "$CHECK_DIR"

{
  echo "[grab] $(date -Is) traj=${TRAJ}"
  echo "[grab] out=${OUT_MP4} display=:2"
} | tee "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|export_pursuit_traj_video.py|replay_isaac_pursuit_traj.py' >/dev/null 2>&1; do
  echo "[grab] busy; sleeping 30s" | tee -a "$LOG"
  sleep 30
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
export DISPLAY=:2
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

rm -f "$OUT_RAW"
python -u "${ROOT}/replay_isaac_pursuit_traj.py" \
  --traj "$TRAJ" \
  --env-config "$ENV_CFG" \
  --physx-closed-loop \
  --visualize --follow-camera --camera-view side \
  --visual-scale 4.0 --no-body-markers \
  --speed 1.2 --trail-len 100 \
  >>"$LOG" 2>&1 &
REPLAY_PID=$!
echo "[grab] replay_pid=${REPLAY_PID}" | tee -a "$LOG"

for i in $(seq 1 120); do
  sleep 3
  if ! kill -0 "$REPLAY_PID" 2>/dev/null; then
    echo "[grab] replay died during startup" | tee -a "$LOG"
    tail -60 "$LOG"
    exit 1
  fi
  if grep -q '\[replay\] step=0/' "$LOG"; then
    echo "[grab] step0 ready after ${i}*3s" | tee -a "$LOG"
    break
  fi
done
if ! grep -q '\[replay\] step=0/' "$LOG"; then
  echo "[grab] timeout waiting for step0" | tee -a "$LOG"
  kill -9 "$REPLAY_PID" 2>/dev/null || true
  exit 1
fi

sleep 4
ffmpeg -y -f x11grab -video_size 1440x900 -framerate 1 -i :2.0+0,0 -frames:v 1 \
  "${CHECK_DIR}/pregrab.png" >/dev/null 2>&1 || true
python3 - <<PY
from pathlib import Path
import numpy as np
from PIL import Image
p=Path("${CHECK_DIR}/pregrab.png")
if p.exists():
    im=np.asarray(Image.open(p))
    print(f"[grab] pregrab_mean={float(im.mean()):.2f} shape={im.shape}", flush=True)
else:
    print("[grab] pregrab missing", flush=True)
PY

ffmpeg -y -f x11grab -video_size 1440x900 -framerate 12 -i :2.0+0,0 \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast "$OUT_RAW" \
  >"$FFLOG" 2>&1 &
echo $! > "${OUT_DIR}/ffmpeg_mappo_grab.pid"
echo "[grab] ffmpeg_pid=$(cat "${OUT_DIR}/ffmpeg_mappo_grab.pid")" | tee -a "$LOG"

for i in $(seq 1 120); do
  sleep 2
  if grep -q '\[replay\] done' "$LOG"; then
    echo "[grab] replay done; draining hold" | tee -a "$LOG"
    sleep 8
    break
  fi
  if ! kill -0 "$REPLAY_PID" 2>/dev/null; then
    break
  fi
done

FFPID=$(cat "${OUT_DIR}/ffmpeg_mappo_grab.pid")
kill -INT "$FFPID" 2>/dev/null || true
sleep 2
kill -9 "$FFPID" 2>/dev/null || true
wait "$REPLAY_PID" 2>/dev/null || true

ffmpeg -y -i "$OUT_RAW" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT_MP4"
ls -lah "$OUT_MP4" | tee -a "$LOG"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT_MP4" | tee -a "$LOG"
echo "[grab] $(date -Is) complete ${OUT_MP4}" | tee -a "$LOG"
