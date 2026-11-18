#!/usr/bin/env bash
# Isaac GUI pose-replay (Oct20 coop style) + per-window xwd capture → MP4.
# Root x11grab is black under current Vulkan/Mutter; xwd -id works.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
TRAJ="${TRAJ:-/home/wb/runs/isaac_dual_logs/physx_closed_mpc_seed7200000_isaac_side.npz}"
ENV_CFG="${ENV_CFG:-${ROOT}/configs/pursuit_v2/radar5_dual_hard_physx_closed.json}"
OUT_DIR="${OUT_DIR:-/home/wb/runs/isaac_dual_logs}"
CAMERA_VIEW="${CAMERA_VIEW:-side}"
STEM="${STEM:-physx_closed_mpc_seed7200000_isaac_${CAMERA_VIEW}}"
OUT_MP4="${OUT_DIR}/${STEM}.mp4"
FRAMES="${OUT_DIR}/${STEM}_gui_frames"
LOG="${OUT_DIR}/grab_${STEM}.log"
SPEED="${SPEED:-2.5}"
mkdir -p "$OUT_DIR" "$FRAMES"
rm -f "$FRAMES"/frame_*.png

{
  echo "[grab] $(date -Is) traj=${TRAJ}"
  echo "[grab] out=${OUT_MP4} display=:2 camera=${CAMERA_VIEW} method=xwd-window (Oct20 replay flags)"
} | tee "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|export_pursuit_traj_video.py|replay_isaac_pursuit_traj.py' >/dev/null 2>&1; do
  echo "[grab] busy; sleeping 20s" | tee -a "$LOG"
  sleep 20
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

python -u "${ROOT}/replay_isaac_pursuit_traj.py" \
  --traj "$TRAJ" \
  --env-config "$ENV_CFG" \
  --physx-closed-loop \
  --visualize --follow-camera --camera-view "$CAMERA_VIEW" \
  --visual-scale 4.0 --no-body-markers --target-wireframe \
  --speed "$SPEED" --trail-len 100 \
  >>"$LOG" 2>&1 &
REPLAY_PID=$!
echo "[grab] replay_pid=${REPLAY_PID} camera=${CAMERA_VIEW}" | tee -a "$LOG"

for i in $(seq 1 200); do
  sleep 3
  if ! kill -0 "$REPLAY_PID" 2>/dev/null; then
    echo "[grab] replay died during startup" | tee -a "$LOG"
    tail -80 "$LOG"
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

# Freeze briefly to resolve Isaac window id
kill -STOP "$REPLAY_PID" || true
sleep 1
TREE=/tmp/physx_side_tree.txt
DISPLAY=:2 xwininfo -root -tree >"$TREE" 2>&1 || true
WID=$(python3 - <<'PY'
import re
txt=open("/tmp/physx_side_tree.txt", encoding="utf-8", errors="ignore").read()
best=None
for m in re.finditer(r'(0x[0-9a-fA-F]+)\s+"([^"]*)".*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)', txt):
    wid, title, w, h = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if "Isaac Sim" in title and w >= 800 and h >= 600:
        best = wid
        break
print(best or "")
PY
)
if [[ -z "$WID" ]]; then
  echo "[grab] Isaac window not found" | tee -a "$LOG"
  kill -CONT "$REPLAY_PID" 2>/dev/null || true
  kill -9 "$REPLAY_PID" 2>/dev/null || true
  exit 1
fi
echo "[grab] Isaac WID=${WID}" | tee -a "$LOG"

# Smoke-test one frame
XWD=/tmp/physx_side_frame.xwd
DISPLAY=:2 xwd -silent -id "$WID" -out "$XWD"
ffmpeg -y -i "$XWD" "${FRAMES}/frame_0000.png" >/dev/null 2>&1
python3 - <<PY
from PIL import Image
import numpy as np, sys
im=np.asarray(Image.open("${FRAMES}/frame_0000.png"))
print(f"[grab] smoke mean={im.mean():.1f} shape={im.shape}", flush=True)
sys.exit(0 if im.mean() > 5 else 1)
PY

kill -CONT "$REPLAY_PID" || true

# Capture loop (xwd is slower than realtime; replay speed=2.5 helps)
IDX=1
FAILS=0
while kill -0 "$REPLAY_PID" 2>/dev/null; do
  if grep -q '\[replay\] done' "$LOG"; then
    # a few extra end frames
    for _ in 1 2 3 4 5 6; do
      if DISPLAY=:2 xwd -silent -id "$WID" -out "$XWD" 2>/dev/null; then
        ffmpeg -y -i "$XWD" "$(printf "%s/frame_%04d.png" "$FRAMES" "$IDX")" >/dev/null 2>&1 || true
        IDX=$((IDX + 1))
      fi
      sleep 0.15
    done
    break
  fi
  if DISPLAY=:2 xwd -silent -id "$WID" -out "$XWD" 2>/dev/null; then
    if ffmpeg -y -i "$XWD" "$(printf "%s/frame_%04d.png" "$FRAMES" "$IDX")" >/dev/null 2>&1; then
      IDX=$((IDX + 1))
      FAILS=0
    else
      FAILS=$((FAILS + 1))
    fi
  else
    FAILS=$((FAILS + 1))
  fi
  if (( FAILS > 40 )); then
    echo "[grab] too many xwd failures" | tee -a "$LOG"
    break
  fi
  # light pacing
  sleep 0.05
done

wait "$REPLAY_PID" 2>/dev/null || true
NFRAMES=$(find "$FRAMES" -name 'frame_*.png' | wc -l)
echo "[grab] captured_frames=${NFRAMES}" | tee -a "$LOG"
if (( NFRAMES < 5 )); then
  echo "[grab] FAILED too few frames" | tee -a "$LOG"
  exit 1
fi

FPS="${FPS:-8}"
ffmpeg -y -framerate "$FPS" -i "${FRAMES}/frame_%04d.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT_MP4"
ls -lah "$OUT_MP4" | tee -a "$LOG"
ffprobe -hide_banner "$OUT_MP4" 2>&1 | tee -a "$LOG" || true
echo "[grab] $(date -Is) complete ${OUT_MP4}" | tee -a "$LOG"
