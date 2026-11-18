#!/usr/bin/env bash
# Evaluate the new STRIKER/BLOCKER/RESERVE MPC on the 5-level PhysX-closed ladder.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
OUT="${OUT:-/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_mpc_roles_5levels}"
LOG="${LOG:-/srv/data/projects/kyy/runs/isaac_dual_physx_closed/logs/eval_mpc_roles_5levels.log}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-200}"
NUM_ENVS="${NUM_ENVS:-6}"
LEVELS="${LEVELS:-nominal medium hard severe extreme}"

mkdir -p "$(dirname "$LOG")" "$OUT"
{
  echo "[mpc-eval] $(date -Is) start"
  echo "[mpc-eval] out=${OUT}"
  echo "[mpc-eval] levels=${LEVELS} seeds=${SEED_START}+${SEED_COUNT} num_envs=${NUM_ENVS}"
} | tee "$LOG"

while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py|record_isaac_pursuit|export_pursuit_traj' >/dev/null 2>&1; do
  echo "[mpc-eval] $(date -Is) busy; sleeping 60s" | tee -a "$LOG"
  sleep 60
done

# Fresh env: avoid inherited PYTHONPATH/LD_LIBRARY_PATH blow-ups ("Argument list too long").
unset PYTHONPATH LD_LIBRARY_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH PYTHONSTARTUP || true
source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set -euo pipefail
# Keep Isaac Sim pip_prebundle (torch etc.) that setup_conda_env.sh injected.
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

# shellcheck disable=SC2086
python -u "${ROOT}/evaluate_isaac_pursuit.py" \
  --method mpc \
  --levels ${LEVELS} \
  --physx-closed-loop \
  --no-native-lidar \
  --num-envs "${NUM_ENVS}" \
  --seed-start "${SEED_START}" \
  --seed-count "${SEED_COUNT}" \
  --log-every 10 \
  --no-skip-existing-levels \
  --out "${OUT}" \
  >>"$LOG" 2>&1

echo "[mpc-eval] $(date -Is) done" | tee -a "$LOG"
python - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
print("=== new MPC (STRIKER/BLOCKER/RESERVE) PhysX-closed ===")
for level in "nominal medium hard severe extreme".split():
    p = out / f"{level}.json"
    if not p.is_file():
        print(f"{level}: MISSING")
        continue
    data = json.loads(p.read_text())
    s = data.get("summary") or {}
    print(
        f"{level}: capture={s.get('capture_rate')} "
        f"n={s.get('episodes')} collisions={s.get('mean_collisions')} "
        f"success_steps={s.get('mean_success_steps')}"
    )
PY
