#!/usr/bin/env bash
# Full-ladder α∈{0.5, 0.75} for MAPPO/A6 + MPC residual (reuse hard@200).
# Output: isaac_dual_eval_alpha_sweep_hard/<name>/{nominal,medium,hard,severe,extreme}.json
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
RUN=/srv/data/projects/kyy/runs/isaac_dual_physx_closed
TRAIN="${RUN}/train"
OUT="${TRAIN}/isaac_dual_eval_alpha_sweep_hard"
LOG="${RUN}/logs/alpha_sweep_full_levels.nohup"
NUM_ENVS="${NUM_ENVS:-10}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-200}"
LEVELS=(nominal medium hard severe extreme)
WEIGHTS=(0.5 0.75)

mkdir -p "$OUT" "${RUN}/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[alpha-full] $(date -Is) start levels=${LEVELS[*]} weights=${WEIGHTS[*]} seeds=${SEED_COUNT}"

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
cd "$ROOT"

CKPT_MAPPO="${TRAIN}/isaac_dual_mappo_plus_mpc_hard_physx_closed_seed11_residual_mpc_zeroinit_ep200/best_capture.pt"
CKPT_A6="${TRAIN}/isaac_dual_A6_mpc_hard_physx_closed_seed11_residual_mpc_zeroinit_ep200/best_capture.pt"

run_one() {
  local method="$1" ckpt="$2" name="$3" w="$4"
  local out="${OUT}/${name}"
  mkdir -p "$out"
  echo "[alpha-full] $(date -Is) START ${name} w=${w}"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels "${LEVELS[@]}" \
    --skip-existing-levels \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    --out "$out" \
    --residual-base mpc \
    --residual-mode blend \
    --residual-weight "$w"
  python3 - <<PY
import json
from pathlib import Path
out = Path("${out}")
want = list(range(${SEED_START}, ${SEED_START} + ${SEED_COUNT}))
print(f"RESULT ${name}:", flush=True)
for lv in ("nominal", "medium", "hard", "severe", "extreme"):
    p = out / f"{lv}.json"
    if not p.exists():
        print(f"  {lv}: MISSING"); continue
    d = json.loads(p.read_text())
    n = len(d.get("seeds") or [])
    ok = list(map(int, d.get("seeds") or [])) == want
    r = d["summary"]["capture_rate"]
    print(f"  {lv}: capture={r:.3f} n={n}{' OK200' if ok else ''}")
PY
  echo "[alpha-full] $(date -Is) DONE ${name}"
}

while pgrep -f 'evaluate_isaac_pursuit.py|train_isaac_pursuit.py' >/dev/null 2>&1; do
  echo "[alpha-full] waiting for leftover isaac jobs..."
  sleep 30
done

for w in "${WEIGHTS[@]}"; do
  tag=$(echo "$w" | tr '.' 'p')
  run_one mappo "$CKPT_MAPPO" "mappo_mpc_blend_w${tag}" "$w"
  run_one matd3 "$CKPT_A6" "a6_mpc_blend_w${tag}" "$w"
done

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train")
lf = root / "isaac_dual_eval_locked_full"
sw = root / "isaac_dual_eval_alpha_sweep_hard"
levels = ["nominal", "medium", "hard", "severe", "extreme"]
want = list(range(7000000, 7000200))

def rate(folder, name, lv):
    p = folder / name / f"{lv}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if list(map(int, d.get("seeds") or [])) != want:
        return None
    return d["summary"]["capture_rate"]

print("\n=== MPC-residual α full ladder @200 ===")
for method, cols in [
    ("MAPPO", [
        ("α=0", lf, "residual_mappo_mpc_w0"),
        ("α=0.25", lf, "residual_mappo_mpc_blend"),
        ("α=0.5", sw, "mappo_mpc_blend_w0p5"),
        ("α=0.75", sw, "mappo_mpc_blend_w0p75"),
    ]),
    ("A6", [
        ("α=0", lf, "residual_a6_mpc_w0"),
        ("α=0.25", lf, "residual_a6_mpc_blend"),
        ("α=0.5", sw, "a6_mpc_blend_w0p5"),
        ("α=0.75", sw, "a6_mpc_blend_w0p75"),
    ]),
]:
    print(f"\n--- {method}+MPC ---")
    print(f"{'α':8s}", " ".join(f"{lv:>8s}" for lv in levels))
    for label, folder, name in cols:
        cells = []
        for lv in levels:
            r = rate(folder, name, lv)
            cells.append(f"{r:8.3f}" if r is not None else f"{'…':>8s}")
        print(f"{label:8s}", " ".join(cells))
print("\nalso MPC teacher:")
print("MPC     ", " ".join(
    f"{rate(lf,'baseline_mpc',lv):8.3f}" if rate(lf,'baseline_mpc',lv) is not None else f"{'…':>8s}"
    for lv in levels
))
PY

echo "[alpha-full] $(date -Is) ALL COMPLETE"
