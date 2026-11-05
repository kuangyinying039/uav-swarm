#!/usr/bin/env bash
# Quick locked-hard α sweep for existing MPC-residual best_capture ckpts.
# Fills α∈{0.5, 0.75}; 0 / 0.25 already in isaac_dual_eval_locked_full;
# α=1 is approximated by pure_rl_*_archive (different ckpt) unless --also-w1.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
RUN=/srv/data/projects/kyy/runs/isaac_dual_physx_closed
TRAIN="${RUN}/train"
OUT="${TRAIN}/isaac_dual_eval_alpha_sweep_hard"
LOG="${RUN}/logs/alpha_sweep_hard.nohup"
NUM_ENVS="${NUM_ENVS:-10}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
WEIGHTS=(0.5 0.75)
ALSO_W1="${ALSO_W1:-0}"

mkdir -p "$OUT" "${RUN}/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[alpha-sweep] $(date -Is) start weights=${WEIGHTS[*]} also_w1=${ALSO_W1}"

wait_trains() {
  echo "[alpha-sweep] $(date -Is) waiting for FRPN train_isaac to finish..."
  while pgrep -f 'train_isaac_pursuit.py.*(residual_frpn_blend_ep400)' >/dev/null 2>&1; do
    sleep 60
  done
  echo "[alpha-sweep] $(date -Is) trains gone"
}

# If FRPN enhance parent is about to start stage4 eval, pause it so we own the GPU.
pause_frpn_eval() {
  if pgrep -f 'queue_physx_closed_frpn_teacher_enhance.sh' >/dev/null 2>&1; then
    echo "[alpha-sweep] $(date -Is) pausing FRPN enhance parent + any early eval"
    # Stop enhance script; master will see it exit — we relaunch eval-only after sweep.
    pkill -f 'queue_physx_closed_frpn_teacher_enhance.sh' 2>/dev/null || true
    # Do not kill improve_master; it may exit when child dies — record flag for resume.
    pkill -f 'evaluate_isaac_pursuit.py' 2>/dev/null || true
    sleep 5
    echo 1 > "${RUN}/logs/alpha_sweep_needs_frpn_eval_resume.flag"
  fi
}

resume_frpn_eval() {
  if [[ -f "${RUN}/logs/alpha_sweep_needs_frpn_eval_resume.flag" ]]; then
    echo "[alpha-sweep] $(date -Is) resume FRPN stage4 eval only"
    rm -f "${RUN}/logs/alpha_sweep_needs_frpn_eval_resume.flag"
    NUM_ENVS=10 SKIP_TRAIN=1 SKIP_DAGGER=1 FORCE_COLLECT=0 \
      nohup bash "${ROOT}/queue_physx_closed_frpn_teacher_enhance.sh" \
      >> "${RUN}/logs/frpn_teacher_enhance_pipeline.log" 2>&1 &
    echo $! > "${RUN}/logs/frpn_eval_resume.pid"
    echo "[alpha-sweep] resumed FRPN eval pid=$!"
  fi
}

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
  echo "[alpha-sweep] $(date -Is) ${name} w=${w}"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels hard \
    --no-skip-existing-levels \
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
s=json.loads((Path("${out}")/"hard.json").read_text())["summary"]
print(f"RESULT ${name}: capture={s['capture_rate']:.3f} return={s['mean_return']:.2f} w={s.get('residual_weight')}")
PY
}

wait_trains
pause_frpn_eval

# Ensure GPU quiet
while pgrep -f 'evaluate_isaac_pursuit.py|train_isaac_pursuit.py' >/dev/null 2>&1; do
  echo "[alpha-sweep] waiting for leftover isaac jobs..."
  sleep 30
done

for w in "${WEIGHTS[@]}"; do
  tag=$(echo "$w" | tr '.' 'p')
  run_one mappo "$CKPT_MAPPO" "mappo_mpc_blend_w${tag}" "$w"
  run_one matd3 "$CKPT_A6" "a6_mpc_blend_w${tag}" "$w"
done

if [[ "${ALSO_W1}" == "1" ]]; then
  run_one mappo "$CKPT_MAPPO" "mappo_mpc_blend_w1" 1.0
  run_one matd3 "$CKPT_A6" "a6_mpc_blend_w1" 1.0
fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train")
lf = root / "isaac_dual_eval_locked_full"
sw = root / "isaac_dual_eval_alpha_sweep_hard"

def hard_rate(folder, name):
    p = folder / name / "hard.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]["capture_rate"]

print("=== MPC-residual α sweep (hard, 100 seeds) ===")
print(f"{'method':18s} {'α=0':>7} {'0.25':>7} {'0.5':>7} {'0.75':>7} {'1.0':>7}")
for method, w0, w25, prefix in [
    ("MAPPO", "residual_mappo_mpc_w0", "residual_mappo_mpc_blend", "mappo_mpc_blend"),
    ("A6", "residual_a6_mpc_w0", "residual_a6_mpc_blend", "a6_mpc_blend"),
]:
    r0 = hard_rate(lf, w0)
    r25 = hard_rate(lf, w25)
    r50 = hard_rate(sw, f"{prefix}_w0p5")
    r75 = hard_rate(sw, f"{prefix}_w0p75")
    r100 = hard_rate(sw, f"{prefix}_w1")
    # α=1 fallback: pure rl archive is different ckpt — mark separately if sweep missing
    if r100 is None:
        arch = "pure_rl_mappo_best_archive" if method == "MAPPO" else "pure_rl_a6_best_archive"
        r100_arch = hard_rate(lf, arch)
        r100_s = f"{r100_arch:.3f}*" if r100_arch is not None else "—"
    else:
        r100_s = f"{r100:.3f}"
    def fmt(x):
        return f"{x:.3f}" if x is not None else "—"
    print(f"{method:18s} {fmt(r0):>7} {fmt(r25):>7} {fmt(r50):>7} {fmt(r75):>7} {r100_s:>7}")
print("* α=1 column uses pure_rl archive if residual@w=1 not swept (different ckpt)")
PY

resume_frpn_eval
echo "[alpha-sweep] $(date -Is) complete"
