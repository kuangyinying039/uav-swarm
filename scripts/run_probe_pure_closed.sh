#!/usr/bin/env bash
set -euo pipefail
source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate /home/wb/.conda/envs/omnidrones-isaac41-writable
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set +u
cd /srv/data/projects/kyy/xieton/repro
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:/srv/data/projects/kyy/xieton/repro:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY || true
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
exec python -u scripts/probe_pure_closed_lee_lidar.py
