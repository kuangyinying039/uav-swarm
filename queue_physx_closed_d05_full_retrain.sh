#!/usr/bin/env bash
# DEPRECATED entrypoint: advisor protocol redirects here.
# Prefer: bash queue_physx_closed_teacher_enhance.sh
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
echo "[d05-full] redirecting to teacher-enhance mainline (residual-first, not pure-RL chase)" >&2
exec bash "${ROOT}/queue_physx_closed_teacher_enhance.sh"
