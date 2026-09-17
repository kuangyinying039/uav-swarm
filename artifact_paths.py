"""Stable locations for generated artifacts outside the source checkout."""
from __future__ import annotations

import os
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent


def _configured_output_root() -> Path:
    configured = os.environ.get("UAV_SWARM_OUTPUT_ROOT")
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return (PROJECT_ROOT / "outputs").resolve()


OUTPUT_ROOT = _configured_output_root()


def artifact_path(value) -> Path:
    """Resolve relative generated-data paths from the project, not the shell cwd.

    Thus ``outputs/run`` always means the sibling ``outputs`` directory next
    to ``repro``, including when a command is launched from inside ``repro``.
    Absolute paths remain valid for external disks and cluster scratch space.
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def default_output(*parts: str) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)
