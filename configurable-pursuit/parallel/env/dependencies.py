"""Dependency path selection only; never imports Isaac or installs software."""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def known_sources():
    return (Path.home()/'.local/share/wb-platform/src/OmniDrones-isaac41',
            Path('/opt/lab/src/OmniDrones-isaac41'))


def resolve_source(value=None):
    if value is not None:
        candidates = [value]
    elif 'OMNIDRONES_SOURCE' in os.environ:
        candidates = [os.environ['OMNIDRONES_SOURCE']]
    else:
        candidates = known_sources()
    for candidate in candidates:
        if not str(candidate).strip():
            raise ValueError('Explicit OMNIDRONES_SOURCE must not be empty')
        source = Path(candidate).expanduser().resolve()
        if (source/'examples/demo.yaml').is_file():
            return source
        if value is not None or 'OMNIDRONES_SOURCE' in os.environ:
            raise ValueError(f'Invalid explicit OmniDrones source: {source}')
    raise ValueError('No compatible OmniDrones Isaac 4.1 checkout found; set OMNIDRONES_SOURCE or --omnidrones-source')


def select_source(value=None):
    """Select before importing omni_drones; CLI overrides the activated default."""
    source = resolve_source(value)
    sys.path.insert(0, str(source))
    os.environ['OMNIDRONES_SOURCE'] = str(source)
    return source


def project_output(value):
    """Absolute/~ paths stay absolute; relative paths are anchored to this project."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
