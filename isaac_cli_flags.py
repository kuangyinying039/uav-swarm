"""Shared CLI helpers for Isaac pursuit entrypoints."""
from __future__ import annotations

import argparse


def add_isaac_display_flags(parser: argparse.ArgumentParser, *, default_headless: bool = True) -> None:
    """Add ``--headless/--no-headless`` and ``--visualize`` (implies GUI)."""
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=default_headless,
        help="Run without Isaac Viewport (default for training/eval suites)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Isaac Kit Viewport (implies --no-headless). Needs DISPLAY.",
    )


def resolve_headless(args: argparse.Namespace) -> bool:
    """Return the effective headless flag after applying ``--visualize``."""
    if bool(getattr(args, "visualize", False)):
        return False
    return bool(getattr(args, "headless", True))
