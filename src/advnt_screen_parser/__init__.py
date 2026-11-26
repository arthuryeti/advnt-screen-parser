"""Advnt Screen Parser (v4) — CPU-fast iOS UI icon detector + RapidOCR + UI-tree builder."""

from __future__ import annotations

from .icon_detector import DEFAULT_HF_REPO, IconDetector, download_weights
from .uitree import build as build_ui_tree
from .visualize import render_overlay

try:
    from .parser import ScreenParser
except ImportError:  # pragma: no cover - rapidocr optional for icon-only installs
    ScreenParser = None  # type: ignore[assignment,misc]

__all__ = [
    "DEFAULT_HF_REPO",
    "IconDetector",
    "ScreenParser",
    "build_ui_tree",
    "download_weights",
    "render_overlay",
]
__version__ = "4.0.0"
