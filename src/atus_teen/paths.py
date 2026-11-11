"""Repository paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("ATUS_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def raw_dir() -> Path:
    return repo_root() / "data" / "raw"


def extract_path() -> Path:
    return repo_root() / "data" / "extract" / "teens_15_18.parquet"


def manifest_path() -> Path:
    return repo_root() / "data" / "extract" / "manifest.json"


def figures_dir() -> Path:
    return repo_root() / "figures"


def results_dir() -> Path:
    return repo_root() / "results"
