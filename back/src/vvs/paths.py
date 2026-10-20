"""Chemins du projet. Tout le code passe par ici, jamais de chemin en dur ailleurs."""

from __future__ import annotations

import os
from pathlib import Path

# back/src/vvs/paths.py -> racine du dépôt
ROOT = Path(os.environ.get("VVS_ROOT", Path(__file__).resolve().parents[3]))
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
REGISTRY = DATA / "registry"
RUNS = ROOT / "runs"


def ensure_dirs() -> None:
    for d in (RAW, CURATED, REGISTRY, RUNS):
        d.mkdir(parents=True, exist_ok=True)
