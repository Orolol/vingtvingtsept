"""Registre de référence : candidats, partis, blocs, élections, instituts (data/registry/)."""

from vvs.registry.loader import (
    Registry,
    RegistryError,
    bloc_of,
    default_registry,
    load_registry,
    normalize_label,
    pollster_of,
)
from vvs.registry.models import BLOCS, candidate_id_for

__all__ = [
    "BLOCS",
    "Registry",
    "RegistryError",
    "bloc_of",
    "candidate_id_for",
    "default_registry",
    "load_registry",
    "normalize_label",
    "pollster_of",
]
