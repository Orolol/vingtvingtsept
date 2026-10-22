"""Inventaire des enquêtes du Cevipof : data/registry/cevipof_datasets.yaml.

Une entrée par jeu Dataverse (une vague de panel = un jeu), avec ses fichiers, son statut
d'accès et les questions utiles repérées dans le codebook ou le questionnaire. Le fichier est
tenu à la main ; `vvs ingest cevipof --check` le compare à l'API (versions, fichiers, accès).
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vvs import paths

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PID_RE = re.compile(r"^doi:10\.21410/7E4/[A-Z0-9]{6}$")

# Rôle d'un fichier ; seuls `data` et `data_alt` sont des microdonnées.
Role = Literal[
    "data",  # fichier de microdonnées retenu (Stata de préférence, lu par pandas)
    "data_alt",  # mêmes microdonnées dans un autre format (SPSS, SAS, CSV)
    "verbatims",
    "codebook",  # codebook DDI XML ou dictionnaire des variables
    "questionnaire",
    "documentation",
    "technical_report",
    "weights_doc",
    "other",
]
Access = Literal["public", "restricted_request", "embargo", "mixed"]
# Moment de la vague par rapport à la présidentielle `election`
Stage = Literal[
    "pre_campaign",  # plus de trois mois avant le premier tour
    "campaign",  # trois derniers mois avant le premier tour
    "eve_round1",  # dernière semaine avant le premier tour
    "between_rounds",
    "post_round2",  # après le second tour, avant les législatives
    "post_legislative",
    "later",  # au-delà (vagues de suivi)
    "other_election",  # vague centrée sur un autre scrutin (européennes, législatives 2024)
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class DataFile(_Model):
    id: int
    name: str
    format: str  # type MIME du fichier déposé (`originalFileFormat`) ou servi
    size: int  # octets du fichier déposé
    restricted: bool
    role: Role
    fetch: bool = False  # téléchargé par `vvs ingest cevipof`
    sha256: str | None = None  # empreinte au premier téléchargement


class Question(_Model):
    """Variable repérée : nom dans le fichier, libellé du codebook, note éventuelle."""

    var: str
    label: str
    note: str | None = None


class Dataset(_Model):
    key: str
    pid: str
    url: str
    title: str
    series: str
    election: int | None = None  # présidentielle de rattachement (None : hors présidentielle)
    wave: str | None = None
    wave_order: float | None = None
    stage: Stage | None = None
    fieldwork_start: dt.date | None = None
    fieldwork_end: dt.date | None = None
    fieldwork_note: str | None = None
    mode: str | None = None
    sampling: str | None = None
    version: str
    license: str
    access: Access
    request: bool = False  # demande d'accès à lancer pour le fichier `data`
    priority: Literal["core", "useful", "listed"] = "listed"
    files: list[DataFile]
    questions: dict[str, list[Question]] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Dataset:
        if not KEY_RE.match(self.key):
            raise ValueError(f"{self.key} : clé mal formée")
        if not PID_RE.match(self.pid):
            raise ValueError(f"{self.key} : persistentId mal formé {self.pid}")
        if self.url != "https://doi.org/" + self.pid.removeprefix("doi:"):
            raise ValueError(f"{self.key} : url ≠ DOI")
        if (
            self.fieldwork_start
            and self.fieldwork_end
            and self.fieldwork_end < self.fieldwork_start
        ):
            raise ValueError(f"{self.key} : terrain inversé")
        data = [f for f in self.files if f.role == "data"]
        if len(data) > 1:
            raise ValueError(f"{self.key} : plusieurs fichiers `data`")
        if self.request and not data:
            raise ValueError(f"{self.key} : `request` sans fichier `data`")
        ids = [f.id for f in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.key} : fichier en double")
        return self

    @property
    def data_file(self) -> DataFile | None:
        return next((f for f in self.files if f.role == "data"), None)


class Inventory(_Model):
    retrieved: dt.date
    server: str
    datasets: list[Dataset]

    @model_validator(mode="after")
    def _unique(self) -> Inventory:
        keys = [d.key for d in self.datasets]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            raise ValueError(f"clés en double : {sorted(dup)}")
        pids = [d.pid for d in self.datasets]
        if len(pids) != len(set(pids)):
            raise ValueError("persistentId en double")
        return self

    def get(self, key: str) -> Dataset:
        for d in self.datasets:
            if d.key == key:
                return d
        raise KeyError(key)


def default_path() -> Path:
    return paths.REGISTRY / "cevipof_datasets.yaml"


def load_inventory(path: Path | None = None) -> Inventory:
    path = default_path() if path is None else Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Inventory.model_validate(data)


@functools.cache
def default_inventory() -> Inventory:
    return load_inventory()
