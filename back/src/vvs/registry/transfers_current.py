"""Reports publiés avant le premier tour pour l'élection en cours : data/registry/transfers_current.yaml.

Une entrée = un tableau « reports de voix » d'un sondage de duel hypothétique : répartition de
l'électorat de chaque candidat non finaliste (intention de vote au premier tour) entre les deux
finalistes de l'hypothèse et « ni l'un ni l'autre » (abstention, blanc, nul, ne se prononce pas).
Option du second tour, désactivée par défaut (`SecondRoundModel(current_transfers=…)`,
docs/results/2026-11-24-reports-publies.md) ; distinct de transfers.yaml (reports réalisés).

Disponibilité d'un tableau : sa date de publication si elle a été lue, sinon la fin du terrain
plus `PUBLICATION_DELAY` jours (règle du gel de la consolidation).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from vvs import paths
from vvs.registry.loader import Registry, default_registry
from vvs.registry.models import ID_RE, _Model
from vvs.registry.transfers import ROW_SUM_TOLERANCE, TransferError

NONE = "none"
PUBLICATION_DELAY = 2  # jours après la fin du terrain quand la publication n'est pas lue


class CurrentRow(_Model):
    from_: str = Field(alias="from")
    to: dict[str, float]
    note: str | None = None


class CurrentHypothesis(_Model):
    label: str
    rows: list[CurrentRow] = Field(min_length=1)


class CurrentTable(_Model):
    id: str
    election: int
    pair: tuple[str, str]
    pollster: str
    sponsor: str
    title: str
    url: str
    file: str | None = None
    pages: str
    fieldwork_start: dt.date
    fieldwork_end: dt.date
    published: dt.date | None = None
    sample_size: int | None = Field(default=None, gt=0)
    mode: Literal["online", "phone", "mixed"]
    base: Literal["registered", "likely_voters", "other"]
    base_note: str
    hypotheses: list[CurrentHypothesis] = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CurrentTable:
        if not ID_RE.match(self.id):
            raise ValueError(f"{self.id} : id mal formé")
        if self.pair[0] == self.pair[1]:
            raise ValueError(f"{self.id} : paire dégénérée")
        if self.fieldwork_end < self.fieldwork_start:
            raise ValueError(f"{self.id} : terrain inversé")
        if self.published is not None and self.published < self.fieldwork_end:
            raise ValueError(f"{self.id} : publié avant la fin du terrain")
        cols = {*self.pair, NONE}
        for h in self.hypotheses:
            froms = [r.from_ for r in h.rows]
            if len(froms) != len(set(froms)):
                raise ValueError(f"{self.id} ({h.label}) : électorat en double")
            for r in h.rows:
                if set(r.to) != cols:
                    raise ValueError(f"{self.id}, {r.from_} : colonnes {sorted(r.to)} au lieu de {sorted(cols)}")
                if any(v < 0 or v > 100 for v in r.to.values()):
                    raise ValueError(f"{self.id}, {r.from_} : pourcentage hors de [0, 100]")
                if abs(sum(r.to.values()) - 100) > ROW_SUM_TOLERANCE:
                    raise ValueError(f"{self.id}, {r.from_} : somme {sum(r.to.values())} ≠ 100")
                if r.from_ in self.pair:
                    raise ValueError(f"{self.id} : ligne d'un finaliste ({r.from_}) non prévue")
        return self

    @property
    def available(self) -> dt.date:
        """Date à partir de laquelle le tableau est connu."""
        if self.published is not None:
            return self.published
        return self.fieldwork_end + dt.timedelta(days=PUBLICATION_DELAY)

    def vector(self, row: CurrentRow, pair: tuple[str, str]) -> tuple[float, float, float]:
        """(vers pair[0], vers pair[1], ni l'un ni l'autre), en fractions sommant à 1."""
        x = (row.to[pair[0]], row.to[pair[1]], row.to[NONE])
        s = sum(x)
        return (x[0] / s, x[1] / s, x[2] / s)


class CurrentTransfersFile(_Model):
    tables: list[CurrentTable]


def default_path() -> Path:
    return paths.REGISTRY / "transfers_current.yaml"


def load_current_transfers(
    path: Path | None = None,
    registry: Registry | None = None,
    elections: set[int] | None = None,
) -> tuple[CurrentTable, ...]:
    """Charge et valide le fichier (défaut : `transfers_current.yaml` du registre). `elections` :
    ne garder (et ne valider contre le registre) que ces élections. Lève `TransferError`."""
    path = default_path() if path is None else Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TransferError(f"{path.name} : document YAML vide ou mal formé")
    try:
        tables = CurrentTransfersFile.model_validate(data).tables
    except ValueError as e:
        raise TransferError(f"{path.name} invalide : {e}") from e
    ids = [t.id for t in tables]
    if len(ids) != len(set(ids)):
        raise TransferError(f"{path.name} : id de tableau en double")
    if elections is not None:
        tables = [t for t in tables if t.election in elections]
    reg = registry or default_registry()
    errors = []
    for t in tables:
        if t.pollster not in reg.pollsters:
            errors.append(f"{t.id} : institut inconnu {t.pollster!r}")
        for c in {*t.pair, *(r.from_ for h in t.hypotheses for r in h.rows)}:
            if c not in reg.candidates:
                errors.append(f"{t.id} : candidat inconnu {c!r}")
    if errors:
        raise TransferError("tableaux de reports incohérents :\n- " + "\n- ".join(errors))
    return tuple(tables)
