"""Modèles pydantic des fichiers de data/registry/ (voir data/registry/README.md).

Chaque modèle valide un fichier isolément ; les validations croisées entre fichiers sont
dans `vvs.registry.loader`.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from unidecode import unidecode

Bloc = Literal["far_right", "right", "centre", "left", "far_left", "other"]
BLOCS: tuple[str, ...] = get_args(Bloc)
PollsterMode = Literal["online", "phone", "mixed"]

ID_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")

# Tolérances des contrôles de résultats (en points de pourcentage)
SHARE_SUM_TOLERANCE = 0.2
SHARE_ROUNDING_TOLERANCE = 0.006


def slugify(text: str) -> str:
    """Minuscules ASCII, tout caractère non alphanumérique remplacé par `_`."""
    return re.sub(r"[^a-z0-9]+", "_", unidecode(text).lower()).strip("_")


def candidate_id_for(last_name: str, first_name: str) -> str:
    """Convention du registre : `nom_prenom` (`le_pen_marine`, `de_villiers_philippe`)."""
    return f"{slugify(last_name)}_{slugify(first_name)}"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class _Span(_Model):
    """Intervalle d'années inclusif ; `None` = ouvert de ce côté."""

    from_: int | None = Field(default=None, alias="from")
    to: int | None = None

    @model_validator(mode="after")
    def _ordered(self) -> _Span:
        if self.from_ is not None and self.to is not None and self.to < self.from_:
            raise ValueError(f"intervalle inversé : from={self.from_} > to={self.to}")
        return self

    def active(self, year: int) -> bool:
        return (self.from_ is None or self.from_ <= year) and (self.to is None or year <= self.to)


# --- candidates.yaml


class Affiliation(_Span):
    party: str


class Candidate(_Model):
    name: str
    first_name: str
    last_name: str
    parties: list[Affiliation] = Field(min_length=1)
    notes: str | None = None

    def parties_at(self, year: int) -> list[str]:
        return [a.party for a in self.parties if a.active(year)]

    def party_at(self, year: int) -> str | None:
        """Parti actif l'année donnée ; en cas de chevauchement, le plus récemment rejoint."""
        active = [a for a in self.parties if a.active(year)]
        if not active:
            return None
        return max(active, key=lambda a: -10_000 if a.from_ is None else a.from_).party


class CandidatesFile(_Model):
    candidates: dict[str, Candidate]


# --- parties.yaml


class Party(_Span):
    name: str
    short: str
    kind: Literal["party", "nuance"] = "party"
    bloc: Bloc
    successor: str | None = None
    notes: str | None = None


class PartiesFile(_Model):
    parties: dict[str, Party]


# --- blocs.yaml


class BlocDefinition(_Model):
    label: str
    definition: str


class BoundaryCase(_Model):
    """Cas limite tranché : candidats (ou parti) et élections auxquels s'applique `bloc`."""

    id: str
    candidate_ids: list[str] = Field(default_factory=list)
    party: str | None = None
    elections: list[int] = Field(min_length=1)
    bloc: Bloc
    rationale: str

    @model_validator(mode="after")
    def _has_target(self) -> BoundaryCase:
        if not self.candidate_ids and self.party is None:
            raise ValueError(f"cas limite {self.id} : ni candidate_ids ni party")
        return self


class BlocsFile(_Model):
    blocs: dict[Bloc, BlocDefinition]
    boundary_cases: list[BoundaryCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete(self) -> BlocsFile:
        missing = set(BLOCS) - set(self.blocs)
        if missing:
            raise ValueError(f"blocs non définis : {sorted(missing)}")
        ids = [c.id for c in self.boundary_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("boundary_cases : id en double")
        return self


# --- elections.yaml


class CandidateResult(_Model):
    candidate_id: str
    votes: int = Field(ge=0)
    share: float = Field(ge=0, le=100)


class Round(_Model):
    round: Literal[1, 2]
    date: dt.date
    date_status: Literal["official", "announced"]
    source: str | None = None
    source_url: str | None = None
    registered: int | None = Field(default=None, ge=0)
    voters: int | None = Field(default=None, ge=0)
    blank_ballots: int | None = Field(default=None, ge=0)
    null_ballots: int | None = Field(default=None, ge=0)
    expressed: int | None = Field(default=None, ge=0)
    results: list[CandidateResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Round:
        if not self.results:
            return self
        where = f"tour {self.round} du {self.date}"
        if self.registered is None or self.voters is None or self.expressed is None:
            raise ValueError(f"{where} : résultats sans inscrits / votants / exprimés")
        if not self.source_url:
            raise ValueError(f"{where} : résultats sans source_url")
        if not self.expressed <= self.voters <= self.registered:
            raise ValueError(f"{where} : il faut exprimés <= votants <= inscrits")
        if (
            self.blank_ballots is not None
            and self.null_ballots is not None
            and self.blank_ballots + self.null_ballots + self.expressed != self.voters
        ):
            raise ValueError(f"{where} : blancs + nuls + exprimés != votants")
        ids = [r.candidate_id for r in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{where} : candidat en double")
        if self.round == 2 and len(ids) != 2:
            raise ValueError(f"{where} : un second tour oppose deux candidats")
        total = sum(r.votes for r in self.results)
        if total != self.expressed:
            raise ValueError(f"{where} : somme des voix {total} != exprimés {self.expressed}")
        for r in self.results:
            expected = 100 * r.votes / self.expressed
            if abs(r.share - expected) > SHARE_ROUNDING_TOLERANCE:
                raise ValueError(
                    f"{where} : {r.candidate_id} share={r.share} "
                    f"incohérent avec voix / exprimés = {expected:.3f}"
                )
        share_sum = sum(r.share for r in self.results)
        if abs(share_sum - 100) > SHARE_SUM_TOLERANCE:
            raise ValueError(f"{where} : les pourcentages somment à {share_sum:.2f}")
        return self


class Candidacy(_Model):
    candidate_id: str
    party: str
    bloc: Bloc


class Election(_Model):
    date_note: str | None = None
    date_source_url: str | None = None
    candidates: list[Candidacy] = Field(default_factory=list)
    rounds: list[Round] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _consistent(self) -> Election:
        numbers = [r.round for r in self.rounds]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"tours mal numérotés : {numbers}")
        if len(self.rounds) == 2 and self.rounds[1].date <= self.rounds[0].date:
            raise ValueError("le second tour doit suivre le premier")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidat officiel en double")
        for r in self.rounds:
            if not r.results:
                continue
            got = {x.candidate_id for x in r.results}
            if not got <= set(ids):
                raise ValueError(f"tour {r.round} : hors liste officielle {sorted(got - set(ids))}")
            if r.round == 1 and got != set(ids):
                raise ValueError(f"tour 1 : sans résultat {sorted(set(ids) - got)}")
        return self

    def candidacy(self, candidate_id: str) -> Candidacy | None:
        return next((c for c in self.candidates if c.candidate_id == candidate_id), None)

    def round_(self, number: int) -> Round:
        return next(r for r in self.rounds if r.round == number)


class ElectionsFile(_Model):
    elections: dict[int, Election]


# --- pollsters.yaml


class Pollster(_Model):
    name: str
    aliases: list[str] = Field(default_factory=list)
    mode: PollsterMode
    successor: str | None = None
    notes: str | None = None


class PollstersFile(_Model):
    pollsters: dict[str, Pollster]
