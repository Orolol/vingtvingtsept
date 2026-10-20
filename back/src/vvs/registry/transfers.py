"""Reports de voix historiques entre les deux tours : data/registry/transfers.yaml.

Une entrée = une source (enquête jour du vote, post-électorale) pour un second tour, avec une
matrice électorat du premier tour → finalistes, blancs-nuls, abstention. Sert de prior aux
reports `T` du second tour (docs/04-modele.md § 4).

Les cellules absentes d'une source restent absentes : aucune n'est déduite par différence.
"""

from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal

import yaml
from pydantic import Field, model_validator

from vvs import paths
from vvs.registry.loader import Registry, default_registry
from vvs.registry.models import ID_RE, _Model

# Écart toléré entre la somme d'une ligne complète et 100 (arrondis des instituts)
ROW_SUM_TOLERANCE = 2.0

BLANK_NULL = "blank_null"
ABSTAIN = "abstain"
BLANK_OR_ABSTAIN = "blank_or_abstain"

# Colonnes hors finalistes selon le format de la source
Columns = Literal["split", "combined", "voters", "expressed"]
OUTCOME_COLUMNS: dict[str, tuple[str, ...]] = {
    "split": (BLANK_NULL, ABSTAIN),  # blancs-nuls et abstention distingués
    "combined": (BLANK_OR_ABSTAIN,),  # une seule colonne blancs, nuls et abstention
    "voters": (BLANK_NULL,),  # base : votants du second tour (pas d'abstention)
    "expressed": (),  # base : exprimés du second tour (finalistes seuls)
}
Base = Literal["first_round_electorate", "second_round_voters", "second_round_expressed", "other"]
EXPECTED_BASE: dict[str, set[str]] = {
    "split": {"first_round_electorate", "other"},
    "combined": {"first_round_electorate", "other"},
    "voters": {"second_round_voters"},
    "expressed": {"second_round_expressed"},
}
# Électorats de premier tour qui ne sont pas des candidats
SPECIAL_ELECTORATES = {"abstain", "blank_null", "abstain_or_blank_null"}
# Producteurs admis hors pollsters.yaml
OTHER_PRODUCERS = {"cevipof"}

Basis = Literal["electorate", "split", "expressed"]


class TransferError(ValueError):
    """Incohérence du fichier de reports ; le message liste tous les problèmes."""


class TransferRow(_Model):
    from_: str = Field(alias="from")
    # électorat regroupé : candidats membres, ou à défaut une note sur sa composition
    members: list[str] = Field(default_factory=list)
    group_note: str | None = None
    to: dict[str, float]

    @model_validator(mode="after")
    def _shares(self) -> TransferRow:
        if not self.to:
            raise ValueError(f"ligne {self.from_} : aucune cellule")
        if any(v < 0 or v > 100 for v in self.to.values()):
            raise ValueError(f"ligne {self.from_} : pourcentage hors de [0, 100]")
        return self


class TransferSource(_Model):
    id: str
    election: int
    pollster: str
    title: str
    url: str
    pages: str | None = None
    fieldwork_start: dt.date | None = None
    fieldwork_end: dt.date | None = None
    sample_size: int | None = Field(default=None, gt=0)
    mode: Literal["online", "phone", "mixed"] | None = None
    base: Base
    base_note: str
    columns: Columns
    notes: str | None = None
    rows: list[TransferRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> TransferSource:
        if self.base not in EXPECTED_BASE[self.columns]:
            raise ValueError(
                f"{self.id} : base {self.base} incompatible avec columns={self.columns}"
            )
        if (
            self.fieldwork_start
            and self.fieldwork_end
            and self.fieldwork_end < self.fieldwork_start
        ):
            raise ValueError(f"{self.id} : terrain inversé")
        froms = [r.from_ for r in self.rows]
        if len(froms) != len(set(froms)):
            raise ValueError(f"{self.id} : électorat en double")
        return self


class TransfersFile(_Model):
    transfers: list[TransferSource]


def _finalists(reg: Registry, election: int) -> tuple[str, str] | None:
    e = reg.elections.get(election)
    if e is None or len(e.rounds) < 2 or not e.round_(2).results:
        return None
    a, b = (r.candidate_id for r in e.round_(2).results)
    return a, b


def _check(sources: list[TransferSource], reg: Registry) -> list[str]:
    errors: list[str] = []
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        errors.append("transfers : id de source en double")
    for s in sources:
        where = s.id
        if not ID_RE.match(s.id):
            errors.append(f"{where} : id mal formé")
        if s.pollster not in reg.pollsters and s.pollster not in OTHER_PRODUCERS:
            errors.append(f"{where} : institut inconnu {s.pollster!r}")
        pair = _finalists(reg, s.election)
        if pair is None:
            errors.append(f"{where} : pas de second tour connu en {s.election}")
            continue
        candidates = {c.candidate_id for c in reg.elections[s.election].candidates}
        allowed = set(pair) | set(OUTCOME_COLUMNS[s.columns])
        for r in s.rows:
            row = f"{where}, ligne {r.from_}"
            if r.from_ in candidates or r.from_ in SPECIAL_ELECTORATES:
                if r.members or r.group_note:
                    errors.append(f"{row} : `members` / `group_note` réservés aux groupes")
            elif not (r.members or r.group_note):
                errors.append(
                    f"{row} : ni candidat de {s.election}, ni groupe avec `members` ou `group_note`"
                )
            for m in r.members:
                if m not in candidates:
                    errors.append(f"{row} : membre {m} pas candidat en {s.election}")
            extra = set(r.to) - allowed
            if extra:
                errors.append(f"{row} : colonnes inattendues {sorted(extra)} (columns={s.columns})")
            total = sum(r.to.values())
            if set(r.to) == allowed:
                if abs(total - 100) > ROW_SUM_TOLERANCE:
                    errors.append(f"{row} : somme {total:g} au lieu de 100")
            elif total > 100 + ROW_SUM_TOLERANCE:
                errors.append(f"{row} : ligne incomplète dont la somme dépasse 100 ({total:g})")
    return errors


def default_path() -> Path:
    return paths.REGISTRY / "transfers.yaml"


def load_transfers(
    path: Path | None = None, registry: Registry | None = None
) -> list[TransferSource]:
    """Charge et valide transfers.yaml ; lève `TransferError` à la moindre incohérence."""
    path = default_path() if path is None else Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TransferError(f"{path.name} : document YAML vide ou mal formé")
    try:
        sources = TransfersFile.model_validate(data).transfers
    except ValueError as e:
        raise TransferError(f"{path.name} invalide : {e}") from e
    errors = _check(sources, registry or default_registry())
    if errors:
        raise TransferError("reports incohérents :\n- " + "\n- ".join(errors))
    return sources


@functools.cache
def default_transfers() -> tuple[TransferSource, ...]:
    return tuple(load_transfers())


# --- prior


@dataclass(frozen=True)
class TransferPrior:
    """Moyenne simple des sources disponibles, en % de l'électorat ou des exprimés.

    `shares` : clés = les deux finalistes dans l'ordre de `pair`, puis selon `basis` :
    `blank_or_abstain` (electorate), `blank_null` et `abstain` (split), rien (expressed).
    `per_source` : les vecteurs de chaque source, pour mesurer la dispersion.
    """

    election: int
    from_candidate: str
    pair: tuple[str, str]
    basis: Basis
    shares: dict[str, float]
    per_source: dict[str, dict[str, float]]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.per_source)


def _vector(
    s: TransferSource, row: TransferRow, pair: tuple[str, str], basis: Basis
) -> dict[str, float] | None:
    """Vecteur d'une ligne dans la base demandée, ou None si la ligne ne le permet pas."""
    to = row.to
    if set(to) != set(pair) | set(OUTCOME_COLUMNS[s.columns]):
        return None  # ligne incomplète : on ne déduit rien
    i, j = pair
    if basis == "expressed":
        total = to[i] + to[j]
        return None if total == 0 else {i: 100 * to[i] / total, j: 100 * to[j] / total}
    if basis == "split":
        if s.columns != "split":
            return None
        return {i: to[i], j: to[j], BLANK_NULL: to[BLANK_NULL], ABSTAIN: to[ABSTAIN]}
    # electorate : blancs, nuls et abstention regroupés
    if s.columns == "split":
        return {i: to[i], j: to[j], BLANK_OR_ABSTAIN: to[BLANK_NULL] + to[ABSTAIN]}
    if s.columns == "combined":
        return {i: to[i], j: to[j], BLANK_OR_ABSTAIN: to[BLANK_OR_ABSTAIN]}
    return None


def transfer_prior(
    election: int,
    from_candidate: str,
    pair: tuple[str, str],
    *,
    basis: Basis = "electorate",
    transfers: list[TransferSource] | tuple[TransferSource, ...] | None = None,
    registry: Registry | None = None,
    include_other: bool = False,
) -> TransferPrior:
    """Moyenne des sources disponibles pour l'électorat `from_candidate` vers `pair`.

    `basis` :
    - `electorate` (défaut) : en % de l'électorat de premier tour, blancs, nuls et
      abstention regroupés ; utilise les sources `split` et `combined` ;
    - `split` : idem avec blancs-nuls et abstention séparés ; sources `split` seulement ;
    - `expressed` : part de chaque finaliste parmi les exprimés de cet électorat ; toutes
      les sources (les autres bases sont ramenées aux deux finalistes).
    Seules les lignes complètes comptent. Les sources de base `other` (intentions du soir du
    premier tour, colonnes « ne se prononce pas »…) sont exclues sauf `include_other`. Lève
    `KeyError` si aucune source ne convient.
    """
    reg = registry or default_registry()
    finalists = _finalists(reg, election)
    if finalists is None:
        raise KeyError(f"pas de second tour connu en {election}")
    pair = tuple(pair)  # type: ignore[assignment]
    if len(pair) != 2 or set(pair) != set(finalists):
        raise KeyError(f"{election} : la paire {pair} n'est pas le second tour {finalists}")
    sources = default_transfers() if transfers is None else transfers
    per_source: dict[str, dict[str, float]] = {}
    for s in sources:
        if s.election != election or (s.base == "other" and not include_other):
            continue
        row = next((r for r in s.rows if r.from_ == from_candidate), None)
        if row is None:
            continue
        v = _vector(s, row, pair, basis)
        if v is not None:
            per_source[s.id] = v
    if not per_source:
        raise KeyError(f"{election} : aucune source ({basis}) pour l'électorat {from_candidate}")
    keys = next(iter(per_source.values())).keys()
    shares = {k: fmean(v[k] for v in per_source.values()) for k in keys}
    return TransferPrior(election, from_candidate, pair, basis, shares, per_source)
