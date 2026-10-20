"""Scénarios de candidats 2027 : data/registry/scenarios_2027.yaml (voir docs/04-modele.md § 3).

Le fichier décrit des dimensions indépendantes (états exclusifs dont les probabilités somment
à 1), des petits candidats à présence de Bernoulli indépendante et des contraintes explicites.
`enumerate_configurations` en tire la liste des configurations de candidats, avec leur
probabilité renormalisée après retrait des combinaisons impossibles.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from vvs import paths
from vvs.registry.loader import Registry, default_registry
from vvs.registry.models import ID_RE, Bloc, _Model

# Tolérance sur la somme des probabilités d'une dimension
PROBABILITY_TOLERANCE = 1e-6
# Au-delà, `enumerate_configurations` refuse d'énumérer : regrouper ou élaguer
MAX_CONFIGURATIONS = 200_000

SmallMode = Literal["enumerate", "bloc", "marginal"]


class ScenarioError(ValueError):
    """Incohérence du fichier de scénarios ; le message liste tous les problèmes."""


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


# --- modèles


class State(_Model):
    id: str
    probability: float = Field(ge=0, le=1)
    description: str
    candidates: list[str] = Field(default_factory=list)
    # candidats non identifiés, connus seulement par leur bloc
    placeholders: list[Bloc] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class Dimension(_Model):
    id: str
    question: str
    review: list[str] = Field(default_factory=list)
    states: list[State] = Field(min_length=2)

    @model_validator(mode="after")
    def _consistent(self) -> Dimension:
        ids = [s.id for s in self.states]
        if len(ids) != len(set(ids)):
            raise ValueError(f"dimension {self.id} : état en double")
        total = sum(s.probability for s in self.states)
        if abs(total - 1) > PROBABILITY_TOLERANCE:
            raise ValueError(f"dimension {self.id} : les probabilités somment à {total:.6f}")
        for s in self.states:
            if len(s.candidates) != len(set(s.candidates)):
                raise ValueError(f"dimension {self.id}, état {s.id} : candidat en double")
        return self

    def state(self, state_id: str) -> State:
        return next(s for s in self.states if s.id == state_id)

    def candidate_ids(self) -> set[str]:
        return {c for s in self.states for c in s.candidates}


class SmallCandidate(_Model):
    candidate_id: str
    probability: float = Field(ge=0, le=1)
    bloc: Bloc
    rationale: str
    sources: list[str] = Field(default_factory=list)


class Constraint(_Model):
    """Quand tous les états de `when` sont actifs : `absent` forcés absents, ou combinaison
    impossible (`impossible: true`)."""

    id: str
    when: dict[str, str] = Field(min_length=1)
    absent: list[str] = Field(default_factory=list)
    impossible: bool = False
    rationale: str

    @model_validator(mode="after")
    def _one_effect(self) -> Constraint:
        if self.impossible == bool(self.absent):
            raise ValueError(f"contrainte {self.id} : soit `absent`, soit `impossible: true`")
        return self

    def applies(self, states: Mapping[str, str]) -> bool:
        return all(states.get(d) == s for d, s in self.when.items())


class Revision(_Model):
    date: dt.date
    author: str
    summary: str
    rationale: str


class ScenariosFile(_Model):
    election: int
    as_of: dt.date
    references: dict[str, str] = Field(default_factory=dict)
    dimensions: list[Dimension] = Field(min_length=1)
    small_candidates: list[SmallCandidate] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    revisions: list[Revision] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ScenariosFile:
        errors = _check_file(self)
        if errors:
            raise ValueError("\n- " + "\n- ".join(errors))
        return self

    def dimension(self, dim_id: str) -> Dimension:
        return next(d for d in self.dimensions if d.id == dim_id)

    def small(self, candidate_id: str) -> SmallCandidate:
        return next(s for s in self.small_candidates if s.candidate_id == candidate_id)

    def candidate_ids(self) -> set[str]:
        ids = {s.candidate_id for s in self.small_candidates}
        for d in self.dimensions:
            ids |= d.candidate_ids()
        return ids


def _check_file(f: ScenariosFile) -> list[str]:
    """Validations internes au fichier, toutes collectées."""
    errors: list[str] = []

    dim_ids = [d.id for d in f.dimensions]
    if len(dim_ids) != len(set(dim_ids)):
        errors.append("dimensions : id en double")
    owner: dict[str, str] = {}
    for d in f.dimensions:
        if not ID_RE.match(d.id):
            errors.append(f"dimension : id mal formé {d.id!r}")
        for cid in d.candidate_ids():
            if cid in owner and owner[cid] != d.id:
                errors.append(f"{cid} figure dans deux dimensions ({owner[cid]}, {d.id})")
            owner.setdefault(cid, d.id)

    small_ids = [s.candidate_id for s in f.small_candidates]
    if len(small_ids) != len(set(small_ids)):
        errors.append("small_candidates : candidat en double")
    for cid in small_ids:
        if cid in owner:
            errors.append(f"{cid} est à la fois petit candidat et dans la dimension {owner[cid]}")

    known = set(owner) | set(small_ids)
    dims = {d.id: d for d in f.dimensions}
    constraint_ids = [c.id for c in f.constraints]
    if len(constraint_ids) != len(set(constraint_ids)):
        errors.append("constraints : id en double")
    for c in f.constraints:
        where = f"contrainte {c.id}"
        for dim_id, state_id in c.when.items():
            if dim_id not in dims:
                errors.append(f"{where} : dimension inconnue {dim_id!r}")
            elif state_id not in {s.id for s in dims[dim_id].states}:
                errors.append(f"{where} : état inconnu {dim_id}.{state_id}")
            elif set(c.absent) & set(dims[dim_id].state(state_id).candidates):
                errors.append(
                    f"{where} : {sorted(set(c.absent) & set(dims[dim_id].state(state_id).candidates))} "
                    f"à la fois présent(s) dans {dim_id}.{state_id} et forcé(s) absent(s)"
                )
        for cid in c.absent:
            if cid not in known:
                errors.append(f"{where} : {cid} n'est ni petit candidat ni dans une dimension")
        if c.impossible and len(c.when) < 2:
            errors.append(f"{where} : un état seul impossible doit avoir une probabilité nulle")

    def check_sources(where: str, sources: list[str]) -> None:
        for s in sources:
            if not _is_url(s) and s not in f.references:
                errors.append(f"{where} : source ni URL ni alias de references : {s!r}")

    for d in f.dimensions:
        for s in d.states:
            check_sources(f"{d.id}.{s.id}", s.sources)
    for s in f.small_candidates:
        check_sources(s.candidate_id, s.sources)

    dates = [r.date for r in f.revisions]
    if dates != sorted(dates):
        errors.append("revisions : dates non croissantes")
    if f.as_of != dates[-1]:
        errors.append(f"as_of {f.as_of} != date de la dernière révision {dates[-1]}")
    return errors


# --- chargement


def default_path() -> Path:
    return paths.REGISTRY / "scenarios_2027.yaml"


def _cross_check(f: ScenariosFile, reg: Registry) -> list[str]:
    errors: list[str] = []
    for cid in sorted(f.candidate_ids()):
        if cid not in reg.candidates:
            errors.append(f"{cid} absent de candidates.yaml")
    for s in f.small_candidates:
        if s.candidate_id not in reg.candidates:
            continue
        try:
            bloc = reg.bloc_of(s.candidate_id, f.election)
        except KeyError as e:
            errors.append(f"{s.candidate_id} : {e}")
            continue
        if bloc != s.bloc:
            errors.append(f"{s.candidate_id} : bloc {s.bloc} != bloc du registre {bloc}")
    return errors


def load_scenarios(path: Path | None = None, registry: Registry | None = None) -> ScenariosFile:
    """Charge et valide le fichier de scénarios, y compris contre le registre."""
    path = default_path() if path is None else Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ScenarioError(f"{path.name} : document YAML vide ou mal formé")
    try:
        f = ScenariosFile.model_validate(data)
    except ValueError as e:
        raise ScenarioError(f"{path.name} invalide : {e}") from e
    errors = _cross_check(f, registry or default_registry())
    if errors:
        raise ScenarioError("scénarios incohérents avec le registre :\n- " + "\n- ".join(errors))
    return f


# --- énumération


@dataclass(frozen=True)
class Configuration:
    """Une configuration de premier tour.

    `candidates` : candidats présents à coup sûr dans cette configuration. En mode `bloc` ou
    `marginal`, les petits candidats non énumérés figurent dans `small_presence` avec leur
    probabilité de présence conditionnelle à la configuration ; en mode `bloc`,
    `small_blocs` liste les blocs où au moins un petit candidat est présent.
    """

    states: Mapping[str, str]
    candidates: frozenset[str]
    probability: float
    placeholders: tuple[str, ...] = ()
    small_blocs: frozenset[str] = frozenset()
    small_presence: Mapping[str, float] = field(default_factory=dict)

    def presence(self, candidate_id: str) -> float:
        if candidate_id in self.candidates:
            return 1.0
        return self.small_presence.get(candidate_id, 0.0)


@dataclass(frozen=True)
class Enumeration(Sequence[Configuration]):
    """Configurations triées par probabilité décroissante, renormalisées à 1.

    `impossible_mass` : masse retirée par les contraintes ; `pruned_mass` : masse élaguée
    sous `min_probability` (toutes deux en probabilité brute, avant renormalisation).
    """

    configurations: tuple[Configuration, ...]
    mode: SmallMode
    impossible_mass: float
    pruned_mass: float

    def __len__(self) -> int:
        return len(self.configurations)

    def __getitem__(self, i):  # type: ignore[override]
        return self.configurations[i]

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self.configurations)

    def presence(self) -> dict[str, float]:
        """Probabilité de présence au premier tour de chaque candidat nommé."""
        out: dict[str, float] = {}
        for c in self.configurations:
            for cid in c.candidates | set(c.small_presence):
                out[cid] = out.get(cid, 0.0) + c.probability * c.presence(cid)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _small_probs(f: ScenariosFile, absent: set[str]) -> dict[str, float]:
    return {
        s.candidate_id: 0.0 if s.candidate_id in absent else s.probability
        for s in f.small_candidates
    }


def _enumerate_small(
    probs: dict[str, float], base: float, min_probability: float
) -> tuple[list[tuple[frozenset[str], float]], float]:
    """Sous-ensembles présents (énumération en profondeur, élagage sous le seuil)."""
    order = sorted(probs)
    out: list[tuple[frozenset[str], float]] = []
    pruned = 0.0

    def walk(i: int, present: tuple[str, ...], p: float) -> None:
        nonlocal pruned
        if p < min_probability:
            pruned += p
            return
        if i == len(order):
            out.append((frozenset(present), p))
            return
        cid = order[i]
        q = probs[cid]
        if q > 0:
            walk(i + 1, (*present, cid), p * q)
        if q < 1:
            walk(i + 1, present, p * (1 - q))

    walk(0, (), base)
    return out, pruned


def enumerate_configurations(
    scenarios: ScenariosFile | None = None,
    *,
    small: SmallMode = "enumerate",
    min_probability: float = 0.0,
    max_configurations: int = MAX_CONFIGURATIONS,
) -> Enumeration:
    """Produit des dimensions et des petits candidats, contraintes appliquées, renormalisé.

    `small` règle le traitement des petits candidats :
    - `enumerate` : chaque petit candidat est une Bernoulli énumérée (2^n sous-ensembles par
      combinaison de dimensions, à élaguer avec `min_probability`) ;
    - `bloc` : les petits candidats sont regroupés par bloc ; on énumère seulement « au
      moins un petit candidat du bloc présent », et `small_presence` porte la probabilité
      conditionnelle de chacun (p_i / P(au moins un)) ;
    - `marginal` : on n'énumère que les dimensions ; `small_presence` porte la probabilité
      de présence de chaque petit candidat dans la configuration.

    `min_probability` élague les branches dont la probabilité brute passe sous le seuil
    (la masse élaguée est rapportée dans `pruned_mass`). Lève `ScenarioError` si plus de
    `max_configurations` configurations sont produites.
    """
    f = load_scenarios() if scenarios is None else scenarios
    out: list[Configuration] = []
    impossible = pruned = 0.0

    def emit(c: Configuration) -> None:
        out.append(c)
        if len(out) > max_configurations:
            raise ScenarioError(
                f"plus de {max_configurations} configurations : utiliser small='bloc' ou "
                "'marginal', ou relever min_probability"
            )

    for combo in itertools.product(*(d.states for d in f.dimensions)):
        p = math.prod(s.probability for s in combo)
        if p == 0:
            continue
        states = {d.id: s.id for d, s in zip(f.dimensions, combo)}
        active = [c for c in f.constraints if c.applies(states)]
        named = frozenset(cid for s in combo for cid in s.candidates)
        absent = {cid for c in active for cid in c.absent}
        if any(c.impossible for c in active) or named & absent:
            impossible += p
            continue
        if p < min_probability:
            pruned += p
            continue
        placeholders = tuple(b for s in combo for b in s.placeholders)
        probs = _small_probs(f, absent)

        if small == "marginal":
            presence = {cid: q for cid, q in probs.items() if q > 0}
            emit(Configuration(states, named, p, placeholders, small_presence=presence))

        elif small == "bloc":
            groups: dict[str, dict[str, float]] = {}
            for s in f.small_candidates:
                if probs[s.candidate_id] > 0:
                    groups.setdefault(s.bloc, {})[s.candidate_id] = probs[s.candidate_id]
            any_present = {b: 1 - math.prod(1 - q for q in g.values()) for b, g in groups.items()}
            blocs = sorted(any_present)
            for flags in itertools.product((True, False), repeat=len(blocs)):
                q = p * math.prod(
                    any_present[b] if on else 1 - any_present[b] for b, on in zip(blocs, flags)
                )
                if q == 0:
                    continue
                if q < min_probability:
                    pruned += q
                    continue
                on_blocs = frozenset(b for b, on in zip(blocs, flags) if on)
                presence = {
                    cid: qi / any_present[b] for b in on_blocs for cid, qi in groups[b].items()
                }
                emit(Configuration(states, named, q, placeholders, on_blocs, presence))

        elif small == "enumerate":
            subsets, lost = _enumerate_small(probs, p, min_probability)
            pruned += lost
            for present, q in subsets:
                emit(Configuration(states, named | present, q, placeholders))

        else:
            raise ValueError(f"mode de petits candidats inconnu : {small!r}")

    total = sum(c.probability for c in out)
    if total <= 0:
        raise ScenarioError("aucune configuration possible après contraintes et élagage")
    normalized = sorted(
        (
            Configuration(
                c.states,
                c.candidates,
                c.probability / total,
                c.placeholders,
                c.small_blocs,
                c.small_presence,
            )
            for c in out
        ),
        key=lambda c: -c.probability,
    )
    return Enumeration(tuple(normalized), small, impossible, pruned)
