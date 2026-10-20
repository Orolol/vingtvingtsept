"""Chargement du registre et validations croisées entre fichiers YAML."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from unidecode import unidecode

from vvs import paths
from vvs.registry.models import (
    ID_RE,
    BlocDefinition,
    BlocsFile,
    BoundaryCase,
    Candidate,
    CandidatesFile,
    Election,
    ElectionsFile,
    PartiesFile,
    Party,
    Pollster,
    PollstersFile,
    candidate_id_for,
)


class RegistryError(ValueError):
    """Incohérence entre fichiers du registre ; le message liste tous les problèmes."""


def normalize_label(label: str) -> str:
    """Forme canonique d'un libellé : ASCII, minuscules, séparateurs réduits à un espace."""
    return re.sub(r"[^a-z0-9]+", " ", unidecode(label).lower()).strip()


@dataclass(frozen=True)
class Registry:
    candidates: dict[str, Candidate]
    parties: dict[str, Party]
    blocs: dict[str, BlocDefinition]
    boundary_cases: list[BoundaryCase]
    elections: dict[int, Election]
    pollsters: dict[str, Pollster]
    _pollster_index: dict[str, str] = field(default_factory=dict, repr=False)

    def bloc_of(self, candidate_id: str, election: int) -> str:
        """Bloc d'un candidat à une élection.

        Ordre : liste officielle de elections.yaml, puis cas limite de blocs.yaml visant ce
        candidat et cette élection, puis bloc par défaut de son parti cette année-là.
        """
        if candidate_id not in self.candidates:
            raise KeyError(f"candidat inconnu : {candidate_id}")
        e = self.elections.get(election)
        if e is not None and (c := e.candidacy(candidate_id)) is not None:
            return c.bloc
        for case in self.boundary_cases:
            if candidate_id in case.candidate_ids and election in case.elections:
                return case.bloc
        party = self.candidates[candidate_id].party_at(election)
        if party is None:
            raise KeyError(f"{candidate_id} : aucun parti actif en {election}, bloc indéterminé")
        return self.parties[party].bloc

    def pollster_of(self, label: str) -> str:
        """Id d'institut pour un libellé (id, nom ou alias), à la casse et aux accents près."""
        key = normalize_label(label)
        try:
            return self._pollster_index[key]
        except KeyError:
            raise KeyError(f"institut inconnu : {label!r} (ajouter l'alias dans pollsters.yaml)")


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RegistryError(f"{path.name} : document YAML vide ou mal formé")
    return data


def _pollster_index(pollsters: dict[str, Pollster], errors: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for pid, p in pollsters.items():
        for label in (pid, p.name, *p.aliases):
            key = normalize_label(label)
            if key in index and not (index[key] == pid and label in (pid, p.name)):
                errors.append(
                    f"pollsters : libellé {label!r} redondant ou ambigu ({index[key]}, {pid})"
                )
            index.setdefault(key, pid)
    return index


def _cross_check(reg: Registry) -> list[str]:
    errors: list[str] = []
    cands, parties = reg.candidates, reg.parties

    for cid, c in cands.items():
        if not ID_RE.match(cid):
            errors.append(f"candidates : id mal formé {cid!r}")
        if cid != candidate_id_for(c.last_name, c.first_name):
            errors.append(
                f"candidates : {cid} ne suit pas nom_prenom "
                f"({candidate_id_for(c.last_name, c.first_name)})"
            )
        for a in c.parties:
            if a.party not in parties:
                errors.append(f"candidates : {cid} affilié à un parti inconnu {a.party!r}")

    for pid, p in parties.items():
        if not ID_RE.match(pid):
            errors.append(f"parties : id mal formé {pid!r}")
        if p.successor is not None and p.successor not in parties:
            errors.append(f"parties : {pid} a un successeur inconnu {p.successor!r}")

    for year, e in reg.elections.items():
        for c in e.candidates:
            where = f"elections {year} : {c.candidate_id}"
            if c.candidate_id not in cands:
                errors.append(f"{where} absent de candidates.yaml")
                continue
            if c.party not in parties:
                errors.append(f"{where} : parti inconnu {c.party!r}")
            elif c.party not in cands[c.candidate_id].parties_at(year):
                errors.append(f"{where} : {c.party} pas actif en {year} dans candidates.yaml")

    for case in reg.boundary_cases:
        where = f"blocs : cas {case.id}"
        if case.party is not None:
            if case.party not in parties:
                errors.append(f"{where} : parti inconnu {case.party!r}")
            elif parties[case.party].bloc != case.bloc:
                errors.append(f"{where} : bloc du parti {parties[case.party].bloc} != {case.bloc}")
        for cid in case.candidate_ids:
            if cid not in cands:
                errors.append(f"{where} : candidat inconnu {cid!r}")
        for year in case.elections:
            e = reg.elections.get(year)
            if e is None:
                errors.append(f"{where} : élection inconnue {year}")
                continue
            for c in e.candidates:
                targeted = c.candidate_id in case.candidate_ids or c.party == case.party
                if targeted and c.bloc != case.bloc:
                    errors.append(
                        f"{where} : elections {year} classe {c.candidate_id} en {c.bloc}, "
                        f"le cas limite dit {case.bloc}"
                    )

    for pid, p in reg.pollsters.items():
        if not ID_RE.match(pid):
            errors.append(f"pollsters : id mal formé {pid!r}")
        if p.successor is not None and p.successor not in reg.pollsters:
            errors.append(f"pollsters : {pid} a un successeur inconnu {p.successor!r}")
    return errors


def load_registry(root: Path | None = None) -> Registry:
    """Charge et valide les YAML du registre ; lève `RegistryError` à la moindre incohérence."""
    root = paths.REGISTRY if root is None else Path(root)
    candidates = CandidatesFile.model_validate(_read(root / "candidates.yaml")).candidates
    parties = PartiesFile.model_validate(_read(root / "parties.yaml")).parties
    blocs = BlocsFile.model_validate(_read(root / "blocs.yaml"))
    elections = ElectionsFile.model_validate(_read(root / "elections.yaml")).elections
    pollsters = PollstersFile.model_validate(_read(root / "pollsters.yaml")).pollsters

    errors: list[str] = []
    index = _pollster_index(pollsters, errors)
    reg = Registry(
        candidates=candidates,
        parties=parties,
        blocs=dict(blocs.blocs),
        boundary_cases=list(blocs.boundary_cases),
        elections=elections,
        pollsters=pollsters,
        _pollster_index=index,
    )
    errors += _cross_check(reg)
    if errors:
        raise RegistryError("registre incohérent :\n- " + "\n- ".join(errors))
    return reg


@functools.cache
def default_registry() -> Registry:
    """Registre de data/registry/, chargé une fois par processus."""
    return load_registry()


def bloc_of(candidate_id: str, election: int, registry: Registry | None = None) -> str:
    return (registry or default_registry()).bloc_of(candidate_id, election)


def pollster_of(label: str, registry: Registry | None = None) -> str:
    return (registry or default_registry()).pollster_of(label)
