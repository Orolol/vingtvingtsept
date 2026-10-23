"""Registre historique daté des candidatures (`data/registry/candidacies_<année>.yaml`).

Lot 1 de docs/12-prochaines-etapes.md. Pour une élection, le fichier liste :

- `persons` : toutes les personnes publiquement envisagées comme candidates pendant la
  période de collecte, y compris celles finalement absentes ou jamais sondées, avec leur
  parti et leur bloc au moment de l'élection et leur statut final (`official`,
  `declared_absent` : déclarée ou engagée dans une primaire puis absente de la liste
  officielle, `never_declared` : envisagée sans jamais se déclarer). Une personne absente de
  `candidates.yaml` porte son nom et respecte la convention `candidate_id_for` ;
- `processes` : primaires, congrès et consultations qui désignent un candidat parmi
  plusieurs participants, avec leurs tours, les qualifiés de chaque tour et le vainqueur.
  Règle : le résultat exclut les participants battus, et seulement eux, sauf engagement
  documenté (`commitments`) ;
- `events` : événements datés (mention publique, déclaration, entrée ou sortie de primaire,
  désignation, investiture, retrait, renoncement, dépôt des parrainages, liste officielle).

Chaque événement distingue la date où il se produit (`date`, `None` si seule la date de
publication est connue) de la date où l'information devient publique (`published`), avec la
base de cette date (`published_basis`) : `source` (date de l'article cité le plus ancien),
`wikipedia_revision` (horodatage de la première révision qui l'enregistre, borne par excès)
ou `imputed` (imputée, justification obligatoire). Les références sont des alias de
`references` ou des dictionnaires {url, title, outlet, published}.

`state_at(date)` rejoue, dans l'ordre de publication, les seuls événements publiés
strictement avant `date` (y compris ceux déduits des tours de primaire) : ajouter un
événement plus tardif ne peut pas modifier un état passé. États : `possible`, `declared`,
`designated`, `excluded` (par un processus ou par la liste officielle), `withdrawn`,
`undetermined` (personne exclue ou retirée de nouveau évoquée sans déclaration),
`official`. Une personne sans événement publié n'apparaît pas (aucune trace publique).
"""

from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vvs import paths
from vvs.registry.models import BLOCS, Bloc, candidate_id_for

EventType = Literal[
    "mention",
    "declaration",
    "primary_entry",
    "primary_exit",
    "designation",
    "investiture",
    "sponsorship_filing",
    "withdrawal",
    "renouncement",
    "official_list",
]
State = Literal[
    "possible", "declared", "designated", "excluded", "withdrawn", "undetermined", "official"
]
# États où la personne reste une candidature possible à la date (audit, lot 2).
ACTIVE_STATES: frozenset[str] = frozenset(
    {"possible", "declared", "designated", "undetermined", "official"}
)
FinalStatus = Literal["official", "declared_absent", "never_declared"]
PublishedBasis = Literal["source", "wikipedia_revision", "imputed"]
ProcessKind = Literal[
    "open_primary", "closed_primary", "members_vote", "congress", "council", "citizen_primary"
]

# Ordre d'application d'événements publiés le même jour (et à date d'événement égale).
TYPE_ORDER = {
    t: i
    for i, t in enumerate(
        [
            "mention",
            "declaration",
            "primary_entry",
            "sponsorship_filing",
            "primary_exit",
            "designation",
            "investiture",
            "withdrawal",
            "renouncement",
            "official_list",
        ]
    )
}
DECLARING = {"declaration", "primary_entry", "sponsorship_filing", "designation", "investiture"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_Model):
    url: str
    title: str | None = None
    outlet: str | None = None
    published: dt.date | None = None
    note: str | None = None


SourceRef = str | Source


class CollectionSource(_Model):
    """Source et période parcourues d'avance pour constituer la liste des personnes."""

    id: str
    kind: Literal["wikipedia_revisions", "wikipedia_page", "press", "official"]
    title: str
    url: str | None = None
    period: tuple[dt.date, dt.date] | None = None
    cadence: str | None = None
    note: str | None = None


class Collection(_Model):
    protocol: str
    collected_on: dt.date
    sources: list[CollectionSource] = Field(min_length=1)
    exclusions: str | None = None


class Person(_Model):
    id: str
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    party: str | None = None
    party_label: str | None = None
    bloc: Bloc
    final_status: FinalStatus
    notes: str | None = None


class Round(_Model):
    """Tour d'un processus : date du vote ou de la décision, publication du résultat."""

    date: dt.date
    published: dt.date
    qualified: list[str] = Field(min_length=1)
    sources: list[SourceRef] = Field(min_length=1)
    note: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Round:
        if self.published < self.date:
            raise ValueError(f"tour du {self.date} publié avant d'avoir lieu")
        return self


class Commitment(_Model):
    """Engagement documenté d'un non-participant à ne pas se présenter si l'un des `winners`
    l'emporte (règle des primaires élargie au seul cas documenté)."""

    person: str
    winners: list[str] = Field(min_length=1)
    published: dt.date
    sources: list[SourceRef] = Field(min_length=1)
    note: str | None = None


class Process(_Model):
    id: str
    name: str
    kind: ProcessKind
    organizers: list[str] = Field(min_length=1)
    participants: list[str] = Field(min_length=2)
    winner: str
    rounds: list[Round] = Field(min_length=1)
    commitments: list[Commitment] = Field(default_factory=list)
    events_yaml: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Process:
        where = f"processus {self.id}"
        if len(set(self.participants)) != len(self.participants):
            raise ValueError(f"{where} : participant en double")
        if self.winner not in self.participants:
            raise ValueError(f"{where} : vainqueur hors des participants")
        previous = set(self.participants)
        last = None
        for r in self.rounds:
            if last is not None and r.date < last:
                raise ValueError(f"{where} : tours non ordonnés")
            last = r.date
            q = set(r.qualified)
            if not q <= previous:
                raise ValueError(f"{where} : qualifiés du {r.date} hors du tour précédent")
            previous = q
        if self.rounds[-1].qualified != [self.winner]:
            raise ValueError(f"{where} : le dernier tour doit ne qualifier que le vainqueur")
        return self


class Event(_Model):
    id: str
    type: EventType
    person: str | None = None
    persons: list[str] | None = None  # official_list seulement
    process: str | None = None
    date: dt.date | None = None
    published: dt.date
    published_basis: PublishedBasis
    sources: list[SourceRef] = Field(min_length=1)
    events_yaml: str | None = None
    imputation: str | None = None
    note: str | None = None
    derived: bool = False  # tours de primaire : engendré par le chargeur, jamais écrit

    @model_validator(mode="after")
    def _consistent(self) -> Event:
        where = f"événement {self.id}"
        if self.date is not None and self.published < self.date:
            raise ValueError(f"{where} : publication {self.published} avant l'événement {self.date}")
        if self.type == "official_list":
            if not self.persons or self.person is not None:
                raise ValueError(f"{where} : une liste officielle porte `persons`, pas `person`")
        elif self.person is None or self.persons is not None:
            raise ValueError(f"{where} : `person` requis (et `persons` réservé aux listes)")
        if self.type in ("primary_entry", "primary_exit") and self.process is None:
            raise ValueError(f"{where} : `process` requis pour {self.type}")
        if self.published_basis == "imputed" and not self.imputation:
            raise ValueError(f"{where} : date de publication imputée sans justification")
        return self


class CandidaciesFile(_Model):
    election: int
    collection: Collection
    references: dict[str, Source] = Field(default_factory=dict)
    persons: list[Person] = Field(min_length=1)
    processes: list[Process] = Field(default_factory=list)
    events: list[Event] = Field(min_length=1)


@dataclass(frozen=True)
class PersonState:
    """État d'une personne à une date et les événements qui l'expliquent (ordre appliqué)."""

    person: str
    state: str
    since: dt.date
    reason: str
    trail: tuple[str, ...]
    returned: bool = False  # déclaration ou nouvelle mention après exclusion ou retrait


@dataclass
class CandidacyRegistry:
    election: int
    file: CandidaciesFile
    persons: dict[str, Person]
    processes: dict[str, Process]
    events: list[Event]  # explicites puis déduits, triés dans l'ordre d'application
    references: dict[str, Source] = field(default_factory=dict)

    def source(self, ref: SourceRef) -> Source:
        return self.references[ref] if isinstance(ref, str) else ref

    def published_before(self, date: dt.date) -> list[Event]:
        return [e for e in self.events if e.published < date]

    def first_trace(self, person: str) -> Event | None:
        return next((e for e in self.events if person in _targets(e)), None)

    def state_at(self, date: dt.date) -> dict[str, PersonState]:
        return state_at(self, date)


def _targets(e: Event) -> list[str]:
    return list(e.persons) if e.persons is not None else [e.person]


def _sort_key(e: Event) -> tuple:
    return (e.published, e.date or e.published, TYPE_ORDER[e.type], e.id)


def derive_process_events(p: Process) -> list[Event]:
    """Sorties de primaire à chaque tour et désignation du vainqueur au dernier tour."""
    out: list[Event] = []
    remaining = list(p.participants)
    for i, r in enumerate(p.rounds, start=1):
        last = i == len(p.rounds)
        for person in remaining:
            if person in r.qualified and not last:
                continue
            kind = "designation" if person == p.winner else "primary_exit"
            out.append(
                Event(
                    id=f"{p.id}__r{i}__{person}",
                    type=kind,
                    person=person,
                    process=p.id,
                    date=r.date,
                    published=r.published,
                    published_basis="source",
                    sources=list(r.sources),
                    events_yaml=p.events_yaml if kind == "designation" else None,
                    note=f"déduit du tour {i} de {p.id}",
                    derived=True,
                )
            )
        remaining = [x for x in remaining if x in r.qualified]
    return out


def _validate(f: CandidaciesFile, registry, events_yaml: dict[str, dict] | None) -> list[str]:
    from vvs.eval.configurations import LIST_PUBLICATION

    errors: list[str] = []
    e = f.election
    if e not in registry.elections:
        return [f"élection {e} absente d'elections.yaml"]
    first_round = registry.elections[e].round_(1).date
    official = [c.candidate_id for c in registry.elections[e].candidates]
    ids = [p.id for p in f.persons]
    persons = {p.id: p for p in f.persons}
    if len(ids) != len(set(ids)):
        errors.append("personne en double : " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    for p in f.persons:
        where = f"personne {p.id}"
        if p.id in registry.candidates:
            c = registry.candidates[p.id]
            if p.name is not None and p.name != c.name:
                errors.append(f"{where} : nom {p.name!r} différent de candidates.yaml")
        else:
            if not (p.name and p.first_name and p.last_name):
                errors.append(f"{where} : hors candidates.yaml, name/first_name/last_name requis")
            elif candidate_id_for(p.last_name, p.first_name) != p.id:
                errors.append(f"{where} : id attendu {candidate_id_for(p.last_name, p.first_name)}")
        if p.party is not None and p.party not in registry.parties:
            errors.append(f"{where} : parti {p.party!r} absent de parties.yaml")
        if p.party is None and not p.party_label:
            errors.append(f"{where} : ni party ni party_label")
        if p.bloc not in BLOCS:
            errors.append(f"{where} : bloc {p.bloc!r}")
        if (p.final_status == "official") != (p.id in official):
            errors.append(f"{where} : statut {p.final_status} incohérent avec la liste officielle")
        if p.id in official:
            cand = registry.elections[e].candidacy(p.id)
            if cand.bloc != p.bloc or cand.party != p.party:
                errors.append(f"{where} : parti/bloc différents d'elections.yaml")
    for o in official:
        if o not in persons:
            errors.append(f"candidat officiel {o} absent de persons")
    for alias, ref in f.references.items():
        if ref.published is None:
            errors.append(f"référence {alias} : date de publication manquante")

    def check_refs(where: str, refs: list[SourceRef]) -> list[Source]:
        out = []
        for r in refs:
            if isinstance(r, str):
                if r not in f.references:
                    errors.append(f"{where} : référence inconnue {r!r}")
                    continue
                out.append(f.references[r])
            else:
                out.append(r)
        return out

    procs = {p.id: p for p in f.processes}
    if len(procs) != len(f.processes):
        errors.append("processus en double")
    for p in f.processes:
        where = f"processus {p.id}"
        for x in [*p.participants, *(c.person for c in p.commitments)]:
            if x not in persons:
                errors.append(f"{where} : {x} absent de persons")
        for c in p.commitments:
            if not set(c.winners) <= set(p.participants):
                errors.append(f"{where} : engagement de {c.person} sur un non-participant")
            if c.person in p.participants:
                errors.append(f"{where} : {c.person} participe, l'engagement est implicite")
            check_refs(where, c.sources)
        for r in p.rounds:
            dates = [s.published for s in check_refs(where, r.sources) if s.published]
            if not dates or min(dates) != r.published:
                errors.append(f"{where} : tour du {r.date} publié le {r.published}, source "
                              f"la plus ancienne datée du {min(dates) if dates else None}")
            if r.published >= first_round:
                errors.append(f"{where} : tour publié après le premier tour présidentiel")
        check_refs(where, p.sources)
        if p.events_yaml and events_yaml is not None:
            ev = events_yaml.get(p.events_yaml)
            if ev is None or ev["election"] != e or ev["candidate"] != p.winner:
                errors.append(f"{where} : renvoi events.yaml {p.events_yaml!r} incohérent")
    ev_ids = [x.id for x in f.events]
    if len(ev_ids) != len(set(ev_ids)):
        errors.append("événement en double : " + ", ".join(sorted({i for i in ev_ids if ev_ids.count(i) > 1})))
    lists = [x for x in f.events if x.type == "official_list"]
    if len(lists) != 1:
        errors.append(f"{len(lists)} listes officielles au lieu d'une")
    for x in f.events:
        where = f"événement {x.id}"
        for t in _targets(x):
            if t not in persons:
                errors.append(f"{where} : {t} absent de persons")
        if x.process is not None:
            if x.process not in procs:
                errors.append(f"{where} : processus {x.process} inconnu")
            elif x.person not in procs[x.process].participants:
                errors.append(f"{where} : {x.person} ne participe pas à {x.process}")
        if x.published >= first_round:
            errors.append(f"{where} : publié après le premier tour")
        srcs = check_refs(where, x.sources)
        dates = [s.published for s in srcs if s.published is not None]
        if x.published_basis == "source" and (not dates or min(dates) != x.published):
            errors.append(f"{where} : `published` doit valoir la date de la source la plus ancienne")
        if x.published_basis == "wikipedia_revision":
            wiki = [s for s in srcs if "oldid=" in s.url]
            if not wiki or min(s.published for s in wiki) != x.published:
                errors.append(f"{where} : révision Wikipédia (oldid) datée de `published` requise")
        if x.events_yaml and events_yaml is not None:
            ev = events_yaml.get(x.events_yaml)
            if ev is None or ev["election"] != e or ev["candidate"] != x.person:
                errors.append(f"{where} : renvoi events.yaml {x.events_yaml!r} incohérent")
        if x.type == "official_list":
            if sorted(x.persons) != sorted(official):
                errors.append(f"{where} : liste différente d'elections.yaml")
            if e in LIST_PUBLICATION and x.date != LIST_PUBLICATION[e]:
                errors.append(f"{where} : date différente de LIST_PUBLICATION ({LIST_PUBLICATION[e]})")
    # Statut final cohérent avec les événements, et entrée datée de chaque participant.
    declaring = {x.person for x in f.events if x.type in DECLARING}
    declaring |= {pp for p in f.processes for pp in p.participants}
    for p in f.persons:
        if p.final_status == "never_declared" and p.id in declaring:
            errors.append(f"personne {p.id} : déclarée dans les événements, statut never_declared")
        if p.final_status == "declared_absent" and p.id not in declaring:
            errors.append(f"personne {p.id} : aucune déclaration, statut declared_absent")
        if not any(p.id in _targets(x) for x in f.events if x.type != "official_list"):
            errors.append(f"personne {p.id} : aucun événement (première trace manquante)")
    for p in f.processes:
        first = p.rounds[0].date
        for person in p.participants:
            entries = [x for x in f.events if x.type == "primary_entry" and x.process == p.id
                       and x.person == person]
            if not entries:
                errors.append(f"processus {p.id} : entrée de {person} non datée")
            elif min(x.published for x in entries) > first:
                errors.append(f"processus {p.id} : entrée de {person} publiée après le 1er tour")
    return errors


def load_candidacies(
    election: int,
    path: Path | None = None,
    registry=None,
    events_path: Path | None = None,
) -> CandidacyRegistry:
    """Lit et valide `candidacies_<election>.yaml`. Lève ValueError avec tous les problèmes."""
    from vvs.registry import default_registry

    reg = registry or default_registry()
    path = Path(path or paths.REGISTRY / f"candidacies_{election}.yaml")
    f = CandidaciesFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if f.election != election:
        raise ValueError(f"{path} : élection {f.election}, attendue {election}")
    ev_yaml = None
    ep = Path(events_path or paths.REGISTRY / "events.yaml")
    if ep.exists():
        raw = yaml.safe_load(ep.read_text(encoding="utf-8"))
        ev_yaml = {r["id"]: r for r in raw.get("events", [])}
    errors = _validate(f, reg, ev_yaml)
    if errors:
        raise ValueError(f"{path.name} incohérent :\n- " + "\n- ".join(errors))
    return assemble(f)


def assemble(f: CandidaciesFile) -> CandidacyRegistry:
    """Registre interrogeable : événements explicites et déduits des processus, triés dans
    l'ordre d'application. Sans validation croisée (voir `load_candidacies`)."""
    derived = [d for p in f.processes for d in derive_process_events(p)]
    return CandidacyRegistry(
        election=f.election,
        file=f,
        persons={p.id: p for p in f.persons},
        processes={p.id: p for p in f.processes},
        events=sorted([*f.events, *derived], key=_sort_key),
        references=dict(f.references),
    )


@functools.cache
def default_candidacies(election: int) -> CandidacyRegistry:
    return load_candidacies(election)


def state_at(cr: CandidacyRegistry, date: dt.date) -> dict[str, PersonState]:
    """État de chaque personne déjà tracée, d'après les seuls événements publiés avant `date`.

    Règles (dans l'ordre de publication) :
    - `mention` : `possible` si aucune trace ; `undetermined` après exclusion ou retrait
      (retour évoqué) ; sans effet sinon ;
    - `declaration`, `primary_entry`, `sponsorship_filing` : `declared` (retour si la personne
      était exclue ou retirée ; sans effet sur une personne déjà désignée) ;
    - `designation`, `investiture` : `designated` ; la désignation issue d'un processus exclut
      les participants battus qui ne l'étaient pas encore et les non-participants dont
      l'engagement documenté vise ce vainqueur, si cet engagement est déjà publié ;
    - `primary_exit` : `excluded` ; `withdrawal`, `renouncement` : `withdrawn` ;
    - `official_list` : `official` pour les inscrits, `excluded` pour toute autre personne
      encore active.
    """
    states: dict[str, PersonState] = {}

    def put(person, state, e, reason, returned=False):
        old = states.get(person)
        trail = (*old.trail, e.id) if old else (e.id,)
        keep_returned = returned or (old.returned if old else False)
        states[person] = PersonState(person, state, e.published, reason, trail, keep_returned)

    for e in cr.published_before(date):
        if e.type == "official_list":
            listed = set(e.persons)
            for person in e.persons:
                put(person, "official", e, "liste officielle")
            for person, s in list(states.items()):
                if person not in listed and s.state in ACTIVE_STATES:
                    put(person, "excluded", e, "absent de la liste officielle")
            continue
        person = e.person
        cur = states.get(person)
        cur_state = cur.state if cur else None
        if e.type == "mention":
            if cur is None:
                put(person, "possible", e, "mention publique")
            elif cur_state in ("excluded", "withdrawn"):
                put(person, "undetermined", e, "de nouveau évoqué après exclusion ou retrait", True)
            else:
                states[person] = PersonState(person, cur_state, cur.since, cur.reason,
                                             (*cur.trail, e.id), cur.returned)
        elif e.type in ("declaration", "primary_entry", "sponsorship_filing"):
            if cur_state in ("designated", "official"):
                states[person] = PersonState(person, cur_state, cur.since, cur.reason,
                                             (*cur.trail, e.id), cur.returned)
            else:
                back = cur_state in ("excluded", "withdrawn", "undetermined")
                reason = {"declaration": "déclaration", "primary_entry": f"participe à {e.process}",
                          "sponsorship_filing": "dépôt des parrainages"}[e.type]
                put(person, "declared", e, reason, back)
        elif e.type in ("designation", "investiture"):
            put(person, "designated", e, f"désigné ({e.process or e.type})")
            if e.type == "designation" and e.process is not None:
                proc = cr.processes[e.process]
                for other in proc.participants:
                    s = states.get(other)
                    if other != person and (s is None or s.state in ACTIVE_STATES):
                        put(other, "excluded", e, f"battu à {proc.id}")
                for c in proc.commitments:
                    if person in c.winners and c.published < date:
                        s = states.get(c.person)
                        if s is None or s.state in ACTIVE_STATES:
                            put(c.person, "excluded", e, f"engagement sur {proc.id}")
        elif e.type == "primary_exit":
            put(person, "excluded", e, f"éliminé de {e.process}")
        elif e.type in ("withdrawal", "renouncement"):
            put(person, "withdrawn", e, "retrait" if e.type == "withdrawal" else "renoncement")
    return states
