"""Configuration de candidats « connue » à une date, pour rejouer une élection passée.

Le backtest (docs/07-evaluation.md) ne doit pas donner au modèle une information qu'il n'avait
pas. La liste officielle des candidats n'est connue qu'à sa publication par le Conseil
constitutionnel, environ cinq semaines avant le premier tour (`LIST_PUBLICATION`). Avant cette
date, la configuration est déduite des seuls sondages publiés.

Règle (`known_configuration`), approximation documentée :

1. Après la publication de la liste : les candidats officiels testés par au moins un sondage
   terminé avant `asof`.
2. Avant : les sondages terminés dans les 30 jours qui précèdent `asof` (fenêtre élargie par pas
   de 30 jours jusqu'à 180 s'il n'y en a aucun). Chaque sondage pèse 1, réparti également sur
   ses hypothèses. Le poids d'un candidat est la part des hypothèses (ainsi pondérées) qui le
   testent. On parcourt les candidats par poids décroissant (puis part moyenne décroissante) et
   on garde un candidat s'il a été testé au moins une fois avec chacun des candidats déjà
   gardés : deux candidats jamais testés ensemble sont des alternatives (Juppé ou Sarkozy avant
   la primaire de la droite, Valls ou Hamon avant celle de la gauche), on garde le plus testé.
   Les primaires se reflètent donc d'elles-mêmes : avant le 27 novembre 2016, le candidat LR
   de la configuration 2017 est Juppé ; après, Fillon.

La consigne initiale proposait de filtrer les candidats testés sur la liste officielle finale ;
avant la publication de la liste, ce filtre donnerait au modèle le nom du vainqueur des
primaires (Hamon, testé dès décembre 2016 parmi d'autres). On ne l'applique donc qu'après la
publication.

Correspondance avec les candidats officiels, pour le score (`to_official`) : un candidat de la
configuration qui n'est pas candidat officiel (Juppé, Bayrou 2017, Montebourg 2022) est un
« tenant lieu ». Sa part va, dans cet ordre : aux candidats officiels de son parti absents de la
configuration (Juppé → Fillon) ; sinon aux candidats officiels de son bloc absents de la
configuration (Jadot 2016 → Hamon), à parts égales ; sinon aux candidats officiels de son bloc
présents, au prorata de leur part dans le tirage (Bayrou 2017 → Macron) ; sinon elle est
redistribuée sur tous (indépendance des alternatives). Un candidat officiel qui ne reçoit
rien n'a pas de prévision (`forecast` faux) : il n'entre pas dans les scores par candidat, sa
part de résultat est rapportée à part (« part non prévue »).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from vvs.model.data import load_polls
from vvs.registry import Registry, default_registry

# Décision du Conseil constitutionnel arrêtant la liste des candidats au premier tour. Dates
# relevées de mémoire, non vérifiées sur conseil-constitutionnel.fr (à reporter dans
# elections.yaml avec leur source) ; 2002 est la plus incertaine. Seuls les horizons de 31 à
# 50 jours environ en dépendent : aux horizons du backtest (180, 90, 30, 7), la liste est
# soit inconnue, soit publiée depuis plusieurs jours.
LIST_PUBLICATION = {
    2002: dt.date(2002, 3, 21),
    2007: dt.date(2007, 3, 19),
    2012: dt.date(2012, 3, 19),
    2017: dt.date(2017, 3, 18),
    2022: dt.date(2022, 3, 7),
}
LIST_LEAD_DAYS = 33  # élections sans date connue : liste publiée ~33 jours avant le vote

WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 180


def list_publication(election: int, registry: Registry | None = None) -> dt.date:
    if election in LIST_PUBLICATION:
        return LIST_PUBLICATION[election]
    reg = registry or default_registry()
    return reg.elections[election].round_(1).date - dt.timedelta(days=LIST_LEAD_DAYS)


@dataclass
class KnownConfig:
    election: int
    asof: dt.date
    candidates: list[str]
    source: str  # official_list ou polls
    window_days: int
    n_polls: int
    weights: dict[str, float] = field(default_factory=dict)  # part des hypothèses testant c
    alternatives: dict[str, str] = field(default_factory=dict)  # écarté -> gardé exclusif

    def describe(self) -> str:
        return (
            f"{self.election} au {self.asof} ({self.source}, {self.n_polls} sondages sur "
            f"{self.window_days} j) : {', '.join(self.candidates)}"
        )


def _hypotheses(df: pl.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """poll_id -> hypothesis_id -> {candidate_id: share}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for pid, hid, cid, share in df.select(
        "poll_id", "hypothesis_id", "candidate_id", "share"
    ).iter_rows():
        out.setdefault(pid, {}).setdefault(hid, {})[cid] = share
    return out


def config_from_polls(hyps: dict[str, dict[str, dict[str, float]]]) -> tuple[list[str], dict, dict]:
    """Configuration déduite d'un ensemble de sondages (règle 2 du module).

    Rend (candidats gardés, poids, alternatives écartées)."""
    weight: dict[str, float] = {}
    share_sum: dict[str, float] = {}
    share_n: dict[str, int] = {}
    cooc: dict[frozenset, float] = {}
    n = len(hyps)
    for poll in hyps.values():
        h = 1.0 / len(poll)
        for shares in poll.values():
            cands = sorted(shares)
            for c in cands:
                weight[c] = weight.get(c, 0.0) + h / n
                share_sum[c] = share_sum.get(c, 0.0) + shares[c]
                share_n[c] = share_n.get(c, 0) + 1
            for i, a in enumerate(cands):
                for b in cands[i + 1 :]:
                    key = frozenset((a, b))
                    cooc[key] = cooc.get(key, 0.0) + h
    order = sorted(weight, key=lambda c: (-weight[c], -share_sum[c] / share_n[c], c))
    kept: list[str] = []
    alternatives: dict[str, str] = {}
    for c in order:
        clash = next((k for k in kept if cooc.get(frozenset((c, k)), 0.0) == 0.0), None)
        if clash is None:
            kept.append(c)
        else:
            alternatives[c] = clash
    return kept, weight, alternatives


def known_configuration(
    election: int,
    asof: dt.date,
    registry: Registry | None = None,
    polls: pl.DataFrame | None = None,
) -> KnownConfig:
    """Configuration connue à `asof` (voir le docstring du module).

    `polls` : table longue de `vvs.model.data.load_polls` (rechargée sinon), filtrée ici à
    `fieldwork_end <= asof`."""
    reg = registry or default_registry()
    df = load_polls(election) if polls is None else polls
    df = df.filter(pl.col("fieldwork_end") <= asof)
    if df.is_empty():
        raise ValueError(f"aucun sondage pour {election} avant le {asof}")
    official = [c.candidate_id for c in reg.elections[election].candidates]
    if asof >= list_publication(election, reg):
        tested = set(df["candidate_id"].unique().to_list())
        recent = df.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=WINDOW_DAYS))
        cands = [c for c in official if c in tested]
        return KnownConfig(
            election=election,
            asof=asof,
            candidates=cands,
            source="official_list",
            window_days=WINDOW_DAYS,
            n_polls=recent["poll_id"].n_unique(),
        )
    window = WINDOW_DAYS
    while True:
        sub = df.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=window))
        if sub.height or window >= MAX_WINDOW_DAYS:
            break
        window += WINDOW_DAYS
    if sub.is_empty():
        raise ValueError(f"aucun sondage pour {election} dans les {window} jours avant {asof}")
    kept, weight, alternatives = config_from_polls(_hypotheses(sub))
    return KnownConfig(
        election=election,
        asof=asof,
        candidates=kept,
        source="polls",
        window_days=window,
        n_polls=sub["poll_id"].n_unique(),
        weights=weight,
        alternatives=alternatives,
    )


# --- correspondance avec les candidats officiels


@dataclass
class OfficialMap:
    """Correspondance configuration -> candidats officiels (voir `to_official`)."""

    config: list[str]
    official: list[str]
    official_bloc: list[str]
    # pour chaque candidat de la configuration : ("self" | "split" | "prorata" | "drop", cibles)
    rules: dict[str, tuple[str, list[str]]]
    forecast: np.ndarray  # (O,) bool : le candidat officiel reçoit une part

    def apply(self, shares: np.ndarray) -> np.ndarray:
        """Parts (S, K) sur la configuration -> parts (S, O) sur les candidats officiels."""
        shares = np.atleast_2d(shares)
        S = shares.shape[0]
        out = np.zeros((S, len(self.official)))
        oidx = {c: i for i, c in enumerate(self.official)}
        kidx = {c: k for k, c in enumerate(self.config)}
        dropped = np.zeros(S)
        for c, (rule, targets) in self.rules.items():
            if rule == "self":
                out[:, oidx[c]] += shares[:, kidx[c]]
        for c, (rule, targets) in self.rules.items():
            x = shares[:, kidx[c]]
            if rule == "split":
                for t in targets:
                    out[:, oidx[t]] += x / len(targets)
            elif rule == "prorata":
                cols = [oidx[t] for t in targets]
                base = np.stack([shares[:, kidx[t]] for t in targets], axis=1)
                w = base / base.sum(1, keepdims=True)
                out[:, cols] += x[:, None] * w
            elif rule == "drop":
                dropped += x
        total = out.sum(1, keepdims=True)
        return out / np.where(total > 0, total, 1.0)


def official_map(
    config: list[str], election: int, registry: Registry | None = None
) -> OfficialMap:
    reg = registry or default_registry()
    e = reg.elections[election]
    official = [c.candidate_id for c in e.candidates]
    party = {c.candidate_id: c.party for c in e.candidates}
    obloc = {c: reg.bloc_of(c, election) for c in official}
    absent = [c for c in official if c not in config]
    rules: dict[str, tuple[str, list[str]]] = {}
    for c in config:
        if c in obloc:
            rules[c] = ("self", [c])
            continue
        cand = reg.candidates.get(c)
        p = cand.party_at(election) if cand is not None else None
        b = reg.bloc_of(c, election)
        same_party = [o for o in absent if p is not None and party[o] == p]
        same_bloc_absent = [o for o in absent if obloc[o] == b]
        same_bloc_present = [o for o in official if o in config and obloc[o] == b]
        if same_party:
            rules[c] = ("split", same_party)
        elif same_bloc_absent:
            rules[c] = ("split", same_bloc_absent)
        elif same_bloc_present:
            rules[c] = ("prorata", same_bloc_present)
        else:
            rules[c] = ("drop", [])
    receiving = {t for rule, ts in rules.values() if rule != "drop" for t in ts}
    return OfficialMap(
        config=list(config),
        official=official,
        official_bloc=[obloc[c] for c in official],
        rules=rules,
        forecast=np.array([c in receiving for c in official]),
    )


def closest_hypothesis(
    hyps: dict[str, dict[str, float]], config: list[str]
) -> tuple[str, int]:
    """Hypothèse d'un sondage la plus proche d'une configuration : moins de candidats manquants
    ou en trop, puis moins de manquants. Rend (hypothesis_id, distance)."""
    target = set(config)
    best = None
    for hid, shares in hyps.items():
        cands = set(shares)
        missing, extra = len(target - cands), len(cands - target)
        key = (missing + extra, missing, hid)
        if best is None or key < best:
            best = key
    return best[2], best[0]
