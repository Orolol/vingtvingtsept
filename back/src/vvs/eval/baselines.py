"""Baselines du backtest (docs/07-evaluation.md, « Comparaisons de référence »).

Deux estimations ponctuelles, sur la configuration connue à la date (`configurations`) :

- `poll_mean` : moyenne simple des sondages terminés dans les `days` jours précédant asof
  (14 par défaut) ; fenêtre élargie par pas de 14 jours, jusqu'à 120, tant qu'un candidat de la
  configuration n'est testé par aucun sondage de la fenêtre.
- `last_by_pollster` : dernier sondage de chaque institut terminé dans les 60 jours précédant
  asof, moyenné entre instituts.

Règles communes, les mêmes que `vvs.curate.derived.poll_errors` : un sondage contribue une
seule hypothèse, la plus proche de la configuration ; les parts sont normalisées à 100 dans
l'hypothèse ; la moyenne d'un candidat porte sur les sondages qui le testent ; les vagues de
rolling sont éclaircies (`load_polls`). Le vecteur final est renormalisé sur la
configuration. Le modèle d'erreur η s'applique ensuite comme au modèle (même `ErrorModel`),
avec pour horizon l'âge moyen des sondages utilisés : la comparaison isole l'apport de
l'agrégation, pas celui du terme d'erreur.

Le modèle sans terme d'erreur (projection de la phase 2) n'est pas une baseline de ce module :
il vient directement des tirages θ à la semaine du vote (`backtest`).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from vvs.eval.configurations import closest_hypothesis
from vvs.model.data import election_date, load_polls
from vvs.registry import Registry, default_registry

MAX_WINDOW_DAYS = 120
MISSING_SHARE = 0.005  # part d'un candidat de la configuration jamais testé (fraction)


@dataclass
class PointForecast:
    config: list[str]
    shares: np.ndarray  # (K,) fractions, somme 1
    age_days: float  # âge moyen des sondages utilisés (jours avant le vote, milieu du terrain)
    n_polls: int
    window_days: int
    missing: list[str]


def _poll_rows(df: pl.DataFrame) -> list[dict]:
    """Un dict par sondage : méta-données et hypothèses {hid: {c: part}}."""
    polls: dict[str, dict] = {}
    for r in df.iter_rows(named=True):
        p = polls.setdefault(
            r["poll_id"],
            {
                "poll_id": r["poll_id"],
                "pollster": r["pollster"],
                "fieldwork_start": r["fieldwork_start"],
                "fieldwork_end": r["fieldwork_end"],
                "hyps": {},
            },
        )
        p["hyps"].setdefault(r["hypothesis_id"], {})[r["candidate_id"]] = r["share"]
    return list(polls.values())


def _average(polls: list[dict], config: list[str], vote: dt.date) -> tuple[dict, float]:
    sums: dict[str, list[float]] = {}
    ages = []
    for p in polls:
        hid, _ = closest_hypothesis(p["hyps"], config)
        shares = p["hyps"][hid]
        total = sum(shares.values())
        for c in config:
            if c in shares:
                sums.setdefault(c, []).append(shares[c] / total)
        mid = p["fieldwork_start"] + (p["fieldwork_end"] - p["fieldwork_start"]) / 2
        ages.append((vote - mid).days)
    return {c: float(np.mean(v)) for c, v in sums.items()}, float(np.mean(ages)) if ages else 0.0


def _finish(config, means, age, n, window) -> PointForecast:
    missing = [c for c in config if c not in means]
    s = np.array([means.get(c, MISSING_SHARE) for c in config])
    return PointForecast(list(config), s / s.sum(), age, n, window, missing)


def poll_mean(
    election: int,
    asof: dt.date,
    config: list[str],
    days: int = 14,
    registry: Registry | None = None,
    polls: pl.DataFrame | None = None,
) -> PointForecast:
    reg = registry or default_registry()
    vote = election_date(reg, election)
    df = load_polls(election, asof) if polls is None else polls.filter(pl.col("fieldwork_end") <= asof)
    rows = _poll_rows(df)
    window = days
    while True:
        lo = asof - dt.timedelta(days=window)
        sub = [p for p in rows if p["fieldwork_end"] > lo]
        means, age = _average(sub, config, vote)
        if all(c in means for c in config) or window >= MAX_WINDOW_DAYS:
            return _finish(config, means, age, len(sub), window)
        window += 14


def last_by_pollster(
    election: int,
    asof: dt.date,
    config: list[str],
    max_age_days: int = 60,
    registry: Registry | None = None,
    polls: pl.DataFrame | None = None,
) -> PointForecast:
    reg = registry or default_registry()
    vote = election_date(reg, election)
    df = load_polls(election, asof) if polls is None else polls.filter(pl.col("fieldwork_end") <= asof)
    lo = asof - dt.timedelta(days=max_age_days)
    last: dict[str, dict] = {}
    for p in _poll_rows(df):
        if p["fieldwork_end"] <= lo:
            continue
        cur = last.get(p["pollster"])
        if cur is None or (p["fieldwork_end"], p["poll_id"]) > (cur["fieldwork_end"], cur["poll_id"]):
            last[p["pollster"]] = p
    sub = list(last.values())
    means, age = _average(sub, config, vote)
    if not all(c in means for c in config):
        # candidats absents des derniers sondages : complétés par la moyenne de la fenêtre
        fallback = poll_mean(election, asof, config, days=max_age_days, registry=reg, polls=df)
        for c, s in zip(config, fallback.shares):
            means.setdefault(c, float(s))
    return _finish(config, means, age, len(sub), max_age_days)
