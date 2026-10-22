"""Variantes expérimentales des scénarios ; protocole docs/11, aucune activation live."""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

import polars as pl


def event_exclusions(candidates, election, asof, registry, events):
    """Événements connus la veille au plus tard ; aucune lecture des résultats du scrutin."""
    excluded, applied = set(), []
    for event in sorted(events, key=lambda e: (str(e["date"]), e["id"])):
        date = dt.date.fromisoformat(str(event["date"]))
        if event["election"] != election or date >= asof:
            continue
        if event["type"] == "withdrawal":
            removed = set(candidates) & {event["candidate"]}
        elif event["type"] == "primary":
            winner = event["candidate"]
            party = registry.candidates[winner].party_at(date.year)
            if party is None:
                raise ValueError(f"parti de désignation inconnu : {event['id']}")
            removed = {
                c
                for c in candidates
                if c != winner and registry.candidates[c].party_at(date.year) == party
            }
        else:
            continue
        excluded.update(removed)
        applied.append(
            {
                "id": event["id"],
                "date": str(date),
                "type": event["type"],
                "candidate": event["candidate"],
                "excluded": sorted(removed),
                "sources": event.get("sources", []),
            }
        )
    return excluded, applied


def reconstruct(df, election, asof, registry, settings, events, variant):
    from vvs.eval.configurations import list_publication

    if variant not in settings["variants"] or variant not in {"memory", "memory_events"}:
        raise ValueError(f"variante inconnue : {variant}")
    window, half_life = settings["window_days"], settings["half_life_days"]
    if window <= 0 or half_life <= 0:
        raise ValueError("fenêtre et demi-vie strictement positives requises")
    if df.filter(pl.col("fieldwork_end") > asof).height:
        raise ValueError("sondage futur fourni à la reconstruction")
    if asof >= list_publication(election, registry):
        candidates = sorted(c.candidate_id for c in registry.elections[election].candidates)
        return [{"candidates": candidates, "probability": 1.0}], {"source": "official_list"}
    recent = df.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=window))
    if recent.is_empty():
        raise ValueError("aucune hypothèse dans la fenêtre de mémoire")
    hyps, dates = defaultdict(set), {}
    for (pid, _), group in recent.group_by("poll_id", "hypothesis_id"):
        if group["fieldwork_end"].n_unique() != 1:
            raise ValueError(f"dates incohérentes dans {pid}")
        date = group["fieldwork_end"][0]
        if pid in dates and dates[pid] != date:
            raise ValueError(f"dates incohérentes dans {pid}")
        dates[pid] = date
        candidates = tuple(sorted(set(group["candidate_id"])))
        if len(candidates) < 2:
            raise ValueError(f"hypothèse à moins de deux candidats : {pid}")
        for candidate in candidates:
            registry.bloc_of(candidate, election)
        hyps[pid].add(candidates)
    mass = defaultdict(float)
    # Ordre stable indépendamment des groupes et du parallélisme Polars.
    for pid in sorted(hyps):
        weight = 2 ** (-(asof - dates[pid]).days / half_life)
        for candidates in sorted(hyps[pid]):
            mass[candidates] += weight / len(hyps[pid])
    total = math.fsum(mass.values())
    candidates = sorted({c for cfg in mass for c in cfg})
    excluded, applied = set(), []
    if variant == "memory_events":
        excluded, applied = event_exclusions(candidates, election, asof, registry, events)
    kept = {cfg: p for cfg, p in mass.items() if not excluded.intersection(cfg)}
    retained = math.fsum(kept.values())
    if retained <= 0:
        raise ValueError("aucune configuration admissible après les événements")
    configs = [
        {"candidates": list(c), "probability": p / retained} for c, p in sorted(kept.items())
    ]
    return configs, {
        "source": variant,
        "window_days": window,
        "half_life_days": half_life,
        "n_polls": len(hyps),
        "n_configurations_before": len(mass),
        "n_configurations_after": len(kept),
        "excluded_candidates": sorted(excluded),
        "discarded_probability": 1 - retained / total,
        "applied_events": applied,
    }


def presence(configurations):
    probabilities = defaultdict(float)
    for cfg in configurations:
        for candidate in cfg["candidates"]:
            probabilities[candidate] += cfg["probability"]
    return dict(sorted(probabilities.items()))
