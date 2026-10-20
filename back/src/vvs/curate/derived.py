"""Tables dérivées de la curation : parts par bloc et erreurs historiques des sondages.

`results_by_bloc` : pour chaque hypothèse, somme des parts (base exprimés) par bloc
(`bloc_of(candidate_id, election)`), l'id réservé `other` allant au bloc `other`.
Abstention, indécis et blancs (base `all_respondents`) n'y entrent pas.

`poll_errors` : pour chaque élection passée et chaque tour, écart entre sondages et résultat
officiel national (part des exprimés), par candidat et par bloc :

- `last_by_pollster` : dernier sondage de chaque institut terminé avant le jour du vote ;
- `mean_14d` : moyenne simple des sondages terminés dans les 14 jours précédant le vote ;
- `horizon` : moyenne simple des sondages terminés dans la fenêtre de 30 jours qui précède
  l'horizon (`horizon_days` jours avant le vote).

Règles communes :

- hypothèses suspectes exclues ; un sondage contribue une seule hypothèse : celle qui
  contient exactement les candidats officiels du tour, sinon (premier tour seulement) la plus
  proche (moins de candidats en trop ou manquants, puis moins de voix officielles manquantes),
  signalée par `exact_config` faux et `config_distance` ; au second tour, seul le duel
  réellement en lice compte ;
- un candidat absent de l'hypothèse est absent, pas à zéro : la moyenne d'un candidat porte
  sur les sondages qui le testent (`n_polls`) ; de même pour un bloc ;
- moyennes : les vagues d'un rolling (`is_rolling`) d'un même institut sont éclaircies pour
  ne garder que des terrains disjoints (la plus récente, puis la précédente qui finit avant
  son début, etc.), pour ne pas compter plusieurs fois les mêmes répondants.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import polars as pl

from vvs import schemas
from vvs.curate import rules

NON_EXPRESSED = ("abstention", "blank", "undecided")
HORIZONS = (180, 90, 30, 7)
HORIZON_WINDOW_DAYS = 30
RECENT_DAYS = 14

BlocFn = Callable[[str, int], str]


def _bloc(bloc_of: BlocFn, cid: str, election: int) -> str:
    return "other" if cid == "other" else bloc_of(cid, election)


def results_by_bloc(polls: pl.DataFrame, results: pl.DataFrame, bloc_of: BlocFn) -> pl.DataFrame:
    expressed = results.filter(~pl.col("candidate_id").is_in(NON_EXPRESSED)).join(
        polls.select("poll_id", "election"), on="poll_id"
    )
    pairs = expressed.select("candidate_id", "election").unique()
    mapping = pairs.with_columns(
        pl.Series("bloc", [_bloc(bloc_of, c, e) for c, e in pairs.iter_rows()], dtype=pl.Utf8)
    )
    out = (
        expressed.join(mapping, on=["candidate_id", "election"])
        .group_by("poll_id", "election", "round", "hypothesis_id", "population", "bloc")
        .agg(
            pl.col("share").sum(),
            (pl.col("candidate_id") != "other").sum().cast(pl.Int32).alias("n_candidates"),
            pl.col("suspect").first(),
        )
        .sort("poll_id", "round", "hypothesis_id", "bloc")
    )
    return schemas.validate_poll_results_by_bloc(out)


def official_shares(results: pl.DataFrame) -> pl.DataFrame:
    """(election, round, candidate_id, share) au niveau national, en % des exprimés."""
    nat = results.filter(pl.col("geo_level") == "national")
    if nat.is_empty():
        raise ValueError("résultats sans niveau national")
    return nat.select(
        "election", "round", "candidate_id",
        (100 * pl.col("votes") / pl.col("expressed")).alias("share"),
    )


def _choose(hyps: dict[str, dict[str, float]], official: dict[str, float], round_: int):
    """(hypothesis_id, distance) de l'hypothèse retenue, ou None."""
    off = set(official)
    best = None
    for hid, shares in hyps.items():
        cands = set(rules.named(shares))
        missing, extra = off - cands, cands - off
        dist = len(missing) + len(extra)
        if round_ == 2 and dist:
            continue
        key = (dist, sum(official[c] for c in missing), hid)
        if best is None or key < best[0]:
            best = (key, hid, dist)
    return None if best is None else (best[1], best[2])


def _thin_rolling(polls: list[dict]) -> list[dict]:
    """Garde les sondages ordinaires et, par institut, des vagues de rolling disjointes."""
    kept = [p for p in polls if not p["is_rolling"]]
    by_pollster: dict[str, list[dict]] = {}
    for p in polls:
        if p["is_rolling"]:
            by_pollster.setdefault(p["pollster"], []).append(p)
    for waves in by_pollster.values():
        waves.sort(key=lambda p: (p["fieldwork_end"], p["fieldwork_start"]), reverse=True)
        last_start = None
        for p in waves:
            if last_start is None or p["fieldwork_end"] < last_start:
                kept.append(p)
                last_start = p["fieldwork_start"]
    return kept


def _units(p: dict, election: int, bloc_of: BlocFn) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for cid, share in p["shares"].items():
        out[("candidate", cid)] = share
        b = ("bloc", _bloc(bloc_of, cid, election))
        out[b] = out.get(b, 0.0) + share
    return out


def poll_errors(
    polls: pl.DataFrame,
    results: pl.DataFrame,
    official: pl.DataFrame,
    round_dates: dict[tuple[int, int], date],
    bloc_of: BlocFn,
    horizons: tuple[int, ...] = HORIZONS,
) -> pl.DataFrame:
    """Voir le docstring du module. `official` : (election, round, candidate_id, share)."""
    rows: list[dict] = []
    for (election, round_), vote_day in sorted(round_dates.items()):
        off = {
            c: s for c, s in official.filter(
                (pl.col("election") == election) & (pl.col("round") == round_)
            ).select("candidate_id", "share").iter_rows()
        }
        if not off:
            continue
        result_units: dict[tuple[str, str], float] = {}
        for cid, s in off.items():
            result_units[("candidate", cid)] = s
            b = ("bloc", _bloc(bloc_of, cid, election))
            result_units[b] = result_units.get(b, 0.0) + s
        meta = polls.filter(
            (pl.col("election") == election) & (pl.col("fieldwork_end") < vote_day)
        ).select("poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size", "is_rolling")
        res = results.filter(
            (pl.col("round") == round_) & ~pl.col("suspect")
            & ~pl.col("candidate_id").is_in(NON_EXPRESSED)
        ).join(meta.select("poll_id"), on="poll_id")
        hyps: dict[str, dict[str, dict[str, float]]] = {}
        for pid, hid, cid, share in res.select("poll_id", "hypothesis_id", "candidate_id", "share").iter_rows():
            hyps.setdefault(pid, {}).setdefault(hid, {})[cid] = share
        chosen: list[dict] = []
        for m in meta.iter_rows(named=True):
            if m["poll_id"] not in hyps:
                continue
            pick = _choose(hyps[m["poll_id"]], off, round_)
            if pick is None:
                continue
            hid, dist = pick
            chosen.append({**m, "shares": hyps[m["poll_id"]][hid], "distance": dist})

        def emit(method, horizon, group, pollster=None, poll_id=None, fieldwork_end=None,
                 *, election=election, round_=round_, off=off, result_units=result_units):
            sums: dict[tuple[str, str], list[float]] = {}
            for p in group:
                for unit, share in _units(p, election, bloc_of).items():
                    sums.setdefault(unit, []).append(share)
            exact = all(p["distance"] == 0 for p in group)
            dist = max(p["distance"] for p in group)
            for (level, unit), shares in sorted(sums.items()):
                if level == "candidate" and unit not in off:
                    continue
                result = result_units.get((level, unit), 0.0)
                mean = sum(shares) / len(shares)
                rows.append({
                    "election": election, "round": round_, "method": method,
                    "horizon_days": horizon, "pollster": pollster, "poll_id": poll_id,
                    "fieldwork_end": fieldwork_end, "level": level, "unit": unit,
                    "poll_share": mean, "result_share": result, "error": mean - result,
                    "n_polls": len(shares), "exact_config": exact, "config_distance": dist,
                })

        by_pollster: dict[str, dict] = {}
        for p in chosen:
            cur = by_pollster.get(p["pollster"])
            if cur is None or (p["fieldwork_end"], p["sample_size"] or 0) > (cur["fieldwork_end"], cur["sample_size"] or 0):
                by_pollster[p["pollster"]] = p
        for pollster, p in sorted(by_pollster.items()):
            emit("last_by_pollster", None, [p], pollster, p["poll_id"], p["fieldwork_end"])
        recent = [p for p in chosen if p["fieldwork_end"] >= vote_day - timedelta(days=RECENT_DAYS)]
        if recent:
            emit("mean_14d", RECENT_DAYS, _thin_rolling(recent))
        for h in horizons:
            hi = vote_day - timedelta(days=h)
            lo = hi - timedelta(days=HORIZON_WINDOW_DAYS)
            window = [p for p in chosen if lo < p["fieldwork_end"] <= hi]
            if window:
                emit("horizon", h, _thin_rolling(window))
    df = pl.DataFrame(rows, schema=schemas.POLL_ERRORS_SCHEMA)
    return schemas.validate_poll_errors(df)
