"""Items de calibration de η tirés des scrutins de liste (européennes, législatives).

docs/04-modele.md, section 2 : « avec cinq présidentielles, on complète la calibration avec les
législatives et européennes ». Étude : `back/studies/calibration_extension.py`,
runs/2026-10-20-calibration-extension ; backtest : runs/<date>-backtest-lists.

Un scrutin de liste entre dans `ErrorModel.fit` par `extra_items` (ou `with_lists=True`) : ses
items s'ajoutent à l'apprentissage, sous la clé du scrutin (`Item.election`, par exemple
"euro2019"), et ne sont jamais laissés de côté par le leave-one-election-out des
présidentielles. Pondération par `Item.weight` : erreur finale w, mouvements w / nombre
d'horizons.

- Erreur finale (`kind="final"`) : log(résultat) − log(moyenne des sondages des 14 derniers
  jours), lue dans `poll_errors_lists.parquet` (méthode `mean_14d`, niveau candidat), unités
  dont la part finale dans les sondages est d'au moins `MIN_SHARE` points (les petites listes,
  sondées à 0 ou « <0,5 », donneraient des log-rapports dominés par l'arrondi), parts
  renormalisées sur ces unités.
- Mouvements (`kind="move"`, hors défaut) : log(moyenne finale) − log(moyenne des 30 jours avant
  l'horizon), h = écart des âges moyens, horizons de `error.MOVE_HORIZONS`, recalculés depuis
  les sondages (`polls_lists`, `poll_results_lists`). Le backtest les écarte : la dynamique d'une
  campagne de liste diffère de celle d'une présidentielle (couverture à 90 % de 92 à 81 % à
  180 jours avec eux).

Défaut retenu (`WEIGHT`, `KINDS`) : erreurs finales seules, pondérées 0,5.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from functools import lru_cache

import numpy as np
import polars as pl

from vvs import paths
from vvs.model import error as er

log = logging.getLogger(__name__)

LIST_KEYS = ("euro2014", "euro2019", "euro2024", "leg2022", "leg2024")
MIN_SHARE = 1.0  # points
WEIGHT = 0.5
KINDS = ("final",)


def election_year(election: int | str) -> int:
    """Année d'une élection : l'entier lui-même, ou les quatre chiffres d'une clé de scrutin
    de liste ("leg2024" -> 2024)."""
    if isinstance(election, (int, np.integer)):
        return int(election)
    m = re.search(r"(\d{4})$", str(election))
    if m is None:
        raise ValueError(f"année introuvable dans {election!r}")
    return int(m.group(1))


def _chosen_polls(key: str, reg, polls: pl.DataFrame, poll_results: pl.DataFrame,
                  official: dict[str, float]) -> list[dict]:
    """Sondages du scrutin avec l'hypothèse retenue (règles de derived.poll_errors), privée
    des unités incomplètes (`lists.unit_poll_results`)."""
    from vvs.curate import derived, lists

    vote = lists.vote_date(key)
    p = polls.filter((pl.col("election") == key) & (pl.col("fieldwork_end") < vote))
    r = lists.unit_poll_results(polls, poll_results, reg, key).filter(~pl.col("suspect"))
    hyps: dict[str, dict[str, dict[str, float]]] = {}
    incomplete: set[tuple[str, str, str]] = set()
    for pid, hid, cid, share, ok in r.select(
        "poll_id", "hypothesis_id", "candidate_id", "share", "complete"
    ).iter_rows():
        hyps.setdefault(pid, {}).setdefault(hid, {})[cid] = share
        if not ok:
            incomplete.add((pid, hid, cid))
    out = []
    for m in p.iter_rows(named=True):
        if m["poll_id"] not in hyps:
            continue
        pick = derived._choose(hyps[m["poll_id"]], official, 1)
        if pick is None:
            continue
        hid = pick[0]
        shares = {u: s for u, s in hyps[m["poll_id"]][hid].items()
                  if (m["poll_id"], hid, u) not in incomplete}
        out.append({**m, "shares": shares})
    return out


def _window_mean(group: list[dict], vote: dt.date) -> tuple[dict[str, float], float]:
    from vvs.curate import derived

    sums: dict[str, list[float]] = {}
    ages = []
    for p in derived._thin_rolling(group):
        for u, s in p["shares"].items():
            sums.setdefault(u, []).append(s)
        mid = p["fieldwork_start"] + (p["fieldwork_end"] - p["fieldwork_start"]) / 2
        ages.append((vote - mid).days)
    return {u: float(np.mean(v)) for u, v in sums.items()}, float(np.mean(ages))


def _move_items(key: str, units: list[str], blocs: list[str], weight: float, reg, polls,
                poll_results, res: pl.DataFrame) -> list[er.Item]:
    from vvs.curate import derived, lists

    vote = lists.vote_date(key)
    official = dict(res.filter(pl.col("election") == key).select("unit", "share").iter_rows())
    chosen = _chosen_polls(key, reg, polls, poll_results, official)
    final = [p for p in chosen
             if p["fieldwork_end"] >= vote - dt.timedelta(days=derived.RECENT_DAYS)]
    fin, fin_age = _window_mean(final, vote)
    moves = []
    for h in er.MOVE_HORIZONS:
        hi = vote - dt.timedelta(days=h)
        lo = hi - dt.timedelta(days=er.MOVE_WINDOW_DAYS)
        window = [q for q in chosen if lo < q["fieldwork_end"] <= hi]
        if not window:
            continue
        mv, age = _window_mean(window, vote)
        keep = [i for i, u in enumerate(units) if mv.get(u, 0.0) > 0]
        if len(keep) < 3:
            continue
        a = np.array([mv[units[i]] for i in keep])
        f = np.array([fin[units[i]] for i in keep])
        a, f = a / a.sum(), f / f.sum()
        moves.append(er.Item(key, "move", max(age - fin_age, 1.0), [units[i] for i in keep],
                             [blocs[i] for i in keep], a, np.log(f) - np.log(a),
                             final_age=fin_age))
    for it in moves:
        it.weight = weight / len(moves)
    return moves


def list_items(
    weight: float = WEIGHT,
    min_share: float = MIN_SHARE,
    kinds: tuple[str, ...] = KINDS,
    keys: tuple[str, ...] = LIST_KEYS,
) -> list[er.Item]:
    """Items des scrutins de liste au format de `ErrorModel.fit(extra_items=…)`.

    `kinds` : "final" (erreur finale, poids `weight`) et/ou "move" (mouvements, poids
    `weight` / nombre d'horizons)."""
    from vvs.curate import lists

    bad = set(kinds) - {"final", "move"}
    if bad:
        raise ValueError(f"kinds inconnus : {sorted(bad)}")
    reg = lists.load_lists()
    errors = pl.read_parquet(paths.CURATED / "poll_errors_lists.parquet").filter(
        (pl.col("level") == "candidate") & (pl.col("method") == "mean_14d")
    )
    if "move" in kinds:
        polls = lists.polls_with_rolling(pl.read_parquet(paths.CURATED / "polls_lists.parquet"))
        poll_results = pl.read_parquet(paths.CURATED / "poll_results_lists.parquet")
        res = pl.read_parquet(paths.CURATED / "results_lists_by_bloc.parquet").filter(
            (pl.col("level") == "list") & (pl.col("round") == 1)
        )
    items: list[er.Item] = []
    for key in keys:
        sub = errors.filter(pl.col("election") == key)
        if sub.is_empty():
            raise ValueError(f"poll_errors_lists : scrutin {key!r} absent")
        sub = sub.filter(pl.col("poll_share") >= min_share)
        units = sub["unit"].to_list()
        blocs = [reg.bloc_of(key, u) for u in units]
        p = sub["poll_share"].to_numpy()
        r = sub["result_share"].to_numpy()
        p, r = p / p.sum(), r / r.sum()
        if "final" in kinds:
            items.append(er.Item(key, "final", 0.0, units, blocs, p, np.log(r) - np.log(p),
                                 weight=weight, final_age=er.FINAL_WINDOW_DAYS / 2))
        if "move" in kinds:
            moves = _move_items(key, units, blocs, weight, reg, polls, poll_results, res)
            items += moves
            log.info("%s : %d unités, %d mouvements (h = %s)", key, len(units), len(moves),
                     [round(it.h) for it in moves])
    return items


@lru_cache(maxsize=8)
def _cached(weight: float, min_share: float, kinds: tuple[str, ...],
            keys: tuple[str, ...]) -> tuple[er.Item, ...]:
    return tuple(list_items(weight, min_share, kinds, keys))


def default_list_items() -> list[er.Item]:
    """Items du défaut retenu (`WEIGHT`, `MIN_SHARE`, `KINDS`), mis en cache. Les items ne
    doivent pas être modifiés en place (`ErrorModel.fit` les copie s'il les décale)."""
    return list(_cached(WEIGHT, MIN_SHARE, KINDS, LIST_KEYS))
