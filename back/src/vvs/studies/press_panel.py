"""Panel hebdomadaire presse pour l'étude de la phase 7 (back/studies/press.py).

Semaine = lundi à dimanche (`panel.week_start`), comme le panel des sondages. Une semaine
n'est retenue que si ses 7 jours sont présents dans la série (début et fin de fenêtre
Media Cloud tronqués).

- Media Cloud (`press_mediacloud.parquet`) : `count` et `total` sommés sur la semaine ;
  `volume_share` = 100 × count / total (points de la collection) ; pour les candidats,
  `sov` (part de voix) = 100 × count / somme des `count` des membres du cycle cette
  semaine-là. La part de voix ne dépend pas de la taille de la collection, qui change
  fortement d'une année à l'autre ; `volume_share` en dépend moins qu'un comptage brut mais
  reste sensible à la composition par média.
- GDELT (`press_timelines.parquet`) : volume et total surveillé sommés, ton moyen pondéré
  par le volume du jour (un jour sans article n'a pas de ton).
- INA (`press_ina.parquet`) : part de la durée des JT par rubrique, toutes chaînes.
- Sondages par bloc : moyenne des instituts de la semaine, chacun moyenné sur ses sondages
  (mêmes règles que `panel.monthly_bloc_shares`, à la semaine).
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from vvs.studies.panel import week_start


def _complete_weeks(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return df.filter(pl.col("_days") == 7).drop("_days").sort(*keys, "week")


def weekly_counts(mc: pl.DataFrame) -> pl.DataFrame:
    """(scope, id, election, week) : count, total, volume_share (points)."""
    keys = ["scope", "id", "election"]
    w = (mc.with_columns(week_start("date").alias("week"))
         .group_by(*keys, "week")
         .agg(pl.col("count").sum(), pl.col("total").sum(), pl.len().alias("_days")))
    w = _complete_weeks(w, keys)
    return w.with_columns(
        pl.when(pl.col("total") > 0).then(100 * pl.col("count") / pl.col("total"))
        .otherwise(None).alias("volume_share"))


def share_of_voice(weekly: pl.DataFrame, members: dict[int, Iterable[str]]) -> pl.DataFrame:
    """Candidats : (election, candidate_id, week, count, total, volume_share, sov).

    `sov` = 100 × count / somme des membres du cycle présents cette semaine ; vide si aucun
    membre n'a d'article (jamais en pratique)."""
    rows = []
    for e, cids in members.items():
        c = weekly.filter((pl.col("scope") == "candidate") & (pl.col("election") == e)
                          & pl.col("id").is_in(list(cids)))
        rows.append(c)
    c = pl.concat(rows) if rows else weekly.clear()
    tot = pl.col("count").sum().over("election", "week")
    return (c.rename({"id": "candidate_id"})
            .with_columns(pl.when(tot > 0).then(100 * pl.col("count") / tot)
                          .otherwise(None).alias("sov"))
            .drop("scope").sort("election", "candidate_id", "week"))


def theme_salience(weekly: pl.DataFrame) -> pl.DataFrame:
    """Thèmes : (election, theme, week, count, total, salience = volume_share)."""
    return (weekly.filter(pl.col("scope") == "theme").rename({"id": "theme"})
            .with_columns(pl.col("volume_share").alias("salience"))
            .drop("scope").sort("election", "theme", "week"))


def gdelt_weekly(tl: pl.DataFrame) -> pl.DataFrame:
    """(candidate_id, week) : volume, total_monitored, volume_share (points), tone."""
    w = (tl.filter(pl.col("candidate_id").is_not_null())
         .with_columns(week_start("date").alias("week"))
         .group_by("candidate_id", "week")
         .agg(pl.col("volume").sum(), pl.col("total_monitored").sum(),
              ((pl.col("tone") * pl.col("volume")).sum()
               / pl.col("volume").filter(pl.col("tone").is_not_null()).sum()).alias("tone"),
              pl.col("tone").is_not_null().sum().alias("_tone_days"),
              pl.len().alias("_days")))
    w = _complete_weeks(w, ["candidate_id"])
    return w.with_columns(
        (100 * pl.col("volume") / pl.col("total_monitored")).alias("volume_share"),
        pl.when(pl.col("_tone_days") > 0).then(pl.col("tone")).otherwise(None).alias("tone"),
    ).drop("_tone_days")


def ina_weekly(ina: pl.DataFrame, rubrics: dict[str, Iterable[str]]) -> pl.DataFrame:
    """(week, key, share) : part (points) de la durée des JT dans les rubriques de chaque clé.

    Semaine complète seulement (7 jours avec au moins un sujet)."""
    d = ina.with_columns(week_start("date").alias("week"))
    days = d.group_by("week").agg(pl.col("date").n_unique().alias("_days"))
    tot = d.group_by("week").agg(pl.col("duration_s").sum().alias("_tot"))
    parts = []
    for key, rub in rubrics.items():
        s = (d.filter(pl.col("theme").is_in(list(rub))).group_by("week")
             .agg(pl.col("duration_s").sum().alias("_dur")))
        parts.append(tot.join(s, on="week", how="left").with_columns(
            pl.lit(key).alias("key"),
            (100 * pl.col("_dur").fill_null(0) / pl.col("_tot")).alias("share")))
    out = pl.concat(parts).join(days, on="week")
    return (out.filter(pl.col("_days") == 7).select("week", "key", "share")
            .sort("key", "week"))


def weekly_bloc_shares(poll_bloc_shares: pl.DataFrame) -> pl.DataFrame:
    """(election, week, bloc) : moyenne des instituts, chacun moyenné sur ses sondages."""
    p = poll_bloc_shares.with_columns(week_start("fieldwork_end").alias("week"))
    by_p = p.group_by("election", "week", "bloc", "pollster").agg(pl.col("share").mean())
    return (by_p.group_by("election", "week", "bloc")
            .agg(pl.col("share").mean(), pl.len().cast(pl.Int32).alias("n_pollsters"))
            .sort("election", "bloc", "week"))
