"""Échantillon à classer et variables hebdomadaires de presse (`press_features.parquet`).

Univers classé : les titres qui citent le candidat dans le titre lui-même (`in_title`), au
plus `CAP` par candidat et par semaine (lundi à dimanche). Au-delà, l'échantillon est réparti
sur les jours de la semaine : dans chaque jour, les titres sont rangés par une empreinte de
story_id, puis on prend le premier de chaque jour, le deuxième de chaque jour, etc., jusqu'au
plafond ; un jour creux laisse sa place aux autres. Déterministe.

Variables (une ligne par cycle, candidat et semaine) :

- `articles` : articles collectés qui citent le candidat (titre ou corps, requête Media Cloud) ;
- `mentions` : titres qui citent le candidat dans le titre (avant plafond) ;
- `share_of_voice` : 100 × mentions / somme des mentions des candidats suivis du cycle ;
- `n_labeled` : titres classés de la semaine (après plafond) ;
- `tone` : ton moyen (−1 à 1) ; `neg_share`, `pos_share` : parts (points) de titres à −1, +1 ;
- `attack_share`, `scandal_share`, `support_share`… : parts (points) de chaque posture
  (`stance_<posture>` pour toutes) ;
- `topic_<thème>` : part (points) des titres classés portant le thème (multi-label : la somme
  peut dépasser 100) ;
- `days` : jours de la semaine dans la fenêtre de collecte (7 sauf aux bords).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import polars as pl

from vvs import paths
from vvs.studies.panel import week_start

CAP = 300
OUTPUT_NAME = "press_features.parquet"


def _h(s: str) -> str:
    return hashlib.sha1(f"cap:{s}".encode()).hexdigest()


def cap_sample(titles: pl.DataFrame, cap: int = CAP) -> pl.DataFrame:
    """Titres `in_title`, au plus `cap` par (cycle, candidat, semaine), répartis sur les jours."""
    t = (titles.filter(pl.col("in_title"))
         .with_columns(week_start("publish_date").alias("week"),
                       pl.col("story_id").map_elements(_h, return_dtype=pl.Utf8).alias("_h"))
         .sort("cycle", "candidate_id", "publish_date", "_h"))
    t = t.with_columns(pl.int_range(pl.len()).over("cycle", "candidate_id", "publish_date")
                       .alias("_r"))
    t = t.sort("cycle", "candidate_id", "week", "_r", "publish_date")
    t = t.with_columns(pl.int_range(pl.len()).over("cycle", "candidate_id", "week").alias("_k"))
    return t.filter(pl.col("_k") < cap).drop("_h", "_r", "_k").sort(
        "cycle", "candidate_id", "publish_date", "story_id")


def weekly_features(titles: pl.DataFrame, labels: pl.DataFrame, stances: Iterable[str],
                    themes: Iterable[str], windows: dict[int, tuple] | None = None
                    ) -> pl.DataFrame:
    """titres (press_titles) + étiquettes d'une version et d'un modèle -> variables."""
    stances, themes = list(stances), list(themes)
    t = titles.with_columns(week_start("publish_date").alias("week"))
    keys = ["cycle", "candidate_id", "week"]
    base = t.group_by(keys).agg(pl.len().alias("articles"),
                                pl.col("in_title").sum().cast(pl.Int64).alias("mentions"))
    tot = pl.col("mentions").sum().over("cycle", "week")
    base = base.with_columns(pl.when(tot > 0).then(100 * pl.col("mentions") / tot)
                             .otherwise(None).alias("share_of_voice"))
    lab = (labels.select("story_id", "candidate_id", "tone", "stance", "themes")
           .join(t.select("story_id", "candidate_id", "cycle", "week"),
                 on=["story_id", "candidate_id"]))
    aggs = [pl.len().alias("n_labeled"), pl.col("tone").cast(pl.Float64).mean().alias("tone"),
            (100 * (pl.col("tone") == -1).mean()).alias("neg_share"),
            (100 * (pl.col("tone") == 1).mean()).alias("pos_share")]
    aggs += [(100 * (pl.col("stance") == s).mean()).alias(f"stance_{s}") for s in stances]
    aggs += [(100 * pl.col("themes").list.contains(th).mean()).alias(f"topic_{th}")
             for th in themes]
    w = lab.group_by(keys).agg(aggs)
    out = base.join(w, on=keys, how="left").with_columns(
        pl.col("n_labeled").fill_null(0),
        pl.col("stance_attack").alias("attack_share"),
        pl.col("stance_scandal").alias("scandal_share"),
        pl.col("stance_support").alias("support_share"))
    if windows:
        days = []
        for e, (a, b) in windows.items():
            d = pl.date_range(a, b, "1d", eager=True)
            days.append(pl.DataFrame({"cycle": e, "day": d}).with_columns(
                week_start("day").alias("week")).group_by("cycle", "week")
                .agg(pl.len().alias("days")))
        out = out.join(pl.concat(days).cast({"cycle": out.schema["cycle"]}),
                       on=["cycle", "week"], how="left")
    return out.sort(keys)


def write_features(df: pl.DataFrame) -> None:
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(paths.CURATED / OUTPUT_NAME)
