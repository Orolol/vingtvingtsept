"""Schémas des tables curated (voir docs/03-donnees.md).

Les ingestions produisent des tables conformes à ces schémas. Les enums sont des chaînes
pour rester lisibles en parquet ; les valeurs admises sont listées ici et vérifiées par
`validate_*`.
"""

from __future__ import annotations

import polars as pl

POLL_MODES = ("online", "phone", "mixed", "unknown")
POLL_SOURCES = ("wikipedia", "commission", "nsppolls")
POPULATIONS = ("registered", "certain_voters", "unknown")
SHARE_BASES = ("expressed", "all_respondents", "unknown")
GEO_LEVELS = ("national", "department", "commune", "polling_station")

# Identifiants réservés dans poll_results.candidate_id
RESERVED_CANDIDATE_IDS = ("abstention", "blank", "undecided", "other")

POLLS_SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "election": pl.Int32,
        "pollster": pl.Utf8,
        "sponsor": pl.Utf8,
        "fieldwork_start": pl.Date,
        "fieldwork_end": pl.Date,
        "publish_date": pl.Date,
        "sample_size": pl.Int32,
        "mode": pl.Utf8,
        "source": pl.Utf8,
        "source_url": pl.Utf8,
        "ingested_at": pl.Datetime("us"),
    }
)

POLL_RESULTS_SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "round": pl.Int8,
        "hypothesis_id": pl.Utf8,
        "population": pl.Utf8,
        "candidate_id": pl.Utf8,
        "share": pl.Float64,
        "share_base": pl.Utf8,
        "raw_share": pl.Float64,
    }
)

RESULTS_SCHEMA = pl.Schema(
    {
        "election": pl.Int32,
        "round": pl.Int8,
        "geo_level": pl.Utf8,
        "geo_code": pl.Utf8,
        "candidate_id": pl.Utf8,
        "votes": pl.Int64,
        "registered": pl.Int64,
        "voters": pl.Int64,
        "expressed": pl.Int64,
    }
)


def hypothesis_id(candidate_ids: list[str]) -> str:
    """Identifiant stable d'une configuration de candidats : ids triés, joints par '+'.

    Les ids réservés (abstention, blank, undecided, other) n'en font pas partie.
    """
    ids = sorted({c for c in candidate_ids if c not in RESERVED_CANDIDATE_IDS})
    return "+".join(ids)


def _check_enum(df: pl.DataFrame, col: str, allowed: tuple[str, ...]) -> None:
    bad = df.filter(~pl.col(col).is_in(allowed) & pl.col(col).is_not_null())
    if bad.height:
        raise ValueError(f"{col}: valeurs hors enum {allowed}: {bad[col].unique().to_list()}")


def validate_polls(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(list(POLLS_SCHEMA.keys())).cast(dict(POLLS_SCHEMA))
    if df["poll_id"].n_unique() != df.height:
        raise ValueError("polls: poll_id non unique")
    _check_enum(df, "mode", POLL_MODES)
    _check_enum(df, "source", POLL_SOURCES)
    bad_dates = df.filter(pl.col("fieldwork_end") < pl.col("fieldwork_start"))
    if bad_dates.height:
        raise ValueError(f"polls: fieldwork_end < fieldwork_start pour {bad_dates['poll_id'].to_list()}")
    return df


def validate_poll_results(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(list(POLL_RESULTS_SCHEMA.keys())).cast(dict(POLL_RESULTS_SCHEMA))
    _check_enum(df, "population", POPULATIONS)
    _check_enum(df, "share_base", SHARE_BASES)
    if df.filter(~pl.col("round").is_in([1, 2])).height:
        raise ValueError("poll_results: round doit valoir 1 ou 2")
    return df


def validate_results(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(list(RESULTS_SCHEMA.keys())).cast(dict(RESULTS_SCHEMA))
    _check_enum(df, "geo_level", GEO_LEVELS)
    return df
