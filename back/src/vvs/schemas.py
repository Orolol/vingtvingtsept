"""Schémas des tables curated (voir docs/03-donnees.md).

Les ingestions produisent des tables conformes à ces schémas. Les enums sont des chaînes
pour rester lisibles en parquet ; les valeurs admises sont listées ici et vérifiées par
`validate_*`.
"""

from __future__ import annotations

import polars as pl

POLL_MODES = ("online", "phone", "mixed", "unknown")
POLL_SOURCES = ("wikipedia", "commission", "nsppolls", "mieuxvoter")
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


# --- tables fusionnées par la curation (vvs.curate) : schémas des sources, plus des colonnes

# Ordre de précédence des sources en cas de conflit de valeur (docs/03-donnees.md).
SOURCE_PRECEDENCE = ("commission", "wikipedia", "mieuxvoter", "nsppolls")
SUSPECT_REASONS = ("sum_out_of_range", "incomplete_hypothesis", "undeclared_candidate")
MATCH_RULES = ("new", "date_match", "same_results", "multi_wave_split")
ERROR_METHODS = ("last_by_pollster", "mean_14d", "horizon")
ERROR_LEVELS = ("candidate", "bloc")

# polls.parquet : une ligne par sondage fusionné. `source` est la source dont les valeurs
# sont retenues (la plus prioritaire), `sources` toutes celles qui contiennent le sondage.
CURATED_POLLS_SCHEMA = pl.Schema(
    {
        **POLLS_SCHEMA,
        "sources": pl.List(pl.Utf8),
        "is_rolling": pl.Boolean,
    }
)

# poll_results.parquet : `source` fournit les valeurs de l'hypothèse ; `suspect` et
# `suspect_reason` (raisons jointes par « ; ») valent pour toute l'hypothèse.
CURATED_POLL_RESULTS_SCHEMA = pl.Schema(
    {
        **POLL_RESULTS_SCHEMA,
        "source": pl.Utf8,
        "suspect": pl.Boolean,
        "suspect_reason": pl.Utf8,
    }
)

# poll_sources.parquet : sondage d'une source -> sondage fusionné (`canonical_poll_id`).
# Un sondage source peut alimenter deux sondages fusionnés (notice à deux vagues, sondages
# regroupés par une page) ; il a alors deux lignes.
POLL_SOURCES_SCHEMA = pl.Schema(
    {
        "source": pl.Utf8,
        "source_poll_id": pl.Utf8,
        "canonical_poll_id": pl.Utf8,
        "match_rule": pl.Utf8,
    }
)

POLL_RESULTS_BY_BLOC_SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "election": pl.Int32,
        "round": pl.Int8,
        "hypothesis_id": pl.Utf8,
        "population": pl.Utf8,
        "bloc": pl.Utf8,
        "share": pl.Float64,
        "n_candidates": pl.Int32,
        "suspect": pl.Boolean,
    }
)

# poll_errors.parquet : écart sondages - résultat officiel national (points d'exprimés).
# `method` : dernier sondage de chaque institut, moyenne des 14 derniers jours, moyenne à
# horizon fixe (`horizon_days`, fenêtre de 30 jours en amont). `exact_config` : toutes les
# hypothèses retenues contiennent exactement les candidats officiels ; sinon
# `config_distance` (nombre maximal de candidats en trop ou manquants) le signale.
POLL_ERRORS_SCHEMA = pl.Schema(
    {
        "election": pl.Int32,
        "round": pl.Int8,
        "method": pl.Utf8,
        "horizon_days": pl.Int32,
        "pollster": pl.Utf8,
        "poll_id": pl.Utf8,
        "fieldwork_end": pl.Date,
        "level": pl.Utf8,
        "unit": pl.Utf8,
        "poll_share": pl.Float64,
        "result_share": pl.Float64,
        "error": pl.Float64,
        "n_polls": pl.Int32,
        "exact_config": pl.Boolean,
        "config_distance": pl.Int32,
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


def validate_curated_polls(df: pl.DataFrame) -> pl.DataFrame:
    """Contrôles de `validate_polls`, plus : `sources` non vide et cohérent avec `source`."""
    validate_polls(df)
    df = df.select(list(CURATED_POLLS_SCHEMA.keys())).cast(dict(CURATED_POLLS_SCHEMA))
    bad = df.filter(
        (pl.col("sources").list.len() == 0) | ~pl.col("sources").list.contains(pl.col("source"))
    )
    if bad.height:
        raise ValueError(f"polls: sources incohérentes pour {bad['poll_id'].to_list()[:5]}")
    for s in df["sources"].explode().drop_nulls().unique():
        if s not in POLL_SOURCES:
            raise ValueError(f"polls: source inconnue dans sources : {s}")
    if df["is_rolling"].null_count():
        raise ValueError("polls: is_rolling nul")
    return df


def validate_curated_poll_results(df: pl.DataFrame) -> pl.DataFrame:
    """Contrôles de `validate_poll_results`, plus unicité (sondage, tour, hypothèse,
    candidat), `suspect_reason` renseigné si et seulement si `suspect`."""
    validate_poll_results(df)
    df = df.select(list(CURATED_POLL_RESULTS_SCHEMA.keys())).cast(
        dict(CURATED_POLL_RESULTS_SCHEMA)
    )
    _check_enum(df, "source", POLL_SOURCES)
    key = ["poll_id", "round", "hypothesis_id", "candidate_id"]
    if df.select(key).is_duplicated().any():
        raise ValueError("poll_results: clé (poll_id, round, hypothesis_id, candidate_id) en double")
    if df.filter(pl.col("suspect") != pl.col("suspect_reason").is_not_null()).height:
        raise ValueError("poll_results: suspect et suspect_reason incohérents")
    reasons = df["suspect_reason"].drop_nulls().str.split("; ").explode().unique()
    unknown = [r for r in reasons if r not in SUSPECT_REASONS]
    if unknown:
        raise ValueError(f"poll_results: raisons de suspicion inconnues {unknown}")
    return df


def validate_poll_sources(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(list(POLL_SOURCES_SCHEMA.keys())).cast(dict(POLL_SOURCES_SCHEMA))
    _check_enum(df, "source", POLL_SOURCES)
    _check_enum(df, "match_rule", MATCH_RULES)
    if df.select("source", "source_poll_id", "canonical_poll_id").is_duplicated().any():
        raise ValueError("poll_sources: ligne en double")
    return df


def validate_poll_results_by_bloc(df: pl.DataFrame) -> pl.DataFrame:
    from vvs.registry.models import BLOCS

    df = df.select(list(POLL_RESULTS_BY_BLOC_SCHEMA.keys())).cast(
        dict(POLL_RESULTS_BY_BLOC_SCHEMA)
    )
    _check_enum(df, "bloc", BLOCS)
    if df.select("poll_id", "round", "hypothesis_id", "bloc").is_duplicated().any():
        raise ValueError("poll_results_by_bloc: clé en double")
    return df


def validate_poll_errors(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(list(POLL_ERRORS_SCHEMA.keys())).cast(dict(POLL_ERRORS_SCHEMA))
    _check_enum(df, "method", ERROR_METHODS)
    _check_enum(df, "level", ERROR_LEVELS)
    if df.filter((pl.col("error") - (pl.col("poll_share") - pl.col("result_share"))).abs() > 1e-9).height:
        raise ValueError("poll_errors: error != poll_share - result_share")
    return df
