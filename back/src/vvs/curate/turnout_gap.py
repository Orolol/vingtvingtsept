"""Abstention différentielle : écart par candidat entre certains d'aller voter et inscrits.

`build()` lit les tables de `ingest.commission.archive` (`poll_results_archive` pour 2017 et
2022, `poll_results_commission_dual` pour 2027, `poll_turnout_intent`) et écrit
`data/curated/turnout_gap.parquet`, une ligne par (sondage, tour, hypothèse, candidat) quand
la même hypothèse est publiée sur les deux bases :

- `share_registered`, `share_certain` : parts des exprimés sur l'ensemble des inscrits et sur
  les certains d'aller voter, même redressement (tableaux OpinionWay à deux bases, voir
  `archive.dual_columns`) ; `raw_registered`, `raw_certain` : bruts quand publiés ;
- `gap` = `share_certain` − `share_registered`, en points : positif, l'électorat du candidat
  est plus sûr d'aller voter que la moyenne (il gagne à la sélection des votants) ; négatif,
  il est « fragile » (une part de ses électeurs déclarés risque de s'abstenir) ;
- `certain_share_sample` : part de certains d'aller voter parmi les inscrits interrogés,
  quand la notice la donne (même tour, sinon premier tour), avec `certain_share_kind` et les
  effectifs ; les définitions changent d'un institut et d'une année à l'autre (voir
  `archive.find_turnout_intent`) ;
- `bloc` : bloc du registre ; `matched_poll_id` : sondage de `polls.parquet` décrivant la même
  collecte (2017, 2022), pour la fusion que fera la curation.

Un candidat présent dans une seule base de l'hypothèse (« - » dans l'autre colonne) n'a pas
de ligne. Les deux parts proviennent d'échantillons emboîtés (les certains sont un
sous-ensemble des inscrits) : l'écart est bien moins bruité que la différence de deux
sondages indépendants, mais les arrondis à l'entier des notices (± 0,5 point par part)
dominent pour les petits candidats.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from vvs import paths

OUT_PATH = paths.CURATED / "turnout_gap.parquet"
SOURCES = (
    ("polls_archive.parquet", "poll_results_archive.parquet"),
    ("polls_commission_dual.parquet", "poll_results_commission_dual.parquet"),
)
TURNOUT_INTENT = "poll_turnout_intent.parquet"

SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "election": pl.Int32,
        "pollster": pl.Utf8,
        "fieldwork_start": pl.Date,
        "fieldwork_end": pl.Date,
        "is_rolling": pl.Boolean,
        "round": pl.Int8,
        "hypothesis_id": pl.Utf8,
        "candidate_id": pl.Utf8,
        "bloc": pl.Utf8,
        "share_registered": pl.Float64,
        "share_certain": pl.Float64,
        "gap": pl.Float64,
        "raw_registered": pl.Float64,
        "raw_certain": pl.Float64,
        "certain_share_sample": pl.Float64,
        "certain_share_kind": pl.Utf8,
        "n_registered": pl.Int32,
        "n_certain": pl.Int32,
        "matched_poll_id": pl.Utf8,
    }
)

BlocFn = Callable[[str, int], str]


def pair_populations(polls: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Hypothèses publiées sur les deux bases -> une ligne par candidat présent dans les deux."""
    res = (
        results.filter(~pl.col("suspect").fill_null(False))
        if "suspect" in results.columns
        else results
    )
    key = ["poll_id", "round", "hypothesis_id", "candidate_id"]
    reg = res.filter(pl.col("population") == "registered").select(
        *key, pl.col("share").alias("share_registered"), pl.col("raw_share").alias("raw_registered")
    )
    cert = res.filter(pl.col("population") == "certain_voters").select(
        *key, pl.col("share").alias("share_certain"), pl.col("raw_share").alias("raw_certain")
    )
    meta = ["poll_id", "election", "pollster", "fieldwork_start", "fieldwork_end", "is_rolling"]
    if "matched_poll_id" in polls.columns:
        meta.append("matched_poll_id")
    return (
        reg.join(cert, on=key, how="inner")
        .with_columns((pl.col("share_certain") - pl.col("share_registered")).alias("gap"))
        .join(polls.select(meta), on="poll_id", how="left")
    )


def attach_turnout_intent(gap: pl.DataFrame, intent: pl.DataFrame) -> pl.DataFrame:
    """Part de certains d'aller voter de l'échantillon : même tour, sinon premier tour."""
    it = intent.select(
        "poll_id",
        "round",
        pl.col("certain_share").alias("certain_share_sample"),
        pl.col("kind").alias("certain_share_kind"),
        "n_registered",
        "n_certain",
    ).unique(["poll_id", "round"], keep="first")
    same = gap.join(it, on=["poll_id", "round"], how="left")
    r1 = it.filter(pl.col("round") == 1).drop("round")
    fallback = (
        same.filter(pl.col("certain_share_sample").is_null())
        .drop("certain_share_sample", "certain_share_kind", "n_registered", "n_certain")
        .join(r1, on="poll_id", how="left")
    )
    return pl.concat(
        [same.filter(pl.col("certain_share_sample").is_not_null()), fallback],
        how="diagonal_relaxed",
    )


def build_table(
    sources: list[tuple[pl.DataFrame, pl.DataFrame]], intent: pl.DataFrame, bloc_of: BlocFn
) -> pl.DataFrame:
    parts = [pair_populations(polls, results) for polls, results in sources]
    parts = [p for p in parts if p.height]
    if not parts:
        return pl.DataFrame(schema=SCHEMA)
    gap = pl.concat(parts, how="diagonal_relaxed")
    gap = attach_turnout_intent(gap, intent)
    gap = gap.with_columns(
        pl.struct("candidate_id", "election")
        .map_elements(lambda s: bloc_of(s["candidate_id"], s["election"]), return_dtype=pl.Utf8)
        .alias("bloc")
    )
    for col in SCHEMA:
        if col not in gap.columns:
            gap = gap.with_columns(pl.lit(None).alias(col))
    return (
        gap.select(list(SCHEMA))
        .cast(dict(SCHEMA))
        .sort("election", "fieldwork_end", "poll_id", "round", "hypothesis_id", "candidate_id")
    )


def build(curated: Path = paths.CURATED, write: bool = True) -> pl.DataFrame:
    from vvs.registry import default_registry

    reg = default_registry()
    sources = []
    for polls_name, results_name in SOURCES:
        pp, rp = curated / polls_name, curated / results_name
        if pp.exists() and rp.exists():
            sources.append((pl.read_parquet(pp), pl.read_parquet(rp)))
    intent_path = curated / TURNOUT_INTENT
    intent = (
        pl.read_parquet(intent_path)
        if intent_path.exists()
        else pl.DataFrame(
            schema={
                "poll_id": pl.Utf8,
                "round": pl.Int8,
                "certain_share": pl.Float64,
                "kind": pl.Utf8,
                "n_registered": pl.Int32,
                "n_certain": pl.Int32,
            }
        )
    )
    table = build_table(sources, intent, reg.bloc_of)
    if write:
        table.write_parquet(curated / OUT_PATH.name)
    return table
