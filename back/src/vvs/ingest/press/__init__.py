"""Ingestion de la presse, phase 7 (docs/06-presse.md) : téléchargement et mise en table,
sans classification NLP.

- `gdelt` : API DOC 2.0, timelines de volume et de ton par requête depuis 2017, titres jour
  par jour sur fenêtres ciblées -> press_timelines.parquet, press_items.parquet ;
- `ina` : baromètre thématique des JT du soir 2000-2020 (data.gouv) -> press_ina.parquet ;
- `mediacloud` : API search de Media Cloud (clé MEDIACLOUD_API_KEY), comptes par jour et
  listes de titres de la collection « France - National » -> press_items.parquet ;
- `outlets` : panel de titres (outlets.yaml) et requêtes (press_queries.yaml) ;
- `cache` : cache immuable des réponses sous data/raw/press/<source>/.

`parse_all()` reconstruit les tables curated à partir du seul contenu de data/raw/press.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from vvs import paths

ITEMS_OUTPUT = "press_items.parquet"

# docs/06-presse.md : item_id, source, outlet, published_at, title, url, lang ; on ajoute le
# domaine (outlet est nul hors panel) et les requêtes qui ont ramené l'article.
ITEMS_SCHEMA = pl.Schema(
    {
        "item_id": pl.Utf8,
        "source": pl.Utf8,
        "outlet": pl.Utf8,
        "domain": pl.Utf8,
        "published_at": pl.Datetime("us"),
        "title": pl.Utf8,
        "url": pl.Utf8,
        "lang": pl.Utf8,
        "query_ids": pl.List(pl.Utf8),
    }
)


def item_id(source: str, url: str) -> str:
    return hashlib.sha1(f"{source}:{url}".encode()).hexdigest()[:16]


def items_frame(rows: list[dict]) -> pl.DataFrame:
    """Lignes brutes (une par requête et article) -> une ligne par (source, url).

    `published_at` : date de publication (Media Cloud, au jour) ou de première collecte
    (GDELT `seendate`), en UTC sans fuseau.
    """
    if not rows:
        return pl.DataFrame(schema=ITEMS_SCHEMA)
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("published_at").cast(pl.Datetime("us"))
    )
    df = df.group_by("source", "url", maintain_order=True).agg(
        pl.col("outlet", "domain", "published_at", "title", "lang").first(),
        pl.col("query_id").unique().sort().alias("query_ids"),
    )
    df = df.with_columns(
        pl.struct("source", "url")
        .map_elements(lambda r: item_id(r["source"], r["url"]), return_dtype=pl.Utf8)
        .alias("item_id")
    )
    return df.select(list(ITEMS_SCHEMA)).cast(dict(ITEMS_SCHEMA)).sort("published_at", "url")


@dataclass
class PressParse:
    written: list[tuple[Path, int]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"écrit : {p} ({n} lignes)" for p, n in self.written]
        lines += [f"ignoré : {s}" for s in self.skipped]
        return "\n".join(lines)


def parse_all(sources: list[str] | None = None) -> PressParse:
    """data/raw/press -> data/curated (timelines GDELT, INA, titres GDELT + Media Cloud)."""
    from vvs.ingest.press import gdelt, ina, mediacloud
    from vvs.ingest.press.outlets import load_outlets, load_queries

    sources = sources or ["gdelt", "ina", "mediacloud"]
    queries, outlets = load_queries(), load_outlets()
    report = PressParse()
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    if "gdelt" in sources:
        df = gdelt.parse_timelines(queries, write=False)
        if df.height:
            df.write_parquet(paths.CURATED / gdelt.OUTPUT_NAME)
            report.written.append((paths.CURATED / gdelt.OUTPUT_NAME, df.height))
        else:
            report.skipped.append(
                "GDELT : aucune timeline en cache (lancer `vvs ingest press --source gdelt`)"
            )
    if "ina" in sources:
        try:
            out, df = ina.parse()
            report.written.append((out, df.height))
        except FileNotFoundError as e:
            report.skipped.append(f"INA : {e}")
    if "gdelt" in sources or "mediacloud" in sources:
        frames = []
        if "gdelt" in sources:
            frames.append(gdelt.parse_article_items(queries, outlets))
        if "mediacloud" in sources:
            frames.append(mediacloud.parse_story_items(queries, outlets))
        items = pl.concat(frames) if frames else pl.DataFrame(schema=ITEMS_SCHEMA)
        if items.height:
            out = paths.CURATED / ITEMS_OUTPUT
            items.sort("published_at", "url").write_parquet(out)
            report.written.append((out, items.height))
        else:
            report.skipped.append("titres : aucune réponse artlist ni story_list en cache")
    return report
