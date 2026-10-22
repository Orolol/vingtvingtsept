"""Cache parquet des étiquettes NLP : on ne reclassifie jamais un titre déjà classé.

Clé : (story_id, candidate_id, prompt_version, model). Chaque lot classé est écrit en ajout
seul sous data/raw/nlp/press_labels/part-<horodatage>-<n>.parquet (survit à un arrêt en
cours de route) ; `consolidate` rassemble les parts dans data/curated/press_labels.parquet.
Une réponse en erreur (schéma non respecté deux fois, serveur injoignable) est consignée
avec `error` et sans étiquette ; elle n'est pas comptée comme classée et repart au prochain
passage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from vvs import paths
from vvs.nlp.client import ChatClient, Item
from vvs.nlp.prompts import PromptSpec
from vvs.registry import default_registry

CACHE_DIR = paths.RAW / "nlp" / "press_labels"
OUTPUT_NAME = "press_labels.parquet"
KEY = ["story_id", "candidate_id", "prompt_version", "model"]

SCHEMA = pl.Schema({
    "story_id": pl.Utf8, "candidate_id": pl.Utf8, "prompt_version": pl.Utf8,
    "prompt_digest": pl.Utf8, "model": pl.Utf8,
    "tone": pl.Int8, "stance": pl.Utf8, "themes": pl.List(pl.Utf8),
    "error": pl.Utf8, "raw": pl.Utf8, "latency_s": pl.Float64,
    "prompt_tokens": pl.Int32, "completion_tokens": pl.Int32, "group_size": pl.Int32,
    "labeled_at": pl.Datetime("us"),
})


def load_cache(cache_dir: Path | None = None) -> pl.DataFrame:
    cache_dir = cache_dir or CACHE_DIR
    parts = sorted(cache_dir.glob("part-*.parquet")) if cache_dir.exists() else []
    if not parts:
        return pl.DataFrame(schema=SCHEMA)
    frames = []
    for p in parts:  # les parts anciennes peuvent manquer de colonnes ajoutées depuis
        df = pl.read_parquet(p)
        df = df.with_columns([pl.lit(None, dtype=t).alias(c) for c, t in SCHEMA.items()
                              if c not in df.columns])
        frames.append(df.select(list(SCHEMA)).cast(dict(SCHEMA)))
    return pl.concat(frames)


def done_keys(cache: pl.DataFrame, version: str, model: str) -> pl.DataFrame:
    return (cache.filter((pl.col("prompt_version") == version) & (pl.col("model") == model)
                         & pl.col("error").is_null())
            .select("story_id", "candidate_id").unique())


def to_items(titles: pl.DataFrame) -> list[Item]:
    """titres (story_id, candidate_id, title, publish_date, outlet) -> Items."""
    names = {cid: c.name for cid, c in default_registry().candidates.items()}
    return [Item(r["story_id"], r["candidate_id"], names.get(r["candidate_id"],
                                                               r["candidate_id"]),
                 r["title"], str(r["publish_date"]), r["outlet"] or "")
            for r in titles.iter_rows(named=True)]


def classify_missing(
    titles: pl.DataFrame,
    spec: PromptSpec,
    client: ChatClient,
    model: str,
    cache_dir: Path | None = None,
    chunk: int = 500,
    workers: int = 8,
    log: Callable[[str], None] = print,
) -> dict:
    """Classe les titres absents du cache pour (spec, model) ; bilan avec débit.

    Le débit est mesuré en titres classés par seconde d'horloge, lot par lot (temps passé à
    attendre le serveur, écriture comprise)."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    have = done_keys(load_cache(cache_dir), spec.version, model)
    todo = (titles.join(have, on=["story_id", "candidate_id"], how="anti")
            .sort("candidate_id", "publish_date", "story_id"))
    log(f"{spec.version} : {titles.height} titres, {titles.height - todo.height} déjà classés, "
        f"{todo.height} à classer")
    stats = {"todo": todo.height, "labeled": 0, "errors": 0, "seconds": 0.0, "chunks": []}
    for k in range(0, todo.height, chunk):
        part = todo.slice(k, chunk)
        items = to_items(part)
        t0 = time.perf_counter()
        labs = client.label_many(items, spec, workers=workers)
        dt_s = time.perf_counter() - t0
        now = datetime.now(UTC).replace(tzinfo=None)
        rows = [{"story_id": it.story_id, "candidate_id": it.candidate_id,
                 "prompt_version": spec.version, "prompt_digest": spec.digest, "model": model,
                 **lab, "labeled_at": now} for it, lab in zip(items, labs)]
        df = pl.DataFrame(rows, schema=SCHEMA)
        df.write_parquet(cache_dir / f"part-{now:%Y%m%dT%H%M%S%f}-{len(rows)}.parquet")
        n_err = df["error"].is_not_null().sum()
        stats["labeled"] += len(rows) - n_err
        stats["errors"] += n_err
        stats["seconds"] += dt_s
        stats["chunks"].append((len(rows), dt_s))
        log(f"  lot {k // chunk + 1} : {len(rows)} titres en {dt_s:.1f} s "
            f"({len(rows) / dt_s:.2f} titres/s), {n_err} erreurs ; "
            f"cumul {stats['labeled']}/{todo.height}")
    stats["rate"] = stats["labeled"] / stats["seconds"] if stats["seconds"] else float("nan")
    return stats


def consolidate(cache_dir: Path | None = None, write: bool = True) -> pl.DataFrame:
    """Parts du cache -> press_labels.parquet : une ligne par clé (la plus ancienne réussie),
    sans les erreurs ni le texte brut."""
    c = load_cache(cache_dir)
    df = (c.filter(pl.col("error").is_null()).sort("labeled_at")
          .unique(KEY, keep="first", maintain_order=True)
          .drop("error", "raw").sort(KEY))
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(paths.CURATED / OUTPUT_NAME)
    return df
