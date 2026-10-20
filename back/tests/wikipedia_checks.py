"""Outils partagés des tests Wikipédia (fixtures figées, invariants)."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from vvs.ingest.wikipedia import parse, parse_with_report
from vvs.ingest.wikipedia.common import fetch_page

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia"


def load_fixture(election: int) -> str:
    with gzip.open(FIXTURES / f"{election}.html.gz", "rt", encoding="utf-8") as f:
        return f.read()


def parse_fixture(election: int):
    return parse_with_report(election, load_fixture(election))


def shares(results: pl.DataFrame, poll_id: str, round_: int, hypothesis_id: str) -> dict:
    sub = results.filter(
        (pl.col("poll_id") == poll_id)
        & (pl.col("round") == round_)
        & (pl.col("hypothesis_id") == hypothesis_id)
    )
    return dict(zip(sub["candidate_id"], sub["share"], strict=True))


def counts_by_round(results: pl.DataFrame) -> dict[int, tuple[int, int, int]]:
    """tour -> (sondages, hypothèses, lignes)."""
    agg = results.group_by("round").agg(
        pl.col("poll_id").n_unique().alias("polls"),
        pl.struct("poll_id", "hypothesis_id").n_unique().alias("hyps"),
        pl.len().alias("rows"),
    )
    return {r: (p, h, n) for r, p, h, n in agg.iter_rows()}


def check_invariants(election: int, polls: pl.DataFrame, results: pl.DataFrame) -> None:
    assert polls["poll_id"].n_unique() == polls.height
    assert (polls["election"] == election).all()
    assert (polls["source"] == "wikipedia").all()
    assert polls.filter(pl.col("fieldwork_end") < pl.col("fieldwork_start")).height == 0
    assert polls["poll_id"].str.starts_with(f"{election}-").all()
    assert set(results["poll_id"]) == set(polls["poll_id"])
    # une ligne par (sondage, tour, hypothèse, candidat)
    key = ["poll_id", "round", "hypothesis_id", "candidate_id"]
    assert results.select(key).is_unique().all()
    # hypothesis_id = ids triés des candidats réellement présents
    hyp = results.filter(pl.col("share_base") == "expressed").group_by(
        "poll_id", "round", "hypothesis_id"
    )
    sums = hyp.agg(
        pl.col("share").sum().alias("total"),
        pl.col("candidate_id").filter(pl.col("candidate_id") != "other").sort().alias("ids"),
    )
    assert (sums["hypothesis_id"] == sums["ids"].list.join("+")).all()
    bad = sums.filter((pl.col("total") < 95) | (pl.col("total") > 105))
    assert bad.height == 0, bad
    assert results["share"].is_between(0, 100).all()
    # le second tour oppose exactement deux candidats
    r2 = sums.filter(pl.col("round") == 2)
    assert (r2["ids"].list.len() == 2).all()


def check_network(election: int) -> None:
    """Téléchargement réel et parsing de la page courante (test lent)."""
    polls, results = parse(election, fetch_page(election))
    check_invariants(election, polls, results)
    assert polls.height > 0
