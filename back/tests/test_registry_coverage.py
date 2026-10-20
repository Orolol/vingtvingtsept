"""Couverture du registre : tout id de candidat ou d'institut rencontré dans les tables de
sondages (data/curated) et dans les tables d'alias existe dans data/registry, et le bloc de
chaque candidat est défini pour chaque élection où il est sondé."""

import polars as pl
import pytest
import yaml

from vvs import paths, schemas
from vvs.ingest.commission.aliases import CANDIDATES as COMMISSION_CANDIDATES
from vvs.ingest.commission.aliases import POLLSTER_ALIASES as COMMISSION_POLLSTERS
from vvs.registry import default_registry

# polls.parquet (fusion) et polls_<source>[_<élection>].parquet
POLL_TABLES = sorted(paths.CURATED.glob("polls*.parquet"))


def _results_path(polls_path):
    return polls_path.with_name(polls_path.name.replace("polls", "poll_results", 1))


@pytest.fixture(scope="module")
def reg():
    return default_registry()


@pytest.fixture(scope="module")
def sondages() -> pl.DataFrame:
    """(table, election, pollster, candidate_id) de toutes les tables de sondages."""
    if not POLL_TABLES:
        pytest.skip("aucune table de sondages dans data/curated")
    frames = []
    for p in POLL_TABLES:
        polls = pl.read_parquet(p, columns=["poll_id", "election", "pollster"])
        results = pl.read_parquet(_results_path(p), columns=["poll_id", "candidate_id"])
        frames.append(
            results.join(polls, on="poll_id", how="right")
            .select("election", "pollster", "candidate_id")
            .unique()
            .with_columns(pl.lit(p.name).alias("table"))
        )
    return pl.concat(frames)


def _yaml_ids(path, section):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))[section]
    out = set()

    def walk(x):
        if isinstance(x, dict):
            if "pollster" in x and "sponsor" in x:
                out.add(x["pollster"])
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if section == "candidates":
                return  # `ambiguous` : libellés, pas des ids
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.add(x)

    walk(data)
    return out


def test_poll_tables_candidates_in_registry(sondages, reg):
    ids = set(sondages["candidate_id"].drop_nulls()) - set(schemas.RESERVED_CANDIDATE_IDS)
    assert sorted(ids - set(reg.candidates)) == []


def test_poll_tables_pollsters_in_registry(sondages, reg):
    assert sorted(set(sondages["pollster"]) - set(reg.pollsters)) == []


@pytest.mark.parametrize("name", ["aliases_wikipedia.yaml", "aliases_mieuxvoter.yaml"])
def test_alias_files_in_registry(name, reg):
    path = paths.REGISTRY / name
    assert sorted(_yaml_ids(path, "candidates") - set(reg.candidates)) == []
    assert sorted(_yaml_ids(path, "pollsters") - set(reg.pollsters)) == []


def test_commission_aliases_in_registry(reg):
    assert sorted(set(COMMISSION_CANDIDATES) - set(reg.candidates)) == []
    assert sorted(set(COMMISSION_POLLSTERS.values()) - set(reg.pollsters)) == []


def test_bloc_defined_for_every_polled_candidate(sondages, reg):
    pairs = (
        sondages.filter(
            pl.col("candidate_id").is_not_null()
            & ~pl.col("candidate_id").is_in(schemas.RESERVED_CANDIDATE_IDS)
        )
        .select("candidate_id", "election")
        .unique()
        .rows()
    )
    failures = []
    for cid, election in sorted(pairs):
        try:
            reg.bloc_of(cid, election)
        except KeyError as e:
            failures.append(str(e))
    assert failures == []
