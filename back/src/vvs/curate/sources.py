"""Lecture des tables par source (data/curated) en `merge.SourcePoll`, et libellés rolling.

Sources lues : `polls_wikipedia_<AAAA>` / `poll_results_wikipedia_<AAAA>` (toutes les
élections présentes), `polls_commission` / `poll_results_commission`, `polls_mieuxvoter` /
`poll_results_mieuxvoter`.

Le drapeau rolling ne figure pas dans les tables par source ; il est relu :

- Wikipédia : libellé d'institut « Ifop (rolling) » des lignes du HTML brut (même moteur que
  l'ingestion, `vvs.ingest.wikipedia.engine`), rapproché du sondage par (institut, dates,
  échantillon) ;
- Commission : titre de la notice dans `data/raw/commission/index.jsonl` (« Presitrack ») ;
- MieuxVoter : nom du fichier de notice dans `source_url` ; le champ `rolling` du dépôt est
  vide partout (relevé du 20 octobre 2026).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from vvs import paths
from vvs.curate import rules
from vvs.curate.merge import Hyp, Row, SourcePoll


def to_source_polls(
    source: str, polls: pl.DataFrame, results: pl.DataFrame, rolling_ids: set[str] | None = None
) -> list[SourcePoll]:
    """Tables `polls` / `poll_results` d'une source -> `SourcePoll` (hypothèses regroupées)."""
    rolling_ids = rolling_ids or set()
    hyps: dict[str, dict] = {}
    for r in results.iter_rows(named=True):
        key = (r["round"], r["hypothesis_id"])
        by_poll = hyps.setdefault(r["poll_id"], {})
        h = by_poll.get(key)
        if h is None:
            h = by_poll[key] = Hyp(r["round"], r["hypothesis_id"], r["population"], {}, source)
        elif h.population != r["population"]:
            raise ValueError(
                f"{source} {r['poll_id']} : population incohérente dans l'hypothèse {key}"
            )
        if r["candidate_id"] in h.rows:
            raise ValueError(f"{source} {r['poll_id']} : {r['candidate_id']} en double dans {key}")
        h.rows[r["candidate_id"]] = Row(r["share"], r["share_base"], r["raw_share"], source)
    out = []
    for p in polls.iter_rows(named=True):
        if p["source"] != source:
            raise ValueError(f"{p['poll_id']} : source {p['source']!r}, attendu {source!r}")
        meta = {k: p[k] for k in (
            "sponsor", "fieldwork_start", "fieldwork_end", "sample_size", "publish_date", "mode",
            "source_url", "ingested_at",
        )}
        out.append(SourcePoll(
            source, p["poll_id"], p["election"], p["pollster"], meta,
            hyps.pop(p["poll_id"], {}), p["poll_id"] in rolling_ids,
        ))
    if hyps:
        raise ValueError(f"{source} : résultats sans sondage {sorted(hyps)[:5]}")
    return out


# ---------------------------------------------------------------------------
# Libellés rolling
# ---------------------------------------------------------------------------


def wikipedia_rolling_ids(election: int, polls: pl.DataFrame) -> set[str]:
    """Sondages Wikipédia d'une élection dont une ligne porte un libellé rolling."""
    from bs4 import BeautifulSoup

    from vvs.ingest.wikipedia import adapter_for
    from vvs.ingest.wikipedia.common import latest_raw, load_aliases
    from vvs.ingest.wikipedia.engine import ParseReport, expand_table, iter_tables, page_notes

    adapter = adapter_for(election)
    html = latest_raw(election).read_text(encoding="utf-8")
    aliases = load_aliases(election)
    soup = BeautifulSoup(html, "lxml")
    notes = page_notes(soup)
    report = ParseReport(election)
    keys: set[tuple] = set()
    for table, ctx in iter_tables(soup):
        round_ = adapter.table_round(ctx)
        if isinstance(round_, str):
            continue
        grid = expand_table(table, notes)
        for h in adapter.parse_table(table, ctx, round_, notes, aliases, report):
            row = int(h.where.rsplit(" l", 1)[1])
            label = grid[row][0].text if grid[row][0] is not None else ""
            if rules.is_rolling_label(label):
                keys.add((h.pollster, h.start, h.end, h.sample_size))
    ids: set[str] = set()
    rows = polls.select("poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size").rows()
    for pollster, start, end, sample in keys:
        same_dates = [r for r in rows if r[1:4] == (pollster, start, end)]
        exact = [r for r in same_dates if r[4] == sample]
        chosen = exact or (same_dates if len(same_dates) == 1 else [])
        ids.update(r[0] for r in chosen)
    return ids


def commission_rolling_ids(polls: pl.DataFrame, index_path: Path | None = None) -> set[str]:
    """Notices dont le titre porte « Presitrack » ou « rolling » (index de l'ingestion)."""
    index_path = index_path or paths.RAW / "commission" / "index.jsonl"
    titles = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            titles[str(e["file_id"])] = e.get("title") or ""
    ids = set()
    for pid, url in polls.select("poll_id", "source_url").rows():
        m = re.search(r"/add/(\d+)$", url or "")
        if m is None:
            raise ValueError(f"commission {pid} : source_url sans file_id ({url!r})")
        if rules.is_rolling_label(titles.get(m.group(1))):
            ids.add(pid)
    return ids


def mieuxvoter_rolling_ids(polls: pl.DataFrame) -> set[str]:
    return {
        pid for pid, url in polls.select("poll_id", "source_url").rows()
        if rules.is_rolling_label(url)
    }


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def wikipedia_elections(curated: Path | None = None) -> list[int]:
    curated = curated or paths.CURATED
    years = []
    for p in curated.glob("polls_wikipedia_*.parquet"):
        m = re.fullmatch(r"polls_wikipedia_(\d{4})\.parquet", p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def load_sources(curated: Path | None = None) -> dict[str, list[SourcePoll]]:
    curated = curated or paths.CURATED
    out: dict[str, list[SourcePoll]] = {"wikipedia": []}
    for year in wikipedia_elections(curated):
        polls = pl.read_parquet(curated / f"polls_wikipedia_{year}.parquet")
        results = pl.read_parquet(curated / f"poll_results_wikipedia_{year}.parquet")
        out["wikipedia"] += to_source_polls(
            "wikipedia", polls, results, wikipedia_rolling_ids(year, polls)
        )
    polls = pl.read_parquet(curated / "polls_commission.parquet")
    results = pl.read_parquet(curated / "poll_results_commission.parquet")
    out["commission"] = to_source_polls("commission", polls, results, commission_rolling_ids(polls))
    polls = pl.read_parquet(curated / "polls_mieuxvoter.parquet")
    results = pl.read_parquet(curated / "poll_results_mieuxvoter.parquet")
    out["mieuxvoter"] = to_source_polls("mieuxvoter", polls, results, mieuxvoter_rolling_ids(polls))
    return out
